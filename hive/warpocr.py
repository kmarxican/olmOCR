import torch
import base64
import urllib.request
import json
import os
import datetime
import hashlib
import traceback
import argparse
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from PIL import Image
from typing import List, Tuple, Dict, Optional, Union
from tqdm import tqdm

# MLX-VLM and OLMOCR imports
from mlx_vlm import load, apply_chat_template, generate
from mlx_vlm.utils import load_image
from mlx_vlm.tokenizer_utils import Detokenizer
from olmocr.data.renderpdf import render_pdf_to_base64png
from olmocr.prompts import PageResponse, build_finetuning_prompt

def safe_generate(*args, **kwargs):
    """Wrapper around generate() that handles UTF-8 decoding errors"""
    try:
        # Patch the Detokenizer class to handle UTF-8 errors
        original_decode = Detokenizer.decode
        def safe_decode(self, encoding="utf-8"):
            try:
                return original_decode(self, encoding)
            except UnicodeDecodeError:
                # Try to salvage partial content with 'replace' error handler
                return self.buffer.decode(encoding, errors='replace')
        Detokenizer.decode = safe_decode
        
        yield from generate(*args, **kwargs)
    finally:
        # Restore original decode method
        Detokenizer.decode = original_decode
from olmocr.prompts.anchor import get_anchor_text
from olmocr.pipeline import PageResult
import PyPDF2  # For PDF page count

# Configuration with improved defaults
MODEL_PATH = "mlx-community/olmOCR-7B-0225-preview-4bit"
MAX_TOKENS_PER_PAGE = 8192  # Further increased for better coverage
TEMPERATURE = 0.5  # Reduced for more deterministic output
TARGET_LONGEST_IMAGE_DIM = 1536  # Higher resolution for better OCR
MAX_WORKERS = 4  # For parallel processing

# Set device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

class OlmOCRProcessor:
    """Class to handle PDF processing with olmOCR model."""
    
    def __init__(self, model_path: str = MODEL_PATH, max_tokens: int = MAX_TOKENS_PER_PAGE, 
                 temperature: float = TEMPERATURE, image_dim: int = TARGET_LONGEST_IMAGE_DIM,
                 max_workers: int = MAX_WORKERS):
        """Initialize the processor with configurable parameters."""
        self.model_path = model_path
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.image_dim = image_dim
        self.max_workers = max_workers
        self.model = None
        self.processor = None
        
    def load_model(self):
        """Load OLMOCR model and processor."""
        print(f"Loading model from {self.model_path}...")
        start_time = time.time()
        self.model, self.processor = load(self.model_path)
        print(f"Model loaded in {time.time() - start_time:.2f} seconds")
        return self.model, self.processor

    def get_pdf_page_count(self, pdf_path: str) -> int:
        """Get the number of pages in a PDF file."""
        try:
            with open(pdf_path, 'rb') as pdf_file:
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                return len(pdf_reader.pages)
        except Exception as e:
            print(f"Error reading PDF: {e}")
            raise

    def process_pdf_page(self, pdf_path: str, page_num: int) -> Tuple[str, PageResult]:
        """Processes a single PDF page and returns extracted text and PageResult."""
        if not self.model or not self.processor:
            raise ValueError("Model not loaded. Call load_model() first.")
            
        try:
            # Render the page as an image
            image_base64 = render_pdf_to_base64png(pdf_path, page_num, target_longest_image_dim=self.image_dim)
            main_image = Image.open(BytesIO(base64.b64decode(image_base64)))

            # Build the prompt with appropriate context
            anchor_text = get_anchor_text(pdf_path, page_num, pdf_engine="pdfreport", target_length=4000)
            prompt = build_finetuning_prompt(anchor_text)
            
            # Apply chat template using the processor
            messages = [{"role": "user", "content": prompt}]
            text_prompt = apply_chat_template(self.processor, self.model.config, messages)

            # Generate text
            page_generated_text = ""
            tokenizer = self.processor.tokenizer

            for tokens in safe_generate(
                self.model, self.processor, text_prompt, main_image,
                max_tokens=self.max_tokens, temperature=self.temperature
            ):
                if isinstance(tokens, str):
                    chunk = tokens
                else:
                    chunk = tokenizer.decode(tokens.tolist(), skip_special_tokens=True)
                if chunk:
                    page_generated_text += chunk

            # Create PageResult object
            page_result = PageResult(
                s3_path=pdf_path,
                page_num=page_num,
                response=PageResponse(
                    natural_text=page_generated_text.strip(), 
                    primary_language="en",
                    is_rotation_valid=True, 
                    rotation_correction=0, 
                    is_table=False, 
                    is_diagram=False
                ),
                input_tokens=len(text_prompt),
                output_tokens=len(page_generated_text),
                is_fallback=False,
            )

            return page_generated_text.strip(), page_result

        except Exception as e:
            error_msg = f"Error processing page {page_num}: {str(e)}"
            print(f"\n{error_msg}")
            traceback.print_exc()
            
            # Return error information in a structured way
            page_result = PageResult(
                s3_path=pdf_path,
                page_num=page_num,
                response=PageResponse(
                    natural_text=f"[ERROR: {error_msg}]", 
                    primary_language="en",
                    is_rotation_valid=False, 
                    rotation_correction=0, 
                    is_table=False, 
                    is_diagram=False
                ),
                input_tokens=0,
                output_tokens=0,
                is_fallback=True,
            )
            
            return f"[ERROR: Unable to process page {page_num}]", page_result

    def process_pdf(self, pdf_path: str, output_dir: Optional[str] = None, 
                   parallel: bool = True) -> Dict:
        """
        Processes an entire PDF file and returns structured extracted data.
        
        Args:
            pdf_path: Path to the PDF file
            output_dir: Directory to save intermediate results (None = don't save)
            parallel: Whether to process pages in parallel
            
        Returns:
            Dictionary with structured extracted data
        """
        if not self.model or not self.processor:
            self.load_model()
            
        # Create output directory if specified
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        # Get page count
        num_pages = self.get_pdf_page_count(pdf_path)
        print(f"PDF has {num_pages} pages. Beginning extraction...")

        document_text = ""
        page_results = []
        char_spans = []
        current_char_pos = 0
        
        # Process pages
        if parallel and num_pages > 1 and self.max_workers > 1:
            # Parallel processing
            with ThreadPoolExecutor(max_workers=min(self.max_workers, num_pages)) as executor:
                # Submit all tasks
                future_to_page = {
                    executor.submit(self.process_pdf_page, pdf_path, page_num): page_num 
                    for page_num in range(1, num_pages + 1)
                }
                
                # Process results as they complete
                results = {}
                for future in tqdm(as_completed(future_to_page), total=num_pages, desc="Processing pages"):
                    page_num = future_to_page[future]
                    try:
                        page_text, page_result = future.result()
                        results[page_num] = (page_text, page_result)
                    except Exception as e:
                        print(f"\nError processing page {page_num}: {e}")
                        results[page_num] = (f"[ERROR: Failed to process page {page_num}]", None)
                
                # Combine results in correct order
                for page_num in range(1, num_pages + 1):
                    if page_num in results:
                        page_text, page_result = results[page_num]
                        
                        # Save intermediate results if requested
                        if output_dir:
                            page_output_file = os.path.join(output_dir, f"page_{page_num:04d}.txt")
                            with open(page_output_file, "w", encoding="utf-8") as f:
                                f.write(page_text)
                        
                        # Track character spans
                        start_pos = current_char_pos
                        document_text += page_text + ("\n" if page_num < num_pages else "")
                        current_char_pos = len(document_text)
                        
                        if page_result:
                            page_results.append(page_result)
                            char_spans.append([start_pos, current_char_pos])
        else:
            # Sequential processing
            for page_num in tqdm(range(1, num_pages + 1), desc="Processing pages"):
                page_text, page_result = self.process_pdf_page(pdf_path, page_num)
                
                # Save intermediate results if requested
                if output_dir:
                    page_output_file = os.path.join(output_dir, f"page_{page_num:04d}.txt")
                    with open(page_output_file, "w", encoding="utf-8") as f:
                        f.write(page_text)
                
                # Track character spans
                start_pos = current_char_pos
                document_text += page_text + ("\n" if page_num < num_pages else "")
                current_char_pos = len(document_text)
                
                page_results.append(page_result)
                char_spans.append([start_pos, current_char_pos])

        # Build structured output
        doc_id = hashlib.sha1(document_text.encode()).hexdigest()
        pdf_filename = os.path.basename(pdf_path)
        
        metadata = {
            "source_file": pdf_path,
            "filename": pdf_filename,
            "pdf_total_pages": num_pages,
            "extraction_date": datetime.datetime.now().isoformat(),
            "model": self.model_path,
            "image_resolution": self.image_dim
        }
        
        return {
            "id": doc_id,
            "text": document_text.strip(),
            "source": "olmocr",
            "added": datetime.datetime.now().strftime("%Y-%m-%d"),
            "created": datetime.datetime.now().strftime("%Y-%m-%d"),
            "metadata": metadata,
            "attributes": {
                "pdf_page_numbers": [[span[0], span[1], res.page_num] for span, res in zip(char_spans, page_results)],
                "success_rate": f"{len([r for r in page_results if not r.is_fallback])}/{num_pages}",
                "extraction_quality": "high" if len([r for r in page_results if not r.is_fallback]) == num_pages else "partial"
            }
        }

    def save_output(self, output_data: Dict, output_path: str = "generated_output.json"):
        """Saves structured extracted text data to a file."""
        try:
            # Determine output format based on extension
            if output_path.endswith('.jsonl'):
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(output_data, f)
                    f.write('\n')
            else:
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(output_data, f, indent=2, ensure_ascii=False)
                    
            print(f"\nGenerated text saved to '{output_path}'")
            return True
        except Exception as e:
            print(f"\nError saving output to file: {e}")
            traceback.print_exc()
            return False


def download_pdf(url: str, filepath: str) -> bool:
    """Download a PDF file from a URL."""
    try:
        print(f"Downloading PDF from {url}...")
        urllib.request.urlretrieve(url, filepath)
        print(f"Downloaded to {filepath}")
        return True
    except Exception as e:
        print(f"Error downloading PDF: {e}")
        return False


def main():
    """Main function to process PDFs with command line arguments."""
    parser = argparse.ArgumentParser(description="Extract text from PDF files using olmOCR")
    parser.add_argument("--pdf", type=str, help="Path to PDF file or URL to download")
    parser.add_argument("--output", type=str, default="output.json", help="Output file path")
    parser.add_argument("--model", type=str, default=MODEL_PATH, help="Model path or name")
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS_PER_PAGE, help="Max tokens per page")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE, help="Generation temperature")
    parser.add_argument("--image-dim", type=int, default=TARGET_LONGEST_IMAGE_DIM, help="Target image dimension")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS, help="Number of worker threads")
    parser.add_argument("--sequential", action="store_true", help="Process pages sequentially")
    parser.add_argument("--intermediate", type=str, help="Directory to save intermediate results")
    
    args = parser.parse_args()
    
    # Use sample PDF if none provided
    pdf_path = args.pdf
    if not pdf_path:
        pdf_url = "https://molmo.allenai.org/paper.pdf"
        pdf_path = "./paper.pdf"
        if not os.path.exists(pdf_path):
            if not download_pdf(pdf_url, pdf_path):
                return
    elif pdf_path.startswith(('http://', 'https://')):
        # Download PDF if URL provided
        local_path = os.path.basename(pdf_path)
        if not download_pdf(pdf_path, local_path):
            return
        pdf_path = local_path
    
    # Initialize processor with arguments
    processor = OlmOCRProcessor(
        model_path=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        image_dim=args.image_dim,
        max_workers=args.workers
    )
    
    # Process PDF
    start_time = time.time()
    extracted_data = processor.process_pdf(
        pdf_path, 
        output_dir=args.intermediate,
        parallel=not args.sequential
    )
    
    # Save output
    processor.save_output(extracted_data, args.output)
    
    # Print summary
    print(f"\nExtraction completed in {time.time() - start_time:.2f} seconds")
    print(f"Total text length: {len(extracted_data['text'])} characters")
    print(f"Success rate: {extracted_data['attributes']['success_rate']}")
    print(f"Output saved to: {args.output}")


if __name__ == "__main__":
    main()
