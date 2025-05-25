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
import subprocess
import signal
import logging
import sys
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from PIL import Image
from typing import List, Tuple, Dict, Optional, Union
from tqdm import tqdm

# MLX-VLM and OLMOCR imports
from mlx_vlm import load, apply_chat_template, generate
from mlx_vlm.utils import load_image
from functools import partial

def safe_decode(tokenizer_decode_method, *args, **kwargs):
    """Safe wrapper for tokenizer decode that handles UTF-8 errors."""
    try:
        return tokenizer_decode_method(*args, **kwargs)
    except UnicodeDecodeError:
        # Get the raw bytes and decode with replacement
        raw_result = tokenizer_decode_method(*args, **kwargs, skip_special_tokens=True)
        if isinstance(raw_result, bytes):
            return raw_result.decode('utf-8', errors='replace')
        return str(raw_result)
from olmocr.data.renderpdf import render_pdf_to_base64png
from olmocr.prompts import PageResponse, build_finetuning_prompt
from olmocr.prompts.anchor import get_anchor_text
from olmocr.pipeline import PageResult, build_dolma_document
from olmocr.viewer.dolmaviewer import main as dolmaviewer_main
import PyPDF2  # For PDF page count

# Configuration with improved defaults
MODEL_PATH = "mlx-community/olmOCR-7B-0225-preview-4bit"
MAX_TOKENS_PER_PAGE = 8192  # Further increased for better coverage
TEMPERATURE = 0.2  # Further reduced for more consistent text block recognition
TARGET_LONGEST_IMAGE_DIM = 1024  # Optimal resolution for text block recognition
MAX_WORKERS = 4  # For parallel processing

# Set device
# Device initialization with debug logging
try:
    if torch.backends.mps.is_available():
        print(f"PyTorch {torch.__version__} MPS backend available")
        device = torch.device("mps")
        print(f"Using MPS device: {torch.mps.current_allocated_memory()} MB allocated")
    else:
        device = torch.device("cpu")
        print("MPS not available, using CPU")
except Exception as e:
    print(f"Error initializing device: {e}")
    raise

class OlmOCRProcessor:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.model is not None:
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            del self.model
            del self.processor
            self.model = None
            self.processor = None
            print("Released model resources")
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
        # Memory debug logging
        if torch.backends.mps.is_available():
            print(f"Pre-load memory: {torch.mps.current_allocated_memory()}MB allocated, {torch.mps.driver_allocated_memory()}MB reserved")
        """Load OLMOCR model and processor."""
        print(f"Loading model from {self.model_path}...")
        start_time = time.time()
        self.model, self.processor = load(self.model_path)
        print(f"Model loaded in {time.time() - start_time:.2f} seconds")
        if torch.backends.mps.is_available():
            print(f"Post-load memory: {torch.mps.current_allocated_memory()}MB allocated, {torch.mps.driver_allocated_memory()}MB reserved")
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

            # Apply safe decode patch to tokenizer
            original_decode = self.processor.tokenizer.decode
            self.processor.tokenizer.decode = partial(safe_decode, original_decode)
            
            # Build the prompt with appropriate context
            anchor_text = get_anchor_text(pdf_path, page_num, pdf_engine="pdfreport", target_length=4000)
            prompt = build_finetuning_prompt(anchor_text)
            logging.info(f"Prompt built for page {page_num}")
            
            # Apply chat template using the processor
            # Ensure UTF-8 compatibility of the prompt
            safe_prompt = prompt.encode('utf-8', errors='replace').decode('utf-8')
            messages = [{"role": "user", "content": safe_prompt}]
            text_prompt = apply_chat_template(self.processor, self.model.config, messages)
            # Ensure the chat template output is also UTF-8 safe
            text_prompt = text_prompt.encode('utf-8', errors='replace').decode('utf-8')

            # Generate text
            logging.info(f"=== Starting processing of page {page_num} ===")
            logging.info(f"Anchor text length: {len(anchor_text)}")
            logging.info(f"Prompt text length: {len(prompt)}")
            
            page_generated_text = ""
            tokenizer = self.processor.tokenizer

            print(f"\nDEBUG: Starting generation for page {page_num}")
            logging.info("Input validation:")
            logging.info(f"Anchor text sample: {anchor_text[:200] if anchor_text else 'None'}")
            logging.info(f"Image dimensions: {main_image.size if main_image else 'None'}")
            print(f"DEBUG: Initial prompt encoding: {text_prompt.encode('utf-8')}")
            
            for tokens in generate(
                self.model, self.processor, text_prompt, main_image,
                max_tokens=self.max_tokens, temperature=self.temperature
            ):
                try:
                    if isinstance(tokens, str):
                        chunk = tokens
                    else:
                        token_list = tokens.tolist() if hasattr(tokens, 'tolist') else [tokens]
                        # Use error replacement for both encode and decode operations
                        chunk = tokenizer.decode(token_list, skip_special_tokens=True)
                        chunk = chunk.encode('utf-8', errors='replace').decode('utf-8', errors='replace')
                except Exception as e:
                    logging.error(f"Token processing error: {str(e)}")
                    chunk = "�"
                
                if chunk:
                    page_generated_text += chunk

            # Clean excessive newlines
            page_generated_text = re.sub(r'(\n\s*){2,}', '\n\n', page_generated_text)
            # Clean ampersand artifacts like &86 (e.g., &86, &123;)
            page_generated_text = re.sub(r'&\d{2,};?', '', page_generated_text)

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
                   parallel: bool = True, max_pages: Optional[int] = None,
                   start_page: int = 1) -> Tuple[Dict, List[PageResult]]:
        """
        Processes an entire PDF file and returns structured extracted data.
        
        Args:
            pdf_path: Path to the PDF file
            output_dir: Directory to save intermediate results (None = don't save)
            parallel: Whether to process pages in parallel
            
        Returns:
            Tuple of (document dictionary, list of PageResult objects)
        """
        if not self.model or not self.processor:
            self.load_model()
            
        # Create output directory if specified
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        # Get page count and validate page range
        total_pages = self.get_pdf_page_count(pdf_path)
        if start_page > total_pages:
            raise ValueError(f"Start page {start_page} is greater than total pages {total_pages}")
            
        effective_pages = total_pages - start_page + 1
        num_pages = min(effective_pages, max_pages) if max_pages else effective_pages
        print(f"PDF has {total_pages} pages. Processing {num_pages} pages starting from page {start_page}...")

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
                    for page_num in range(start_page, start_page + num_pages)
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
                for page_num in range(start_page, start_page + num_pages):
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
            for page_num in tqdm(range(start_page, start_page + num_pages), desc="Processing pages"):
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
        
        document_data = {
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
        
        return document_data, page_results

    def save_dolma_output(self, pdf_path: str, page_results: List[PageResult], output_dir: str = "results") -> str:
        """
        Saves processed PDF results in Dolma format for viewing with dolmaviewer.
        
        Args:
            pdf_path: Path to the processed PDF file
            page_results: List of PageResult objects from processing
            output_dir: Directory to save the Dolma output
            
        Returns:
            Path to the saved Dolma JSONL file
        """
        # Create output directory
        output_folder = Path(output_dir)
        output_folder.mkdir(exist_ok=True, parents=True)
        
        # Build Dolma document
        dolma_doc = build_dolma_document(pdf_path, page_results)
        
        # Save as JSONL
        pdf_name = os.path.basename(pdf_path).replace('.pdf', '')
        dolma_path = output_folder.joinpath(f"{pdf_name}.results.jsonl")
        
        with open(dolma_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(dolma_doc) + "\n")
            
        print(f"Dolma document saved to {dolma_path}")
        return str(dolma_path)
    
    def generate_html_preview(self, dolma_paths: List[str], output_dir: str = "dolma_previews") -> str:
        """
        Generates HTML previews for Dolma documents using dolmaviewer.
        
        Args:
            dolma_paths: List of paths to Dolma JSONL files
            output_dir: Directory to save HTML previews
            
        Returns:
            Path to the output directory with HTML previews
        """
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Call dolmaviewer to generate HTML previews
        try:
            # Method 1: Use subprocess to call the command
            cmd = ["python", "-m", "olmocr.viewer.dolmaviewer"] + dolma_paths + ["--output-dir", output_dir]
            subprocess.run(cmd, check=True)
            print(f"HTML previews generated in {output_dir}")
            
            # List generated HTML files
            html_files = [f for f in os.listdir(output_dir) if f.endswith('.html')]
            if html_files:
                print("Generated HTML files:")
                for html_file in html_files:
                    print(f"  - {os.path.join(output_dir, html_file)}")
            
            return output_dir
        except subprocess.CalledProcessError as e:
            print(f"Error generating HTML previews: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error generating HTML previews: {e}")
            traceback.print_exc()
            return None


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


# Set up logging
logging.basicConfig(
    filename='olmocr-pipeline-debug.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def signal_handler(signum, frame):
    """Handle interrupt signal by printing debug info and exiting gracefully."""
    logging.warning("\nInterrupt received. Printing debug info before exit...")
    logging.warning("Last known execution frame:")
    logging.warning(f"File: {frame.f_code.co_filename}")
    logging.warning(f"Function: {frame.f_code.co_name}")
    logging.warning(f"Line: {frame.f_lineno}")
    sys.exit(1)

def main():
    """Main function to process PDFs with command line arguments."""
    # Set up signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    parser = argparse.ArgumentParser(description="Extract text from PDF files using olmOCR")
    parser.add_argument("--pdf", type=str, help="Path to PDF file or URL to download")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory for results")
    parser.add_argument("--model", type=str, default=MODEL_PATH, help="Model path or name")
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS_PER_PAGE, help="Max tokens per page")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE, help="Generation temperature")
    parser.add_argument("--image-dim", type=int, default=TARGET_LONGEST_IMAGE_DIM, help="Target image dimension")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS, help="Number of worker threads")
    parser.add_argument("--sequential", action="store_true", help="Process pages sequentially")
    parser.add_argument("--intermediate", type=str, help="Directory to save intermediate results")
    parser.add_argument("--preview", action="store_true", help="Generate HTML previews with dolmaviewer")
    parser.add_argument("--preview-dir", type=str, default="dolma_previews", help="Directory for HTML previews")
    parser.add_argument("--max-pages", type=int, help="Maximum number of pages to process")
    parser.add_argument("--start-page", type=int, default=1, help="Page number to start processing from")
    
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
    with OlmOCRProcessor(
        model_path=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        image_dim=args.image_dim,
        max_workers=args.workers
    ) as processor:
        start_time = time.time()
    
    # Process PDF
    start_time = time.time()
    print(f"Processing PDF: {pdf_path}")
    
    document_data, page_results = processor.process_pdf(
        pdf_path, 
        output_dir=args.intermediate,
        parallel=not args.sequential,
        max_pages=args.max_pages,
        start_page=args.start_page
    )
    
    # Save in Dolma format
    dolma_path = processor.save_dolma_output(pdf_path, page_results, args.output_dir)
    
    # Generate HTML preview if requested
    if args.preview:
        preview_dir = processor.generate_html_preview([dolma_path], args.preview_dir)
        if preview_dir:
            print(f"HTML previews available in: {preview_dir}")
            # Try to open the preview in browser if possible
            html_file = os.path.join(preview_dir, os.path.basename(pdf_path).replace('.pdf', '.html'))
            if os.path.exists(html_file):
                try:
                    import webbrowser
                    webbrowser.open(f"file://{os.path.abspath(html_file)}")
                except:
                    pass
    
    # Print summary
    # Force memory cleanup after processing
    torch.mps.empty_cache()
    print(f"\nExtraction completed in {time.time() - start_time:.2f} seconds")
    print(f"Total text length: {len(document_data['text'])} characters")
    print(f"Success rate: {document_data['attributes']['success_rate']}")
    print(f"Dolma output saved to: {dolma_path}")


if __name__ == "__main__":
    main()
