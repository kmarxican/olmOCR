import torch
import base64
import urllib.request
import json  # Import the json module
import os  # Import os module for file path operations
import datetime
import hashlib

from io import BytesIO
from PIL import Image
# from transformers import AutoProcessor # <-- No longer directly using transformers AutoProcessor

# Use mlx_vlm's load to load both model and processor
from mlx_vlm import load, apply_chat_template, generate
from mlx_vlm.utils import load_image  # <-- Import load_image from mlx_vlm.utils

from olmocr.data.renderpdf import render_pdf_to_base64png
from olmocr.prompts import PageResponse, build_finetuning_prompt  # <--- PageResponse import (CORRECTED)
from olmocr.prompts.anchor import get_anchor_text
from olmocr.pipeline import PageResult  # Import PageResult dataclass
from dataclasses import dataclass  # Import dataclass (for PageResult definition)
from typing import Optional, List  # Import Optional and List (for PageResult definition)


@dataclass(frozen=True)  # Keep frozen=True if you want immutability
class PageResult:  # Modified PageResult definition with char_span
    s3_path: str
    page_num: int
    response: PageResponse
    input_tokens: int
    output_tokens: int
    is_fallback: bool
    char_span: Optional[List[int]] = None  # <--- ADD char_span field here, initialize to None


import PyPDF2  # Import PyPDF2 to get PDF page count

# Load model and processor using mlx_vlm.load (like the example)
model_path = "mlx-community/olmOCR-7B-0225-preview-4bit"
olmocr_model, olmocr_processor = load(model_path)  # Load both model and processor from mlx_vlm
olmocr_config = olmocr_model.config  # Get model config

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")  # Keep device check

# Grab a sample PDF (same as original)
pdf_filepath = "./paper.pdf"  # Define filepath for clarity
urllib.request.urlretrieve("https://molmo.allenai.org/paper.pdf", pdf_filepath)

# Get PDF page count using PyPDF2
with open(pdf_filepath, "rb") as pdf_file:
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    num_pages = len(pdf_reader.pages)
print(f"PDF has {num_pages} pages.")

page_results = []  # List to store PageResult objects (like in pipeline.py)
document_text = ""  # Accumulate document text
current_char_pos = 0  # Track character position (like in pipeline.py)

output_image_dir = "rendered_images"  # Directory to save rendered page images
os.makedirs(output_image_dir, exist_ok=True)  # Create directory if it doesn't exist

for page_num in range(1, num_pages + 1):  # Loop through all pages
    print(f"\n--- Processing Page {page_num} ---")

    # Render page to an image
    image_base64 = render_pdf_to_base64png(pdf_filepath, page_num, target_longest_image_dim=1024)
    main_image = Image.open(BytesIO(base64.b64decode(image_base64)))  # Load PIL Image

    # Save rendered image to a file
    image_filename = f"page_{page_num}.png"
    image_filepath = os.path.join(output_image_dir, image_filename)
    main_image.save(image_filepath)
    print(f"Saved rendered image to: {image_filepath}")

    # Build the prompt, using document metadata (same as original)
    anchor_text = get_anchor_text(pdf_filepath, page_num, pdf_engine="pdfreport", target_length=4000)
    prompt = build_finetuning_prompt(anchor_text)  # Original dynamic prompt

    # Build messages
    messages = [
        {"role": "user", "content": prompt},
    ]

    # Apply chat template
    print(f"apply_chat_template function: {apply_chat_template}")
    text_prompt = apply_chat_template(olmocr_processor, olmocr_config, messages)  # Pass messages list

    # Generate text
    page_generated_text = ""  # Store generated text for current page
    try:
        # Get tokenizer reference for easier use
        tokenizer = olmocr_processor.tokenizer

        print("\nStarting token generation for this page...")
        # Generate text and iterate over the tokens as they're generated
        for tokens in generate(
            olmocr_model,
            olmocr_processor,
            text_prompt,
            main_image,
            max_tokens=1024,  # Adjust max_tokens per page if needed
            temperature=0.7,
        ):

            # Handle different token types (using modified logic from Action 11b)
            chunk = ""
            if isinstance(tokens, str):
                chunk = tokens
            elif hasattr(tokens, "tolist"):
                tokens = tokens.tolist()
                if not all(isinstance(t, int) for t in tokens):
                    tokens = [int(t) for t in tokens if str(t).strip()]
                chunk = tokenizer.decode(tokens, skip_special_tokens=True)

            if not chunk:
                continue

            page_generated_text += chunk
            print(chunk, end="", flush=True)

    except Exception as e:
        print(f"\nError during generation for page {page_num}: {e}")
        import traceback

        traceback.print_exc()

    # --- Create PageResult object (like in pipeline.py) - CORRECTED char_span ASSIGNMENT ---
    from olmocr.pipeline import PageResult  # Import PageResult dataclass

    page_result = PageResult(  # Create PageResult instance, setting char_span NOW
        s3_path=pdf_filepath,  # Use pdf_filepath as s3_path for now
        page_num=page_num,
        response=PageResponse(
            natural_text=page_generated_text.strip(),
            primary_language="en",
            is_rotation_valid=True,
            rotation_correction=0,
            is_table=False,
            is_diagram=False,
        ),  # Create PageResponse
        input_tokens=0,  # Dummy values for now
        output_tokens=0,  # Dummy values for now
        is_fallback=False,  # Not fallback for now
        char_span=[start_pos, current_char_pos],  # <--- Set char_span HERE during object creation!
    )
    page_results.append(page_result)

    start_pos = current_char_pos  # Track start position
    # --- Enforce single newlines here ---
    page_generated_text_single_newlines = page_generated_text.strip().replace("\n\n", "\n")  # Replace double newlines with single
    document_text += page_generated_text_single_newlines + (
        "\n" if page_num < num_pages else ""
    )  # Accumulate text with single newlines
    current_char_pos = len(document_text)  # Update current character position


print("\n\n--- Generation Complete for All Pages ---")

# --- Build Dolma Document JSON (Corrected pdf_page_numbers format) ---
metadata = {  # Simplified metadata - just source file and page count
    "Source-File": pdf_filepath,
    "pdf-total-pages": num_pages,
}
id_ = hashlib.sha1(document_text.encode()).hexdigest()  # Generate document ID
dolma_doc = {  # Create Dolma document JSON
    "id": id_,
    "text": document_text.strip(),  # <--- Output ONLY document_text as "text" content (CORRECTED)
    "source": "olmocr",
    "added": datetime.datetime.now().strftime("%Y-%m-%d"),
    "created": datetime.datetime.now().strftime("%Y-%m-%d"),
    "metadata": metadata,
    "attributes": {
        "pdf_page_numbers": [
            [res.char_span[0], res.char_span[1], res.page_num] for res in page_results
        ],  # Corrected format!
    },
}

print("\nDolma Document JSON (Corrected pdf_page_numbers format):")
print(dolma_doc)  # Print the Dolma document JSON

# Save to JSON Lines file (single Dolma document JSON object)
try:
    with open("generated_output.jsonl", "w") as f:
        json.dump(dolma_doc, f)  # Write the single Dolma document JSON object
        f.write("\n")  # Add newline (for JSON Lines format - single line file)
    print(
        "\nGenerated text saved to 'generated_output.jsonl' (JSON Lines format, single Dolma document, corrected pdf_page_numbers)"
    )
except Exception as save_error:
    print(f"\nError saving output to file: {save_error}")