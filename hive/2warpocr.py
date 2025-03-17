import torch
import base64
import urllib.request

from io import BytesIO
from PIL import Image
# from transformers import AutoProcessor # <-- No longer directly using transformers AutoProcessor

# Use mlx_vlm's load to load both model and processor
from mlx_vlm import load, apply_chat_template, generate
from mlx_vlm.utils import load_image # <-- Import load_image from mlx_vlm.utils

from olmocr.data.renderpdf import render_pdf_to_base64png
from olmocr.prompts import build_finetuning_prompt
from olmocr.prompts.anchor import get_anchor_text

import PyPDF2 # Import PyPDF2 to get PDF page count

# Load model and processor using mlx_vlm.load (like the example)
model_path = "mlx-community/olmOCR-7B-0225-preview-4bit"
olmocr_model, olmocr_processor = load(model_path) # Load both model and processor from mlx_vlm
olmocr_config = olmocr_model.config # Get model config

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu") # Keep device check

# Grab a sample PDF (same as original)
pdf_filepath = "./paper.pdf" # Define filepath for clarity
urllib.request.urlretrieve("https://molmo.allenai.org/paper.pdf", pdf_filepath)

# Get PDF page count using PyPDF2
with open(pdf_filepath, 'rb') as pdf_file:
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    num_pages = len(pdf_reader.pages)
print(f"PDF has {num_pages} pages.")

full_ocr_text = "" # Initialize an empty string to store full OCR text

for page_num in range(1, num_pages + 1): # Loop through all pages
    print(f"\n--- Processing Page {page_num} ---")

    # Render page to an image
    image_base64 = render_pdf_to_base64png(pdf_filepath, page_num, target_longest_image_dim=1024)
    main_image = Image.open(BytesIO(base64.b64decode(image_base64))) # Load PIL Image

    # Build the prompt, using document metadata (same as original)
    anchor_text = get_anchor_text(pdf_filepath, page_num, pdf_engine="pdfreport", target_length=4000)
    prompt = build_finetuning_prompt(anchor_text) # Original dynamic prompt

    # Build messages
    messages = [
        {"role": "user", "content": prompt},
    ]

    # Apply chat template
    print(f"apply_chat_template function: {apply_chat_template}")
    text_prompt = apply_chat_template(olmocr_processor, olmocr_config, messages) # Pass messages list

    # Debug prints (optional, but helpful for monitoring progress)
    print(f"text_prompt type: {type(text_prompt)}")
    print(f"text_prompt content (first 100 chars): {str(text_prompt)[:100]}...")
    print("Starting generation for this page...")


    # Generate text
    page_generated_text = "" # Store generated text for current page
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
            max_tokens=1024, # Adjust max_tokens per page if needed
            temperature=0.7,
        ):

            # Handle different token types (using modified logic from Action 11b)
            chunk = ""
            if isinstance(tokens, str):
                chunk = tokens
            elif hasattr(tokens, 'tolist'):
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

    full_ocr_text += page_generated_text # Append page's text to full text


print("\n\n--- Generation Complete for All Pages ---")
print("\nFull generated text (all pages):")
print(full_ocr_text)

# Save the full generated text to file for inspection
try:
    with open("generated_output_full_pdf.txt", "w") as f:
        f.write(full_ocr_text)
    print("\nGenerated text for full PDF saved to 'generated_output_full_pdf.txt'")
except Exception as save_error:
    print(f"\nError saving full PDF output to file: {save_error}")