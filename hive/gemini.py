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

# Load model and processor using mlx_vlm.load (like the example)
model_path = "mlx-community/olmOCR-7B-0225-preview-4bit"
olmocr_model, olmocr_processor = load(model_path) # Load both model and processor from mlx_vlm
olmocr_config = olmocr_model.config # Get model config

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu") # Keep device check

# Grab a sample PDF (same as original)
urllib.request.urlretrieve("https://molmo.allenai.org/paper.pdf", "./paper.pdf")

# Render page 1 to an image (same as original)
image_base64 = render_pdf_to_base64png("./paper.pdf", 1, target_longest_image_dim=1024)
main_image = Image.open(BytesIO(base64.b64decode(image_base64))) # Load PIL Image

# Build the prompt, using document metadata (same as original)
anchor_text = get_anchor_text("./paper.pdf", 1, pdf_engine="pdfreport", target_length=4000)
prompt = build_finetuning_prompt(anchor_text) # Original dynamic prompt

# Build messages
messages = [
    {"role": "user", "content": prompt},
]


# Apply chat template (passing prompt string directly - simplified input)
print(f"apply_chat_template function: {apply_chat_template}")
text_prompt = apply_chat_template(olmocr_processor, olmocr_config, messages) # Pass messages list

# Debug prints
print(f"text_prompt type: {type(text_prompt)}")
print(f"text_prompt content (first 100 chars): {str(text_prompt)[:100]}...")

# Ensure text_prompt is a string
if text_prompt is None:
    print("Warning: text_prompt is None, using an empty string instead")
    text_prompt = ""
elif not isinstance(text_prompt, str):
    text_prompt = str(text_prompt)

print("Starting generation...")

# Process tokens incrementally as they're generated
generated_text = ""
try:
    # Get tokenizer reference for easier use
    tokenizer = olmocr_processor.tokenizer
    
    print("\nStarting token generation...")
    # Generate text and iterate over the tokens as they're generated
    for i, tokens in enumerate(generate(
        olmocr_model,
        olmocr_processor,
        text_prompt,
        main_image,
        max_tokens=1024,
        temperature=0.7,
    )):
        # Only print debug info for first 5 batches
        if i < 5:
            print(f"\nToken batch #{i+1}:")
            print(f"  Type: {type(tokens)}")
            print(f"  Content: {tokens}")
        
        # Handle different token types
        try:
            # If tokens is an array/tensor, convert to list if needed
            if hasattr(tokens, 'tolist'):
                tokens = tokens.tolist()
                if i < 5:
                    print(f"  Converted to list: {tokens}")
            # If tokens is a string, convert to token IDs
            elif isinstance(tokens, str):
                if i < 5:
                    print(f"  Converting string token to IDs: {tokens}")
                # Convert string to token IDs using the tokenizer
                token_ids = tokenizer.convert_tokens_to_ids(tokens)
                tokens = [token_ids] if isinstance(token_ids, int) else token_ids
            
            # Ensure tokens is a list of integers
            if not all(isinstance(t, int) for t in tokens):
                if i < 5:
                    print(f"  Warning: Converting non-integer tokens to integers")
                tokens = [int(t) for t in tokens if str(t).strip()]
            
            # Skip empty token lists
            if not tokens:
                continue
                
            # Decode this batch of tokens
            chunk = tokenizer.decode(tokens, skip_special_tokens=True)
            if i < 5:
                print(f"  Decoded chunk: '{chunk}'")
            
            # Append to our accumulated text
            generated_text += chunk
            print(chunk, end="", flush=True)
        except Exception as inner_e:
            if i < 5:
                print(f"\n  Error processing token batch: {inner_e}")
                print(f"  Skipping this batch and continuing...")
            continue
except Exception as e:
    print(f"\nError during generation: {e}")
    import traceback
    traceback.print_exc()
    
print("\n\nGeneration complete.")
print("\nFull generated text:")
print(generated_text)

# Save the full generated text to file for inspection
try:
    with open("generated_output.txt", "w") as f:
        f.write(generated_text)
    print("\nGenerated text saved to 'generated_output.txt'")
except Exception as save_error:
    print(f"\nError saving output to file: {save_error}")
