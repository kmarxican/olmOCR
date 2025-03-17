# olmOCR-MLX

A high-performance PDF OCR tool powered by MLX-VLM and olmOCR models for efficient text extraction and document processing.

## Features

- PDF text extraction using state-of-the-art MLX-VLM and olmOCR models
- Parallel processing support for faster document processing (planned in the future)
- Flexible output formats including raw text and Dolma-compatible* JSONL (*See note below)
- HTML preview generation for extracted content (formatting of output is being worked on)
- UTF-8 safe text processing with robust error handling
- Support for both local PDF files and URL-based PDF downloads

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install torch pillow tqdm PyPDF2
pip install mlx-vlm olmocr
```

## Usage

Basic usage:
```bash
python hyperwarp.py --pdf your_document.pdf
```

Advanced options:
```bash
python hyperwarp.py \
  --pdf input.pdf \
  --output-dir results \
  --model "mlx-community/olmOCR-7B-0225-preview-4bit" \
  --max-tokens 8192 \
  --temperature 0.2 \
  --image-dim 1024 \
  --workers 4 \
  --preview \
  --preview-dir dolma_previews
```

### Command Line Arguments

- `--pdf`: Path to PDF file or URL to download
- `--output-dir`: Directory to save results (default: "results")
- `--model`: Model path or name (default: mlx-community/olmOCR-7B-0225-preview-4bit)
- `--max-tokens`: Max tokens per page (default: 8192)
- `--temperature`: Generation temperature (default: 0.2)
- `--image-dim`: Target image dimension (default: 1024)
- `--workers`: Number of worker threads (default: 4)
- `--sequential`: Process pages sequentially instead of in parallel
- `--intermediate`: Directory to save intermediate results
- `--preview`: Generate HTML previews with dolmaviewer
- `--preview-dir`: Directory for HTML previews
- `--max-pages`: Maximum number of pages to process
- `--start-page`: Page number to start processing from (default: 1)

## Plans for Future Updates

- Implement parallel processing for faster document processing
- Improve HTML preview generation and formatting
- Add support for additional input formats
- Add support for additional output formats
- Enhance error handling and robustness
- Optimize performance and resource usage
- Create a GUI interface

## License

This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0). See the [LICENSE](license.md) file for details.

Key points of the AGPL-3.0 license:
- You can use, modify, and distribute this software
- If you modify the software, you must make your changes available under the same license
- If you use the software to provide a service over a network, you must make the complete source code available to users of that service