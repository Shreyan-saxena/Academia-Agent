"""Vision-first PDF ingestion pipeline for academic documents."""
from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
from .ollama import OllamaClient

VISION_EXTRACTION_PROMPT = """
You are an expert document layout parser and academic data extraction assistant.
Analyze this image of a university course syllabus page.

Your task is to accurately transcribe the content into clean, structured Markdown format:
1. Preserve all tables (such as module contents, hours, pedagogy, and Course Articulation Matrices) using standard Markdown table syntax.
2. Maintain hierarchical headings for parts (e.g., Part A, Part B, Modules, Books, References).
3. Do not summarize or omit text. Transcribe all text, course codes, outcomes (CO1-CO5), and bloom levels precisely as they appear visually.
"""

def encode_image(image_path: str) -> str:
    """Encode image file to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def convert_pdf_to_images(pdf_path: Path, dpi: int = 300) -> list[Path]:
    """Rasterize PDF pages into high-resolution images."""
    doc = fitz.open(pdf_path)
    page_images = []

    # Create a temporary directory for images to avoid cluttering root
    temp_dir = pdf_path.parent / f".tmp_vision_{pdf_path.stem}"
    temp_dir.mkdir(exist_ok=True)

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        zoom = dpi / 72
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)

        image_path = temp_dir / f"page_{page_num + 1}.png"
        pix.save(str(image_path))
        page_images.append(image_path)

    return page_images

def process_syllabus_with_vision(pdf_path: Path, client: OllamaClient) -> Optional[str]:
    """Convert a PDF to structured Markdown using a vision-capable LLM."""
    try:
        image_paths = convert_pdf_to_images(pdf_path)
        full_markdown_document = []

        for idx, img_path in enumerate(image_paths):
            # Ollama expects images as base64 strings in a list
            img_b64 = encode_image(str(img_path))

            page_markdown = client.generate_multimodal(
                system="You are an expert document layout parser.",
                prompt=VISION_EXTRACTION_PROMPT,
                images=[img_b64]
            )

            if page_markdown:
                full_markdown_document.append(f"\n\n--- Page {idx + 1} ---\n\n" + page_markdown)

            # Cleanup temp image
            img_path.unlink()

        # Cleanup temp directory
        temp_dir = pdf_path.parent / f".tmp_vision_{pdf_path.stem}"
        if temp_dir.exists():
            temp_dir.rmdir()

        return "\n".join(full_markdown_document) if full_markdown_document else None
    except Exception as e:
        # Log error or return None
        return None
