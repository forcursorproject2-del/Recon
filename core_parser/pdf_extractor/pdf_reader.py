import pdfplumber
import fitz  # PyMuPDF
import logging
from typing import Dict, Any
import os
from pathlib import Path
from .ocr_cache import OcrCache
import re
from .paddle_ocr import PaddleOCRExtractor

logger = logging.getLogger(__name__)

class PDFTextExtractor:
    def __init__(self, config_manager=None):
        self.config_manager = config_manager

        self.ocr_cache = OcrCache()
        self.ocr_extractor = PaddleOCRExtractor(cache_db="ocr_cache.db")

    def extract_text_with_structure(self, pdf_path: str) -> Dict[str, object]:
        """
        Extracts text and structure from PDF. Uses OCR if text is short.
        Returns dict with keys: pages (list), metadata (dict), full_text (str).
        """
        logger.debug(f"Starting extraction for {pdf_path}, use_ocr: True")
        file_size = os.path.getsize(pdf_path)
        if file_size > 10 * 1024 * 1024:  # 10 MB limit
            raise ValueError(f"FILE_TOO_LARGE: {pdf_path} ({file_size / 1024 / 1024:.1f} MB)")

        try:
            with pdfplumber.open(pdf_path) as pdf:
                pages = []
                full_text = ""
                for page in pdf.pages:
                    text = page.extract_text() or ""
                    page_data = {
                        'text': text,
                        'words': page.extract_words(),
                        'bbox': page.bbox,
                        'tables': page.extract_tables()
                    }
                    pages.append(page_data)
                    full_text += text + "\n"
                metadata = pdf.metadata
        except Exception as e:
            logger.warning(f"pdfplumber failed for {pdf_path}: {e}. Using PyMuPDF fallback.")
            return self._extract_with_fitz(pdf_path)

        MIN_ACCEPTABLE_LENGTH = 300  # threshold length to trigger OCR
        if len(full_text.strip()) < MIN_ACCEPTABLE_LENGTH:
            logger.info(f"Extracted text too short ({len(full_text)}) characters, using OCR.")
            full_text = self._extract_with_ocr(pdf_path)
            # OCR returns unstructured text
            pages = [{'text': full_text, 'words': [], 'bbox': None, 'tables': []}]

        logger.debug(f"Full extracted text length: {len(full_text)}")
        logger.debug(f"Excerpt: {full_text[:1000]}")
        return {
            'pages': pages,
            'metadata': metadata,
            'full_text': full_text
        }

    def _extract_with_fitz(self, pdf_path: str) -> Dict[str, Any]:
        doc = fitz.open(pdf_path)
        full_text = ""
        pages = []
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text()
            full_text += text + "\n"
            pages.append({
                'text': text,
                'words': [],  # PyMuPDF doesn't extract words easily
                'bbox': page.rect,
                'tables': []  # Tables are harder with PyMuPDF, skip for now
            })
        metadata = doc.metadata
        doc.close()
        return {
            'pages': pages,
            'metadata': metadata,
            'full_text': full_text
        }

    def _extract_with_ocr(self, pdf_path: str) -> str:
        doc = fitz.open(pdf_path)
        text = ""
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            pix = page.get_pixmap()
            page_array = self._pixmap_to_numpy(pix)
            try:
                page_text = self.ocr_extractor.extract_text(page_array)
            except Exception as e:
                logger.error(f"OCR failed on page {page_num}: {e}")
                page_text = ""
            text += page_text + "\n"
        doc.close()
        return text

    def _pixmap_to_numpy(self, pix) -> 'np.ndarray':
        # Convert fitz Pixmap to numpy array (BGR)
        import numpy as np
        img = pix.samples
        if pix.n < 4:
            img_fmt = np.frombuffer(img, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            if pix.n == 3:
                # Convert RGB to BGR
                img_fmt = img_fmt[..., ::-1]
        else:
            img_fmt = np.frombuffer(img, dtype=np.uint8).reshape(pix.height, pix.width, 4)
            # Ignore alpha channel
            img_fmt = img_fmt[..., :3]
            # Convert RGB to BGR
            img_fmt = img_fmt[..., ::-1]
        return img_fmt
