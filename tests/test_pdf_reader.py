import pytest
import os
from core_parser.pdf_extractor.pdf_reader import PDFTextExtractor, PDFBatchProcessor

def test_pdf_text_extractor_init():
    extractor = PDFTextExtractor()
    # PDFTextExtractor does not have use_ocr attribute
    assert extractor.reader is not None

def test_pdf_batch_processor_init():
    processor = PDFBatchProcessor(use_ocr=True)
    assert processor.use_ocr is True
    assert processor.extractor is not None

def test_pdf_batch_processor_process_folder_with_sample_pdf():
    processor = PDFBatchProcessor(use_ocr=True)
    pdf_dir = os.path.join(os.path.dirname(__file__), "../incoming")
    if not os.path.exists(pdf_dir):
        pytest.skip("Incoming folder with sample PDFs does not exist")

    # Process the sample pdf files in incoming
    results = processor.process_folder(pdf_dir)
    # We expect at least one pdf file processing result
    assert isinstance(results, dict)
    assert len(results) > 0
    for filename, result in results.items():
        # Check result dictionary keys
        assert "pages" in result
        assert "full_text" in result
        # The full_text should be a string (possibly empty)
        assert isinstance(result["full_text"], str)

def test_pdf_batch_processor_extract_text_with_structure_uses_ocr_logic():
    processor = PDFBatchProcessor(use_ocr=True)
    pdf_dir = os.path.join(os.path.dirname(__file__), "../incoming")
    pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')]
    if not pdf_files:
        pytest.skip("No PDF files found in incoming directory")

    test_pdf = os.path.join(pdf_dir, pdf_files[0])
    result = processor.extract_text_with_structure(test_pdf)
    assert isinstance(result, dict)
    assert "pages" in result
    assert "full_text" in result
    assert isinstance(result["full_text"], str)
