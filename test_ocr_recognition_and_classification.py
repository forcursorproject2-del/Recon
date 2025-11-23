import logging
from core_parser.config_manager.config_loader import ConfigManager
from core_parser.pdf_extractor.pdf_reader import PDFTextExtractor
from core_parser.classifier.document_classifier import DocumentClassifier

logging.basicConfig(level=logging.INFO)

def main():
    config_manager = ConfigManager()
    extractor = PDFTextExtractor(config_manager=config_manager)
    pdf_path = 'incoming/CCF_003705.pdf'
    data = extractor.extract_text_with_structure(pdf_path)
    full_text = data['full_text']

    print("=== RECOGNIZED TEXT START ===")
    print(full_text[:1000])
    print("=== RECOGNIZED TEXT END ===")

    classifier = DocumentClassifier(config_manager)
    classification = classifier.classify_document(full_text, data)
    print(f"Classification: {classification.doc_type} with confidence {classification.confidence}")

if __name__ == '__main__':
    main()
