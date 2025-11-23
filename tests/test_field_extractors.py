import pytest
from core_parser.semantic_parser.field_extractors import FieldExtractor, SemanticParser
from core_parser.config_manager.config_loader import ConfigManager
from core_parser.table_builder.table_normalizer import TableBuilder

def test_field_extractor():
    config = ConfigManager()
    extractor = FieldExtractor(config)
    text = "ИНН 123456789012"
    fields = extractor.extract_fields(text, 'invoice')
    assert 'inn' in fields
    assert fields['inn'].value == '123456789012'

def test_semantic_parser():
    config = ConfigManager()
    table_builder = TableBuilder()
    parser = SemanticParser(config, table_builder)
    text = "some text"
    structure = {'pages': [{'tables': []}]}
    result = parser.parse_document(text, structure, 'invoice')
    assert 'fields' in result
    assert 'tables' in result
