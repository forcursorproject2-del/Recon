import pytest
import os
import tempfile
from core_parser.config_manager.config_loader import ConfigManager

def test_config_manager_default():
    config = ConfigManager()
    assert 'document_signatures' in config.config
    assert 'field_patterns' in config.config

def test_config_manager_with_file():
    config_data = """
document_signatures:
  test:
    keywords: ['test']
    patterns: ['test']
    exclude: []
field_patterns:
  test_field:
    pattern: 'test'
    validate: 'digits_10_12'
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        try:
            f.write(config_data)
            f.flush()
            config = ConfigManager(f.name)
            assert 'test' in config.get_signatures()
            assert 'test_field' in config.get_patterns()
        finally:
            f.close()
            os.unlink(f.name)
