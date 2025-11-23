import pytest
import tempfile
from core_parser.batch_processor.pipeline import BatchProcessingPipeline
from core_parser.config_manager.config_loader import ConfigManager

def test_pipeline_init():
    config = ConfigManager()
    pipeline = BatchProcessingPipeline(config)
    assert pipeline.max_workers == 4

def test_pipeline_process_empty_folder():
    config = ConfigManager()
    pipeline = BatchProcessingPipeline(config)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = pipeline.process_folder(tmpdir)
        assert 'summary' in results
        assert 'documents' in results
        assert results['summary']['total_documents'] == 0
