import pytest
import pandas as pd
from core_parser.table_builder.table_normalizer import TableBuilder

def test_table_builder_normalize():
    builder = TableBuilder()
    tables = [[['header1', 'header2'], ['val1', 'val2']]]
    dfs = builder.normalize_tables([tables])
    assert len(dfs) == 1
    assert isinstance(dfs[0], pd.DataFrame)

def test_extract_operations():
    builder = TableBuilder()
    # Add second numeric column to satisfy extract_operations condition
    df = pd.DataFrame({'col1': ['сальдо', 'дебет'], 'col2': [100, 200], 'col3': [300, 400]})
    ops = builder.extract_operations(df)
    assert 'turnovers' in ops
