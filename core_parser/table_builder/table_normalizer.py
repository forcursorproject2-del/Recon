import pandas as pd
import logging
from typing import List, Dict, Any
try:
    import camelot
    CAMELOT_AVAILABLE = True
except ImportError:
    CAMELOT_AVAILABLE = False
    logging.warning("Camelot not available, OCR table extraction disabled.")

logger = logging.getLogger(__name__)

class TableBuilder:
    def normalize_tables(self, tables_list: List[List[List]]) -> List[pd.DataFrame]:
        dfs = []
        for tables in tables_list:
            for table in tables:
                if table:
                    df = pd.DataFrame(table)
                    df = self._clean_table(df)
                    dfs.append(df)
        return dfs

    def extract_tables_with_camelot(self, pdf_path: str) -> List[pd.DataFrame]:
        if not CAMELOT_AVAILABLE:
            logger.warning("Camelot not available, using fallback.")
            return []
        try:
            tables = camelot.read_pdf(pdf_path, pages='all')
            dfs = [table.df for table in tables]
            # Validate sums
            for df in dfs:
                self._validate_table_sums(df)
            return dfs
        except Exception as e:
            logger.error(f"Camelot extraction failed: {e}")
            return []

    def _clean_table(self, df: pd.DataFrame) -> pd.DataFrame:
        # Remove empty rows and columns
        df = df.dropna(how='all').dropna(axis=1, how='all')
        # Infer headers if first row looks like headers
        if not df.empty and df.iloc[0].astype(str).str.contains(r'\d').sum() < len(df.columns) / 2:
            df.columns = df.iloc[0]
            df = df[1:].reset_index(drop=True)
        # Parse numbers and dates
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='ignore')
        return df

    def extract_operations(self, df: pd.DataFrame) -> Dict[str, Any]:
        # For reconciliation act, look for saldo, debet, kredit
        operations = {}
        df_str = df.astype(str).apply(lambda x: x.str.lower())
        if 'сальдо' in df_str.values.flatten() or 'дебет' in df_str.values.flatten():
            # Simple extraction: find rows with numbers
            numeric_cols = df.select_dtypes(include=[float, int]).columns
            if len(numeric_cols) >= 2:
                operations['turnovers'] = df[numeric_cols].to_dict('records')
        return operations

    def _validate_table_sums(self, df: pd.DataFrame):
        # Validate if last row is sum of previous rows in numeric columns
        if df.empty or len(df) < 2:
            return
        numeric_cols = df.select_dtypes(include=[float, int]).columns
        for col in numeric_cols:
            values = pd.to_numeric(df[col], errors='coerce').dropna()
            if len(values) >= 2:
                sum_val = values.iloc[-1]
                calc_sum = values.iloc[:-1].sum()
                if abs(calc_sum - sum_val) > 0.01:
                    logger.warning(f"Sum validation failed for column {col}: expected {calc_sum}, got {sum_val}")
