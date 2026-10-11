"""
Module: outlier_handler.py
Package: source.data_cleaning
Dự án: PRJ-01 - Dự đoán Rủi ro Vỡ nợ Tín dụng (Home Credit Default Risk)
Mô tả: [LEGACY / ALIAS] Điều hướng sang module business_rules.py.
       Hủy bỏ Winsorization 99th percentile cứng, thay bằng Business Sanity Rule.
"""

from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
from .business_rules import handle_business_outliers


def handle_financial_outliers(
    df: pd.DataFrame,
    is_train: bool = True,
    financial_cols: Optional[List[str]] = None,
    percentile: float = 0.99,
    method: str = "business_rule",
    outlier_caps_state: Optional[Dict[str, float]] = None,
    auto_detect_skewed: bool = False,
    skew_threshold: float = 10.0,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Hàm tương thích ngược (Backward Compatible Wrapper):
    Điều hướng xử lý ngoại lai sang handle_business_outliers theo quy tắc nghiệp vụ.
    """
    col = "AMT_INCOME_TOTAL"
    if financial_cols and len(financial_cols) > 0:
        col = financial_cols[0]

    df_out, state = handle_business_outliers(
        df=df,
        is_train=is_train,
        target_col=col,
        max_threshold=10_000_000.0,
        action="clip",
        outlier_state=outlier_caps_state,
        verbose=verbose,
    )
    return df_out, state
