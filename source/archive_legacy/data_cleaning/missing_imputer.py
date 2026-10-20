"""
Module: missing_imputer.py
Package: source.data_cleaning
Dự án: PRJ-01 - Dự đoán Rủi ro Vỡ nợ Tín dụng (Home Credit Default Risk)
Mô tả: [LEGACY / ALIAS] Điều hướng sang module structural_imputer.py.
       Loại bỏ việc điền Median cho các biến số, bảo toàn cơ chế Native Missing cho LightGBM/CatBoost.
"""

from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
from .structural_imputer import handle_structural_missing


def handle_missing_data(
    df: pd.DataFrame,
    is_train: bool = True,
    cc_prefix: str = "CC_",
    cc_flag_col: str = "NO_CC",
    imputer_state: Optional[Dict[str, Any]] = None,
    protected_cols: Optional[List[str]] = None,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Hàm tương thích ngược (Backward Compatible Wrapper):
    Điều hướng sang handle_structural_missing để xử lý khuyết thiếu tối ưu cho mô hình cây.
    """
    return handle_structural_missing(
        df=df,
        is_train=is_train,
        cc_prefix=cc_prefix,
        cc_flag_col=cc_flag_col,
        imputer_state=imputer_state,
        fill_categorical_missing=True,
        categorical_fill_value="Missing",
        protected_cols=protected_cols,
        verbose=verbose,
    )
