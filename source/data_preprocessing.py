"""
Module: data_preprocessing.py
Package: source
Dự án: PRJ-01 - Dự đoán Rủi ro Vỡ nợ Tín dụng (Home Credit Default Risk)
Mô tả: Điểm truy cập cấp cao (High-level Interface / Façade) phục vụ trực tiếp cho
       quá trình phân tích (EDA) và huấn luyện mô hình LightGBM & CatBoost:
       - HỦY BỎ hoàn toàn One-Hot Encoding, StandardScaler/RobustScaler, và Median Imputation.
       - Cung cấp hàm clean_data_for_trees(): Làm sạch và bảo toàn Native Missing Values.
       - Cung cấp hàm get_lightgbm_data(): Ép 16 biến phân loại sang pd.Categorical,
         chia Stratified Train/Val (X_train, y_train, X_val, y_val).
       - Cung cấp hàm get_catboost_data(): Giữ biến phân loại ở dạng string,
         xuất cat_features và chia Stratified Train/Val (X_train, y_train, X_val, y_val, cat_features).
"""

from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd

# Đảm bảo console Windows in tiếng Việt chuẩn UTF-8
if sys.platform.startswith("win"):
    try:
        if sys.stdout and hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if sys.stderr and hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Import các thành phần cốt lõi từ package con data_cleaning
from .data_cleaning import (
    CATEGORICAL_COLS,
    clean_data_for_trees,
    get_lightgbm_data,
    get_catboost_data,
    detect_categorical_columns,
    TreeCleaningPipeline,
    safe_downcast_numeric,
    handle_business_anomalies,
    filter_low_quality_features,
    handle_business_outliers,
    handle_structural_missing,
)

__all__ = [
    "CATEGORICAL_COLS",
    "clean_data_for_trees",
    "get_lightgbm_data",
    "get_catboost_data",
    "detect_categorical_columns",
    "TreeCleaningPipeline",
    "safe_downcast_numeric",
    "handle_business_anomalies",
    "filter_low_quality_features",
    "handle_business_outliers",
    "handle_structural_missing",
]
