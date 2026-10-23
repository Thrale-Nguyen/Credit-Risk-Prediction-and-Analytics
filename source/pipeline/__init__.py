"""
Package: source.pipeline
Mô tả: Quản lý quy trình xử lý dữ liệu tuyến tính 3 bước:
       - 01_table_sanitizer.py: Làm sạch dữ liệu cấp bảng độc lập từ data/raw/
       - 02_merge_pipeline.py: Ghép nối đa bảng nén 1:N về 1:1
       - 03_tree_preprocessing.py: Tiền xử lý chuyên biệt tối ưu cho LightGBM
       - run_pipeline.py: Trình điều phối liên hoàn 1-click với Smart Cache Skip
"""

from .table_sanitizer import run_table_sanitizer
from .merge_pipeline import run_merge_pipeline
from .tree_preprocessing import run_tree_preprocessing, get_lightgbm_data

__all__ = [
    "run_table_sanitizer",
    "run_merge_pipeline",
    "run_tree_preprocessing",
    "get_lightgbm_data",
]

