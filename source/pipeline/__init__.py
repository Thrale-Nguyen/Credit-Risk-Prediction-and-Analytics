"""
Package: source.pipeline
Mô tả: Quản lý quy trình xử lý dữ liệu tuyến tính 3 bước:
       - 01_table_sanitizer.py: Làm sạch dữ liệu cấp bảng độc lập từ data/raw/
       - 02_merge_pipeline.py: Ghép nối đa bảng nén 1:N về 1:1
       - 03_tree_preprocessing.py: Tiền xử lý cho mô hình LightGBM & CatBoost
"""

from .table_sanitizer import run_table_sanitizer
from .merge_pipeline import run_merge_pipeline

__all__ = ["run_table_sanitizer", "run_merge_pipeline"]
