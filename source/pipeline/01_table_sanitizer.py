"""
Script: 01_table_sanitizer.py
Package: source.pipeline
Dự án: PRJ-01 - Dự đoán Rủi ro Vỡ nợ Tín dụng (Home Credit Default Risk)
Mô tả: Điểm chạy trực tiếp cho Bước 1: Làm sạch cấp bảng độc lập.
Cách chạy:
    python source/pipeline/01_table_sanitizer.py
"""

from pathlib import Path
import sys

# Đảm bảo source nằm trong PYTHONPATH
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from source.pipeline.table_sanitizer import run_table_sanitizer

if __name__ == "__main__":
    run_table_sanitizer()
