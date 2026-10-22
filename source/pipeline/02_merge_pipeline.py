"""
Script: 02_merge_pipeline.py
Package: source.pipeline
Dự án: PRJ-01 - Dự đoán Rủi ro Vỡ nợ Tín dụng (Home Credit Default Risk)
Mô tả: Điểm chạy trực tiếp cho Bước 2: Nén 1:N và Ghép nối đa bảng an toàn.
Cách chạy:
    python source/pipeline/02_merge_pipeline.py
"""

from pathlib import Path
import sys

# Đảm bảo source nằm trong PYTHONPATH
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from source.pipeline.merge_pipeline import run_merge_pipeline

if __name__ == "__main__":
    run_merge_pipeline()
