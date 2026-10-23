"""
Script: 03_tree_preprocessing.py
Package: source.pipeline
Dự án: PRJ-01 - Dự đoán Rủi ro Vỡ nợ Tín dụng (Home Credit Default Risk)
Mô tả: Điểm chạy trực tiếp cho Bước 3: Tiền xử lý dữ liệu chuyên biệt tối ưu cho LightGBM:
       - Xử lý dị thường nghiệp vụ (DAYS_EMPLOYED_ANOM, NO_CC, AGE_YEARS, DAYS_*).
       - Bổ sung 4 tỷ lệ đòn bẩy tài chính (CREDIT_TO_INCOME, ANNUITY_TO_INCOME, CREDIT_TERM, INCOME_PER_PERSON).
       - Ép 16 biến phân loại sang pd.Categorical & Safe Dynamic Downcasting bộ nhớ RAM.
       - Phân chia phân tầng Stratified Train/Val Split (80/20) bảo toàn 7.89% nợ xấu.
       - Xuất bản: train_lgb.parquet, val_lgb.parquet, pipeline_metadata.json.
Cách chạy:
    python source/pipeline/03_tree_preprocessing.py
"""

from pathlib import Path
import sys

# Đảm bảo thư mục gốc dự án nằm trong PYTHONPATH
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from source.pipeline.tree_preprocessing import run_tree_preprocessing

if __name__ == "__main__":
    run_tree_preprocessing()
