"""
Module: sanitizer.py
Package: source.pipeline.table_sanitizer.application
Mô tả: Bộ điều phối quy trình làm sạch bảng chính application_train:
       1. Đọc dữ liệu thô application_train.csv.
       2. Khử trùng lặp 100%.
       3. Kiểm toán & chuẩn hóa nhãn TARGET (target_cleaner).
       4. Khử nhiễm 104 cột object về số và dọn text (column_sanitizer).
       5. Áp dụng quy tắc miền & lọc ngoại lai phi lý (outlier_cleaner).
       6. Trích xuất danh sách khách hàng hợp lệ (valid_sk_ids) và xuất Parquet.
"""

import gc
import time
from pathlib import Path
from typing import Any, Dict, Set, Tuple
import pandas as pd

try:
    from ..column_sanitizer import sanitize_column_series
    from .target_cleaner import audit_and_clean_target
    from .outlier_cleaner import clean_application_outliers
except (ImportError, ValueError):
    import sys
    _current = Path(__file__).resolve()
    for _parent in _current.parents:
        if (_parent / "source").exists():
            if str(_parent) not in sys.path:
                sys.path.insert(0, str(_parent))
            break
    from source.pipeline.table_sanitizer.column_sanitizer import sanitize_column_series
    from source.pipeline.table_sanitizer.application.target_cleaner import audit_and_clean_target
    from source.pipeline.table_sanitizer.application.outlier_cleaner import clean_application_outliers


def sanitize_application_train(
    raw_path: Path,
    output_path: Path
) -> Tuple[Set[int], Dict[str, Any]]:
    """
    Điều phối toàn bộ quy trình làm sạch chuyên sâu bảng application_train.
    
    Parameters:
        raw_path: Đường dẫn thư mục chứa application_train.csv.
        output_path: Đường dẫn thư mục xuất application_train.parquet.
        
    Returns:
        Tuple[Set[int], Dict[str, Any]]: (valid_sk_ids, stats)
    """
    file_path = raw_path / "application_train.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {file_path}")

    print(f"\n{'='*75}")
    print(f"▶ [1/6] ĐANG XỬ LÝ BẢNG CHÍNH: application_train.csv")
    print(f"{'='*75}")
    t0 = time.time()

    # 1. Đọc dữ liệu
    df = pd.read_csv(file_path, low_memory=False)
    initial_rows, initial_cols = df.shape
    print(f"• Quy mô ban đầu: {initial_rows:,} dòng, {initial_cols} cột")

    # 2. Khử trùng lặp dòng 100%
    before_dup = len(df)
    df = df.drop_duplicates()
    dup_removed = before_dup - len(df)
    if dup_removed > 0:
        print(f"• Đã xóa {dup_removed:,} dòng trùng lặp 100%")

    # 3. Kiểm toán & làm sạch nhãn mục tiêu TARGET
    df, target_stats = audit_and_clean_target(df, initial_rows)
    valid_rows = target_stats["valid_rows"]
    discarded_rows = target_stats["discarded_rows"]
    default_rate = target_stats["default_rate"]

    # 4. Khử nhiễm các cột object & dọn khoảng trắng
    print(f"• Đang khử nhiễm các cột object và khoảng trắng thừa...")
    converted_cols = []
    cleaned_text_cols = []

    for col in df.columns:
        if col in ["SK_ID_CURR", "TARGET"]:
            continue
        cleaned_series, action = sanitize_column_series(df[col], col)
        df[col] = cleaned_series
        if action == "converted_to_numeric":
            converted_cols.append(col)
        elif action == "cleaned_text":
            cleaned_text_cols.append(col)

    print(f"  + Đã chuyển đổi {len(converted_cols)} cột object nhiễm đơn vị sang kiểu số float")
    print(f"  + Đã chuẩn hóa {len(cleaned_text_cols)} cột phân loại text (strip khoảng trắng, gán missing ẩn -> NaN)")

    # 5. Khử ngoại lai phi lý và áp dụng quy tắc miền
    df = clean_application_outliers(df)

    # 6. Thu thập danh sách khách hàng hợp lệ SK_ID_CURR
    valid_sk_ids = set(df["SK_ID_CURR"].astype(int).unique())
    print(f"• Số lượng khóa chính khách hàng SK_ID_CURR hợp lệ: {len(valid_sk_ids):,}")

    # 7. Lưu ra tệp Parquet
    out_file = output_path / "application_train.parquet"
    df.to_parquet(out_file, index=False, engine="pyarrow")
    file_size_mb = out_file.stat().st_size / (1024 * 1024)
    elapsed = time.time() - t0
    print(f"✓ Đã xuất: {out_file} ({file_size_mb:.2f} MB) trong {elapsed:.1f}s")

    stats = {
        "table": "application_train",
        "initial_rows": initial_rows,
        "clean_rows": len(df),
        "discarded_rows": discarded_rows,
        "default_rate": default_rate,
        "converted_numeric_cols": len(converted_cols),
        "file_size_mb": file_size_mb
    }

    del df
    gc.collect()
    return valid_sk_ids, stats
