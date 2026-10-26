"""
Module: sanitizer.py
Package: source.pipeline.table_sanitizer.secondary
Mô tả: Bộ điều phối quy trình làm sạch cho 5 bảng phụ:
       - bureau
       - previous_application
       - POS_CASH_balance
       - installments_payments
       - credit_card_balance
       
       Các bước thực thi:
       1. Đọc tệp CSV thô tương ứng.
       2. Khử trùng lặp 100% dòng dữ liệu.
       3. Lọc triệt để khóa ngoại mồ côi (orphan_filter).
       4. Khử nhiễm các cột object về số và dọn text (column_sanitizer).
       5. Áp dụng quy tắc miền & khử ngoại lai (domain_rules).
       6. Lưu tệp kết quả ra Parquet.
"""

import gc
import time
from pathlib import Path
from typing import Any, Dict, Set
import pandas as pd

try:
    from ..column_sanitizer import sanitize_column_series
    from .orphan_filter import filter_orphan_foreign_keys
    from .domain_rules import apply_secondary_domain_rules
except (ImportError, ValueError):
    import sys
    _current = Path(__file__).resolve()
    for _parent in _current.parents:
        if (_parent / "source").exists():
            if str(_parent) not in sys.path:
                sys.path.insert(0, str(_parent))
            break
    from source.pipeline.table_sanitizer.column_sanitizer import sanitize_column_series
    from source.pipeline.table_sanitizer.secondary.orphan_filter import filter_orphan_foreign_keys
    from source.pipeline.table_sanitizer.secondary.domain_rules import apply_secondary_domain_rules


def sanitize_secondary_table(
    table_name: str,
    valid_sk_ids: Set[int],
    raw_path: Path,
    output_path: Path
) -> Dict[str, Any]:
    """
    Làm sạch trọn gói một bảng phụ.
    
    Parameters:
        table_name: Tên bảng phụ (không có đuôi .csv).
        valid_sk_ids: Tập hợp SK_ID_CURR hợp lệ từ application_train.
        raw_path: Đường dẫn thư mục dữ liệu thô.
        output_path: Đường dẫn thư mục xuất parquet.
        
    Returns:
        Dict[str, Any]: Thống kê kết quả làm sạch của bảng phụ.
    """
    file_path = raw_path / f"{table_name}.csv"
    if not file_path.exists():
        print(f"⚠️ Cảnh báo: Không tìm thấy tệp {file_path}, bỏ qua.")
        return {"table": table_name, "status": "skipped"}

    print(f"\n{'-'*75}")
    print(f"▶ ĐANG XỬ LÝ BẢNG PHỤ: {table_name}.csv")
    print(f"{'-'*75}")
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

    # 3. Lọc bỏ khóa ngoại mồ côi (Orphan Foreign Keys)
    df, orphan_removed = filter_orphan_foreign_keys(df, valid_sk_ids, table_name)

    # 4. Khử nhiễm các cột object
    converted_cols = []
    cleaned_text_cols = []
    for col in df.columns:
        if col in ["SK_ID_CURR", "SK_ID_BUREAU", "SK_ID_PREV"]:
            continue
        cleaned_series, action = sanitize_column_series(df[col], col)
        df[col] = cleaned_series
        if action == "converted_to_numeric":
            converted_cols.append(col)
        elif action == "cleaned_text":
            cleaned_text_cols.append(col)

    if converted_cols:
        print(f"• Đã chuyển đổi {len(converted_cols)} cột object nhiễm đơn vị sang kiểu số float")
    if cleaned_text_cols:
        print(f"• Đã chuẩn hóa {len(cleaned_text_cols)} cột phân loại text")

    # 5. Áp dụng quy tắc miền và khử ngoại lai đặc thù
    df = apply_secondary_domain_rules(df, table_name)

    # 6. Xuất tệp Parquet
    out_file = output_path / f"{table_name}.parquet"
    df.to_parquet(out_file, index=False, engine="pyarrow")
    file_size_mb = out_file.stat().st_size / (1024 * 1024)
    elapsed = time.time() - t0
    print(f"✓ Đã xuất: {out_file} ({file_size_mb:.2f} MB) trong {elapsed:.1f}s")

    stats = {
        "table": table_name,
        "initial_rows": initial_rows,
        "clean_rows": len(df),
        "dup_removed": dup_removed,
        "orphan_removed": orphan_removed,
        "converted_numeric_cols": len(converted_cols),
        "file_size_mb": file_size_mb
    }

    del df
    gc.collect()
    return stats
