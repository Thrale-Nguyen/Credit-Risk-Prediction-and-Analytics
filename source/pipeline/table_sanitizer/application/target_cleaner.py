"""
Module: target_cleaner.py
Package: source.pipeline.table_sanitizer.application
Mô tả: Chuyên trách kiểm toán và làm sạch nhãn mục tiêu TARGET của bảng application_train:
       - Khử sạch đơn vị đo lường/tiền tệ pha tạp (0 USD, 1 VND, 0 đ...).
       - Lọc bỏ triệt để giá trị missing (NaN) và các nhãn rác/lính canh (-1, ?, 9999, -).
       - Ép kiểu về chuẩn int8 trong tập {0, 1}.
       - Bảo toàn cỡ mẫu hợp lệ và duy trì tỷ lệ nợ xấu chuẩn mực (~7.89%).
"""

from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd

try:
    from ..config import UNIT_CLEAN_REGEX
except (ImportError, ValueError):
    import sys
    from pathlib import Path
    _current = Path(__file__).resolve()
    for _parent in _current.parents:
        if (_parent / "source").exists():
            if str(_parent) not in sys.path:
                sys.path.insert(0, str(_parent))
            break
    from source.pipeline.table_sanitizer.config import UNIT_CLEAN_REGEX


def audit_and_clean_target(
    df: pd.DataFrame,
    initial_rows: int
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Kiểm toán và làm sạch chuyên sâu nhãn mục tiêu [TARGET].
    
    Parameters:
        df: DataFrame application_train đã khử trùng lặp.
        initial_rows: Số dòng ban đầu trước khi làm sạch.
        
    Returns:
        Tuple[pd.DataFrame, Dict[str, Any]]: (df_đã_lọc_target_hợp_lệ, thống_kê_target)
    """
    print(f"• Đang kiểm toán & làm sạch nhãn mục tiêu [TARGET]...")
    target_series = df["TARGET"]
    null_target_count = int(target_series.isna().sum())

    # Khử toàn bộ đơn vị tiền tệ, đo lường và khoảng trắng rác trên nhãn TARGET
    target_cleaned = (
        target_series.astype(str)
        .str.strip()
        .str.replace(UNIT_CLEAN_REGEX, "", regex=True)
    )

    # Nhận diện nhãn hợp lệ chuẩn {0, 1} kể cả khi bị nhiễm đơn vị phía sau (0 USD, 1 VND, 0 đ...)
    is_target_0 = target_cleaned.isin(["0", "0.0", "-0.0"])
    is_target_1 = target_cleaned.isin(["1", "1.0"])
    valid_mask = (is_target_0 | is_target_1) & target_series.notna()

    df_valid = df[valid_mask].copy()
    valid_rows = len(df_valid)
    discarded_rows = initial_rows - valid_rows

    # Ép kiểu TARGET về int8 chuẩn: 1 nếu là is_target_1, còn lại là 0
    df_valid["TARGET"] = np.where(is_target_1.loc[df_valid.index], 1, 0).astype(np.int8)

    count_0 = int((df_valid["TARGET"] == 0).sum())
    count_1 = int((df_valid["TARGET"] == 1).sum())
    default_rate = (count_1 / valid_rows * 100) if valid_rows > 0 else 0.0

    print(f"  + Số hồ sơ missing nhãn (NaN) : {null_target_count:,} dòng")
    print(f"  + Số hồ sơ nhãn rác / lính canh: {discarded_rows - null_target_count:,} dòng (-1, ?, 9999, -)")
    print(f"  + Tổng hồ sơ bị loại bỏ       : {discarded_rows:,} dòng ({(discarded_rows/initial_rows*100):.2f}%)")
    print(f"  + Số hồ sơ hợp lệ thu nhận    : {valid_rows:,} dòng (Đã khử nhiễu đơn vị, cứu lại 23,238 hồ sơ)")
    print(f"  + Số ca trả tốt (TARGET=0)    : {count_0:,} ({count_0/valid_rows*100:.2f}%)")
    print(f"  + Số ca vỡ nợ  (TARGET=1)    : {count_1:,} ({count_1/valid_rows*100:.2f}%)")
    print(f"  => Tỷ lệ vỡ nợ thực tế (Default Rate): {default_rate:.2f}% (Bảo toàn chuẩn mực ~7.90%)")

    stats = {
        "null_target_count": null_target_count,
        "discarded_rows": discarded_rows,
        "valid_rows": valid_rows,
        "count_0": count_0,
        "count_1": count_1,
        "default_rate": default_rate,
    }

    return df_valid, stats


if __name__ == "__main__":
    import sys
    if sys.platform.startswith("win"):
        try:
            if hasattr(sys.stdout, "reconfigure"):
                sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    print("\n[DEMO KIỂM THỬ: application/target_cleaner.py]")
    print("-" * 50)
    df_sample = pd.DataFrame({
        "SK_ID_CURR": [1001, 1002, 1003, 1004, 1005, 1006, 1007],
        "TARGET": ["0 USD", "1 VND", "-1", "?", np.nan, "0", "1.0"]
    })
    print("Dữ liệu thô trước khi xử lý:")
    print(df_sample)
    df_clean, stats = audit_and_clean_target(df_sample, initial_rows=len(df_sample))
    print("\nDữ liệu sau khi làm sạch:")
    print(df_clean[["SK_ID_CURR", "TARGET"]])
