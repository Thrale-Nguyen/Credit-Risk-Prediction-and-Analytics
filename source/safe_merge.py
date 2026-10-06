"""
Module: safe_merge.py
Dự án: PRJ-01 - Dự đoán Rủi ro Vỡ nợ Tín dụng (Home Credit Default Risk)
Mô tả: Hàm merge an toàn giữa bảng chính (application_train) và bảng đặc trưng phụ,
       kiểm soát nghiêm ngặt bẫy bùng nổ dòng (Row Explosion) và xung đột cột (_x, _y).
"""

import gc
import sys
import time
from typing import Optional, List
import pandas as pd

# Đảm bảo console Windows in tiếng Việt chuẩn Unicode
if sys.platform.startswith("win"):
    try:
        if sys.stdout and hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if sys.stderr and hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def safe_merge(
    main_df: pd.DataFrame,
    secondary_df: pd.DataFrame,
    on: str = "SK_ID_CURR",
    how: str = "left",
    del_secondary: bool = True,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Thực hiện Left Join an toàn giữa bảng chính và bảng đặc trưng phụ với 3 lớp kiểm định:
    1. Kiểm tra toàn vẹn số dòng: Tuyệt đối không để xảy ra Row Explosion (len(merged) == len(main)).
    2. Kiểm tra xung đột cột: Đảm bảo không phát sinh các cột có đuôi _x, _y.
    3. Giải phóng bộ nhớ RAM: Dọn rác bảng phụ và in dung lượng tiết kiệm được.

    Parameters:
    -----------
    main_df : pd.DataFrame
        Bảng chính (ví dụ application_train).
    secondary_df : pd.DataFrame
        Bảng đặc trưng phụ đã được aggregate và làm phẳng (chuẩn hóa ở Bước 2).
    on : str, default 'SK_ID_CURR'
        Khóa ghép nối.
    how : str, default 'left'
        Phương thức ghép nối (mặc định 'left' để bảo toàn tập mẫu).
    del_secondary : bool, default True
        Nếu True, giải phóng bộ nhớ của bảng phụ sau khi merge.
    verbose : bool, default True
        Nếu True, in kết quả nghiệm thu chi tiết.

    Returns:
    --------
    pd.DataFrame
        DataFrame sau khi ghép nối thành công.
    """
    start_time = time.time()
    n_main_orig = len(main_df)
    n_cols_orig = main_df.shape[1]
    n_cols_sec = secondary_df.shape[1]

    # Đo dung lượng bộ nhớ bảng phụ trước khi giải phóng
    sec_mem_mb = secondary_df.memory_usage(deep=True).sum() / (1024 * 1024)

    # 1. Thao tác Merge
    df_merged = pd.merge(main_df, secondary_df, on=on, how=how)

    # 2. Kiểm tra tính toàn vẹn (Integrity Assertions)
    # Lớp 1: Kiểm tra số dòng (Ngăn chặn Row Explosion)
    n_merged = len(df_merged)
    assert n_merged == n_main_orig, (
        f"LỖI BÙNG NỔ DÒNG (Row Explosion Trap)! "
        f"Số dòng bị tăng từ {n_main_orig:,d} lên {n_merged:,d} (+{n_merged - n_main_orig:,d} dòng). "
        f"Dấu hiệu cho thấy bảng phụ chưa được aggregate triệt để về quan hệ 1 : 1."
    )

    # Lớp 2: Kiểm tra xung đột tên cột (_x, _y)
    collision_cols = [c for c in df_merged.columns if c.endswith("_x") or c.endswith("_y")]
    assert len(collision_cols) == 0, (
        f"LỖI XUNG ĐỘT TÊN CỘT! Phát hiện {len(collision_cols)} cột mang hậu tố _x/_y: "
        f"{collision_cols[:10]}. Hãy kiểm tra lại tiền tố (prefix) trước khi merge."
    )

    # 3. Quản lý bộ nhớ RAM
    if del_secondary:
        del secondary_df
        gc.collect()

    merged_mem_mb = df_merged.memory_usage(deep=True).sum() / (1024 * 1024)
    elapsed_time = time.time() - start_time

    if verbose:
        print("=" * 65)
        print("✓ MERGE AN TOÀN THÀNH CÔNG (ZERO ROW EXPLOSION)")
        print("=" * 65)
        print(f"• Số dòng trước merge : {n_main_orig:,d}")
        print(f"• Số dòng sau merge  : {n_merged:,d} (Khớp 100%, 0 dòng nhân bản)")
        print(f"• Số cột ban đầu     : {n_cols_orig:,d} cột")
        print(f"• Số cột sau merge   : {df_merged.shape[1]:,d} cột (+{df_merged.shape[1] - n_cols_orig:,d} đặc trưng mới)")
        print(f"• Dung lượng bảng phụ giải phóng : ~{sec_mem_mb:.2f} MB")
        print(f"• Dung lượng bảng chính hiện tại: {merged_mem_mb:.2f} MB")
        print(f"• Thời gian thực thi : {elapsed_time:.2f}s")
        print("=" * 65)

    return df_merged


if __name__ == "__main__":
    # Test mẫu giả lập
    import numpy as np

    df_main_mock = pd.DataFrame({
        "SK_ID_CURR": [100001, 100002, 100003],
        "TARGET": [0, 1, 0],
        "AMT_CREDIT": [100000.0, 200000.0, 150000.0]
    })

    df_sec_mock = pd.DataFrame({
        "SK_ID_CURR": [100001, 100002, 100003],
        "BUREAU_AMT_CREDIT_MEAN": [50000.0, 80000.0, 60000.0],
        "BUREAU_DAYS_CREDIT_MAX": [-100, -200, -50]
    })

    res = safe_merge(df_main_mock, df_sec_mock, on="SK_ID_CURR", verbose=True)
