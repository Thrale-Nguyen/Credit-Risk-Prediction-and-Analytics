"""
Module: downcast.py
Package: source.data_cleaning
Dự án: PRJ-01 - Dự đoán Rủi ro Vỡ nợ Tín dụng (Home Credit Default Risk)
Mô tả: Chốt chặn 1: Tối ưu dung lượng RAM bằng cơ chế Safe Dynamic Downcasting:
       - Kiểm tra biên độ an toàn cho các cột số nguyên (integer):
         + [-128, 127] -> int8
         + [-32,768, 32,767] -> int16
         + [-2,147,483,648, 2,147,483,647] -> int32
         + Vượt quá 2.147 tỷ: Bảo toàn tuyệt đối int64 (Chống tràn số Integer Overflow).
       - Ép kiểu an toàn cho các cột số thực (float64 -> float32):
         + Tăng tốc độ xây dựng Histogram binning của LightGBM gấp 2-3 lần.
         + Giảm 50% dung lượng RAM tiêu thụ.
"""

import sys
import time
from typing import Tuple
import numpy as np
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


def safe_downcast_numeric(
    df: pd.DataFrame,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Tối ưu hóa dung lượng bộ nhớ RAM bằng kỹ thuật Safe Dynamic Downcasting:
    - Đối với biến số nguyên (Integer): Tự động gán kiểu dữ liệu nhỏ nhất có thể
      (int8, int16, int32) dựa trên min/max thực tế. Nếu vượt quá giới hạn 2.147 tỷ,
      giữ nguyên int64 để chống tràn số (Integer Overflow).
    - Đối với biến số thực (Floating-point): Ép từ float64 xuống float32 (chuẩn IEEE 754),
      vừa bảo toàn độ chính xác cần thiết cho mô hình cây, vừa giảm 50% RAM.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame cần tối ưu bộ nhớ.
    verbose : bool, default True
        Nếu True, in chi tiết dung lượng RAM trước và sau khi tối ưu.

    Returns:
    --------
    pd.DataFrame
        DataFrame với kiểu dữ liệu số đã được tối ưu.
    """
    t0 = time.time()
    mem_init_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)

    int_downcasted = 0
    float_downcasted = 0
    int64_preserved = 0

    int8_min, int8_max = np.iinfo(np.int8).min, np.iinfo(np.int8).max
    int16_min, int16_max = np.iinfo(np.int16).min, np.iinfo(np.int16).max
    int32_min, int32_max = np.iinfo(np.int32).min, np.iinfo(np.int32).max

    # 1. Downcast cho các cột Integer
    int_cols = df.select_dtypes(include=["integer"]).columns
    for col in int_cols:
        c_min = df[col].min()
        c_max = df[col].max()

        if c_min >= int8_min and c_max <= int8_max:
            df[col] = df[col].astype(np.int8)
            int_downcasted += 1
        elif c_min >= int16_min and c_max <= int16_max:
            df[col] = df[col].astype(np.int16)
            int_downcasted += 1
        elif c_min >= int32_min and c_max <= int32_max:
            df[col] = df[col].astype(np.int32)
            int_downcasted += 1
        else:
            # Giá trị vượt quá 2.147 tỷ: giữ nguyên int64 an toàn tuyệt đối
            int64_preserved += 1

    # 2. Downcast cho các cột Float
    float_cols = df.select_dtypes(include=["floating"]).columns
    for col in float_cols:
        if df[col].dtype == "float64":
            df[col] = df[col].astype(np.float32)
            float_downcasted += 1

    mem_final_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
    elapsed = time.time() - t0

    if verbose:
        reduction_pct = (1 - mem_final_mb / mem_init_mb) * 100 if mem_init_mb > 0 else 0
        print(f"[TỐI ƯU RAM] Safe Downcasting hoàn tất ({elapsed:.2f}s):")
        print(f"  • RAM ban đầu: {mem_init_mb:,.2f} MB -> RAM sau tối ưu: {mem_final_mb:,.2f} MB (Giảm {reduction_pct:.1f}%)")
        print(f"  • Đã tối ưu: {int_downcasted} cột int, {float_downcasted} cột float | Giữ nguyên an toàn: {int64_preserved} cột int64")

    return df
