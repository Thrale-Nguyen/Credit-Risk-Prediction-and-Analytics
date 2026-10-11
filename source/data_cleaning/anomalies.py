"""
Module: anomalies.py
Package: source.data_cleaning
Dự án: PRJ-01 - Dự đoán Rủi ro Vỡ nợ Tín dụng (Home Credit Default Risk)
Mô tả: Bước 2 trong pipeline làm sạch dữ liệu:
       - Xử lý giá trị bất thường đặc thù nghiệp vụ (Business Anomalies):
         + Nhận diện giá trị dị thường DAYS_EMPLOYED = 365243 (mã hóa người nghỉ hưu/thất nghiệp).
         + Tạo biến cờ phân loại DAYS_EMPLOYED_ANOM (1 nếu bằng 365243, ngược lại 0).
         + Thay thế giá trị 365243 trong DAYS_EMPLOYED bằng np.nan để mô hình cây tự xử lý.
         + Thay thế giá trị 365243 trong các cột lịch sử đơn vay (PREV_DAYS_*) thành np.nan.
       - Chốt chặn 4: Quy đổi các giá trị thời gian âm (DAYS_*) thành số dương (abs):
         + Trực quan hóa và hỗ trợ phân tích EDA, trích xuất Feature Importance và SHAP Values.
"""

import sys
from typing import List, Optional
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


def handle_business_anomalies(
    df: pd.DataFrame,
    target_col: str = "DAYS_EMPLOYED",
    anom_value: float = 365243.0,
    flag_col: str = "DAYS_EMPLOYED_ANOM",
    convert_negative_days: bool = True,
    days_cols_to_invert: Optional[List[str]] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Xử lý dị thường nghiệp vụ và chuẩn hóa trục thời gian cho mô hình dạng cây:
    1. Nhận diện giá trị 365243 trong DAYS_EMPLOYED (đại diện người nghỉ hưu/thất nghiệp).
    2. Tạo biến cờ phân loại flag_col: 1 nếu bằng 365243, 0 nếu là giá trị thực tế khác.
    3. Thay thế 365243 trong DAYS_EMPLOYED bằng np.nan (bảo toàn cơ chế Native Missing của LightGBM/CatBoost).
    4. Thay thế giá trị kỹ thuật 365243 trong các cột lịch sử hợp đồng cũ (PREV_DAYS_*) thành np.nan.
    5. Chốt chặn 4: Quy đổi các giá trị thời gian âm trong các cột DAYS_* thành số dương (abs)
       giúp biểu diễn trực giác khoảng cách thời gian đã trôi qua.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame đầu vào cần xử lý dị thường.
    target_col : str, default 'DAYS_EMPLOYED'
        Tên cột số ngày làm việc chứa giá trị dị thường 365243.
    anom_value : float, default 365243.0
        Giá trị kỹ thuật dị thường quy ước.
    flag_col : str, default 'DAYS_EMPLOYED_ANOM'
        Tên cột cờ phân loại dị thường được tạo mới.
    convert_negative_days : bool, default True
        Nếu True, chuyển đổi toàn bộ các cột thời gian quá khứ mang giá trị âm sang số dương.
    days_cols_to_invert : Optional[List[str]], default None
        Danh sách cột DAYS_ cần đảo dấu. Nếu None, tự động quét tất cả các cột chứa 'DAYS_' có giá trị âm.
    verbose : bool, default True
        Nếu True, in chi tiết log kiểm tra và số lượng bản ghi bị ảnh hưởng.

    Returns:
    --------
    pd.DataFrame
        DataFrame đã được xử lý dị thường và chuyển đổi thời gian.
    """
    # 1. Xử lý dị thường DAYS_EMPLOYED == 365243
    if target_col in df.columns:
        anom_mask = df[target_col] == anom_value
        num_anom = int(anom_mask.sum())
        total_rows = len(df)
        pct_anom = (num_anom / total_rows) * 100 if total_rows > 0 else 0.0

        # Tạo cờ nhị phân
        df[flag_col] = anom_mask.astype(np.int8)

        # Gán lại np.nan cho giá trị dị thường 365243
        df.loc[anom_mask, target_col] = np.nan

        if verbose:
            print(f"[DỊ THƯỜNG] Cột '{target_col}':")
            print(f"  • Phát hiện: {num_anom:,} dòng ({pct_anom:.2f}%) mang giá trị dị thường {anom_value:,.0f}.")
            print(f"  • Đã khởi tạo cờ '{flag_col}' (1: Dị thường, 0: Bình thường).")
            print(f"  • Đã thay thế {anom_value:,.0f} trong '{target_col}' bằng np.nan (chuẩn bị cho LightGBM/CatBoost Native Missing).")
    else:
        if verbose:
            print(f"[CẢNH BÁO] Không tìm thấy cột '{target_col}' trong DataFrame.")

    # 2. Xử lý 365243 trong các cột bảng phụ PREV_DAYS_* (nếu có)
    prev_days_cols = [c for c in df.columns if c.startswith("PREV_DAYS_") and "365243" not in c]
    prev_anom_cleared = 0
    for c in prev_days_cols:
        p_mask = df[c] == anom_value
        cnt = int(p_mask.sum())
        if cnt > 0:
            df.loc[p_mask, c] = np.nan
            prev_anom_cleared += 1

    if verbose and prev_anom_cleared > 0:
        print(f"  • Đã làm sạch mã dị thường {anom_value:,.0f} thành np.nan trong {prev_anom_cleared} cột PREV_DAYS_*.")

    # 3. Chốt chặn 4: Dương hóa các đặc trưng thời gian quá khứ (DAYS_*)
    if convert_negative_days:
        if days_cols_to_invert is None:
            # Tự động quét toàn bộ các cột DAYS_ có giá trị âm
            days_candidates = [
                c for c in df.columns
                if "DAYS_" in c and pd.api.types.is_numeric_dtype(df[c])
            ]
            days_cols_to_invert = [
                c for c in days_candidates
                if (df[c] < 0).any()
            ]

        for col in days_cols_to_invert:
            df[col] = df[col].abs()

        if verbose:
            print(f"[CHỐT CHẶN 4] Chuẩn hóa trục thời gian: Đã chuyển đổi {len(days_cols_to_invert)} cột 'DAYS_*' âm sang dương (abs).")

    return df
