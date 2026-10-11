"""
Module: anomalies.py
Package: source.data_cleaning
Dự án: PRJ-01 - Dự đoán Rủi ro Vỡ nợ Tín dụng (Home Credit Default Risk)
Mô tả: Bước 1 trong pipeline làm sạch dữ liệu:
       - Xử lý giá trị bất thường đặc thù nghiệp vụ (Business Anomalies).
       - Nhận diện giá trị dị thường DAYS_EMPLOYED = 365243 (mã hóa người nghỉ hưu/thất nghiệp).
       - Tạo biến cờ phân loại DAYS_EMPLOYED_ANOM (1 nếu bằng 365243, ngược lại 0).
       - Thay thế giá trị 365243 trong DAYS_EMPLOYED bằng np.nan để phục vụ điền trung vị ở bước sau.
"""

import sys
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


def handle_business_anomalies(
    df: pd.DataFrame,
    target_col: str = "DAYS_EMPLOYED",
    anom_value: float = 365243.0,
    flag_col: str = "DAYS_EMPLOYED_ANOM",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Xử lý giá trị bất thường nghiệp vụ đối với biến DAYS_EMPLOYED:
    - Nhận diện giá trị 365243 (khoảng 1000 năm, giá trị gán cứng đại diện cho người nghỉ hưu/thất nghiệp).
    - Tạo biến cờ phân loại flag_col: 1 nếu bằng 365243, 0 nếu là giá trị thực tế khác.
    - Thay thế 365243 trong target_col bằng np.nan.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame đầu vào.
    target_col : str, default 'DAYS_EMPLOYED'
        Tên cột chứa giá trị bất thường cần xử lý.
    anom_value : float, default 365243.0
        Giá trị bất thường đặc thù nghiệp vụ.
    flag_col : str, default 'DAYS_EMPLOYED_ANOM'
        Tên biến cờ chỉ báo được tạo mới.
    verbose : bool, default True
        Nếu True, in chi tiết log kiểm tra (Validation Summary).

    Returns:
    --------
    pd.DataFrame
        DataFrame sau khi tạo biến cờ và thay thế giá trị dị thường bằng np.nan.
    """
    if target_col not in df.columns:
        if verbose:
            print(f"[CẢNH BÁO] Cột '{target_col}' không tồn tại trong dữ liệu. Bỏ qua Bước 1.")
        return df

    # 1. Phát hiện và tạo cờ bất thường
    anom_mask = (df[target_col] == anom_value)
    anom_count = int(anom_mask.sum())
    total_rows = len(df)
    anom_pct = (anom_count / total_rows * 100.0) if total_rows > 0 else 0.0

    # Gán cờ chỉ báo kiểu int32
    df[flag_col] = anom_mask.astype(np.int32)

    # 2. Thay thế giá trị dị thường bằng np.nan
    df[target_col] = df[target_col].replace(anom_value, np.nan)

    # 3. Log kiểm tra (Validation Summary)
    if verbose:
        null_count_after = int(df[target_col].isnull().sum())
        rem_anom = int((df[target_col] == anom_value).sum())
        flag_dist = df[flag_col].value_counts().to_dict()

        print("\n" + "=" * 80)
        print("BƯỚC 1: XỬ LÝ GIÁ TRỊ BẤT THƯỜNG NGHIỆP VỤ (BUSINESS ANOMALIES)")
        print("=" * 80)
        print(f"• Cột kiểm tra                     : {target_col}")
        print(f"• Giá trị bất thường nhận diện     : {anom_value:,.0f}")
        print(f"• Số lượng bản ghi bất thường      : {anom_count:,d} / {total_rows:,d} dòng ({anom_pct:.2f}%)")
        print(f"• Biến cờ mới tạo                  : {flag_col} (1: Bất thường, 0: Bình thường)")
        print(f"  └ Phân phối cờ                   : Flag=1: {flag_dist.get(1, 0):,d} ({flag_dist.get(1, 0)/total_rows*100:.2f}%) | Flag=0: {flag_dist.get(0, 0):,d} ({flag_dist.get(0, 0)/total_rows*100:.2f}%)")
        print(f"• Số lượng NaN mới trong {target_col:<10}: {null_count_after:,d} dòng")
        print(f"• Kiểm tra số lượng {anom_value:,.0f} còn lại: {rem_anom} giá trị (Đã sạch hoàn toàn: {'✓ ĐẠT' if rem_anom == 0 else '✗ LỖI'})")
        print("-" * 80)

    return df
