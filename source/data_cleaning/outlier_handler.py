"""
Module: outlier_handler.py
Package: source.data_cleaning
Dự án: PRJ-01 - Dự đoán Rủi ro Vỡ nợ Tín dụng (Home Credit Default Risk)
Mô tả: Bước 3 trong pipeline làm sạch dữ liệu:
       - Xử lý ngoại lai tài chính (Financial Outlier Handling).
       - Đối với các biến tài chính có phân phối lệch phải cực đoan (ví dụ: AMT_INCOME_TOTAL):
         + Áp dụng Cắt tỉa phân vị (Winsorization tại phân vị 99th) HOẶC Log-transform (np.log1p).
         + Giữ lại giá trị cận trên (99th percentile) từ tập train để biến đổi cho tập test (tránh Data Leakage).
       - In log nghiệm thu (Validation Summary) xác nhận triệt tiêu hiện tượng lệch phải cực đoan.
"""

import sys
from typing import Dict, List, Optional, Tuple, Union
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


def handle_financial_outliers(
    df: pd.DataFrame,
    is_train: bool = True,
    financial_cols: Optional[List[str]] = None,
    percentile: float = 0.99,
    method: str = "winsorize",
    outlier_caps_state: Optional[Dict[str, float]] = None,
    auto_detect_skewed: bool = False,
    skew_threshold: float = 10.0,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Xử lý các ngoại lai tài chính có phân phối lệch phải cực đoan (Right-skewed Outliers):
    1. Cắt tỉa phân vị (Winsorization tại phân vị 99th): giới hạn cận trên bằng phân vị 99% của tập train.
    2. Biến đổi Log-transform (np.log1p) giúp nén miền giá trị và kéo phân phối về gần chuẩn.
    3. Hỗ trợ kết hợp cả hai phương pháp (method='both').

    Chống Data Leakage:
    - Khi is_train=True: Tính toán quantile(percentile) trên tập train và lưu vào outlier_caps_state.
    - Khi is_train=False: Sử dụng trực tiếp outlier_caps_state từ train để clip dữ liệu test.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame cần xử lý ngoại lai.
    is_train : bool, default True
        Xác định chế độ chạy (True: huấn luyện; False: kiểm định/test).
    financial_cols : Optional[List[str]], default None
        Danh sách các biến tài chính cần xử lý. Mặc định là ['AMT_INCOME_TOTAL'].
    percentile : float, default 0.99
        Phân vị cắt tỉa cận trên (mặc định 0.99 = 99th percentile).
    method : str, default 'winsorize'
        Phương pháp xử lý: 'winsorize' (cắt tỉa phân vị), 'log1p' (log transform), hoặc 'both' (cả hai).
    outlier_caps_state : Optional[Dict[str, float]], default None
        Dictionary lưu các giá trị cận trên đã học: {col_name: upper_bound}.
    auto_detect_skewed : bool, default False
        Nếu True, tự động quét bổ sung tất cả các cột tài chính (chứa 'AMT') có độ lệch skew > skew_threshold.
    skew_threshold : float, default 10.0
        Ngưỡng độ lệch Skewness để kích hoạt auto-detect.
    verbose : bool, default True
        Nếu True, in chi tiết log kiểm tra và bảng thống kê trước/sau.

    Returns:
    --------
    Tuple[pd.DataFrame, Dict[str, float]]
        (DataFrame sau khi xử lý ngoại lai, Dictionary lưu giá trị cận trên của từng cột)
    """
    if financial_cols is None:
        financial_cols = ["AMT_INCOME_TOTAL"]

    # Đảm bảo các cột được chọn tồn tại trong df
    target_cols = [c for c in financial_cols if c in df.columns]

    # Quét bổ sung nếu kích hoạt auto_detect_skewed
    if is_train and auto_detect_skewed:
        candidate_amt = [
            c
            for c in df.columns
            if ("AMT_" in c or "_AMT" in c)
            and c not in target_cols
            and pd.api.types.is_numeric_dtype(df[c])
        ]
        if candidate_amt:
            skews = df[candidate_amt].skew()
            extreme_skewed = skews[skews > skew_threshold].index.tolist()
            # Chỉ nhận các cột có giá trị 99th percentile lớn hơn giá trị nhỏ nhất để tránh triệt tiêu biến
            for c in extreme_skewed:
                q_cap = float(df[c].quantile(percentile))
                if q_cap > df[c].min():
                    target_cols.append(c)

    if not target_cols:
        if verbose:
            print("[CẢNH BÁO] Không tìm thấy cột tài chính nào cần xử lý ngoại lai trong DataFrame.")
        return df, (outlier_caps_state or {})

    if is_train:
        outlier_caps_state = {}
        for col in target_cols:
            cap_val = float(df[col].quantile(percentile))
            outlier_caps_state[col] = cap_val
    else:
        if outlier_caps_state is None:
            raise ValueError(
                "Khi is_train=False, bắt buộc phải truyền outlier_caps_state đã lưu từ tập train."
            )

    # Thống kê trước khi xử lý
    stats_before = {}
    for col in target_cols:
        stats_before[col] = {
            "skew": float(df[col].skew()),
            "min": float(df[col].min()),
            "median": float(df[col].median()),
            "max": float(df[col].max()),
        }

    # Thực hiện xử lý ngoại lai
    for col in target_cols:
        cap_val = outlier_caps_state.get(col, None)

        if method in ("winsorize", "both") and cap_val is not None:
            df[col] = df[col].clip(upper=cap_val)

        if method in ("log1p", "both"):
            # Tránh lỗi log số âm bằng np.maximum(..., 0)
            df[col] = np.log1p(np.maximum(df[col], 0))

    # In log kiểm tra (Validation Summary)
    if verbose:
        print("\n" + "=" * 80)
        print("BƯỚC 3: XỬ LÝ NGOẠI LAI TÀI CHÍNH (FINANCIAL OUTLIER HANDLING)")
        print("=" * 80)
        print(f"• Chế độ xử lý                     : {'TRAIN (Tính toán cận trên & Lưu trữ)' if is_train else 'TEST (Áp dụng cận trên từ Train)'}")
        print(f"• Phương pháp áp dụng              : {method.upper()} (Phân vị {percentile*100:.0f}th Winsorization)")
        print(f"• Số lượng biến tài chính xử lý    : {len(target_cols):,d} biến")
        print("\n▶ BẢNG SO SÁNH ĐẶC TRƯNG TÀI CHÍNH TRƯỚC VÀ SAU KHI XỬ LÝ:")
        print("-" * 80)
        header = f"{'Tên Đặc Trưng':<25} | {'Skew Trước':>10} -> {'Skew Sau':>9} | {'Cận 99th Lưu':>12} | {'Max Cũ -> Mới':>16}"
        print(header)
        print("-" * 80)

        for col in target_cols:
            skew_b = stats_before[col]["skew"]
            skew_a = float(df[col].skew())
            cap_v = outlier_caps_state.get(col, np.nan)
            max_b = stats_before[col]["max"]
            max_a = float(df[col].max())

            cap_str = f"{cap_v:,.0f}" if pd.notnull(cap_v) else "N/A"
            max_str = f"{max_b:,.0f} -> {max_a:,.0f}"

            print(
                f"{col:<25} | {skew_b:>10.2f} -> {skew_a:>9.2f} | {cap_str:>12} | {max_str:>16}"
            )

        print("-" * 80)
        # Kiểm tra xác nhận
        skew_income = float(df["AMT_INCOME_TOTAL"].skew()) if "AMT_INCOME_TOTAL" in df.columns else 0.0
        print(f"• Xác nhận triệt tiêu lệch cực đoan: AMT_INCOME_TOTAL skewness = {skew_income:.2f} (Đã chuẩn hóa: {'✓ ĐẠT' if abs(skew_income) < 3.0 else 'Cần xem xét'})")
        print("-" * 80)

    return df, outlier_caps_state
