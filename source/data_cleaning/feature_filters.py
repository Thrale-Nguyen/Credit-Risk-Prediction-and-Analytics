"""
Module: feature_filters.py
Package: source.data_cleaning
Dự án: PRJ-01 - Dự đoán Rủi ro Vỡ nợ Tín dụng (Home Credit Default Risk)
Mô tả: Bước 2 trong pipeline làm sạch dữ liệu:
       - Lọc bỏ đặc trưng kém chất lượng (Low/Zero Variance & High Null Filter).
       - Quét và loại bỏ các cột có tỷ lệ missing > 95%.
       - Loại bỏ các cột hằng số (Zero Variance / nunique <= 1).
       - In danh sách chi tiết các cột bị loại bỏ và lưu trữ danh sách để áp dụng cho tập test (tránh Data Leakage).
"""

import sys
from typing import Dict, List, Optional, Tuple
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


def filter_low_quality_features(
    df: pd.DataFrame,
    is_train: bool = True,
    high_null_threshold: float = 0.95,
    dropped_cols_state: Optional[Dict[str, List[str]]] = None,
    protected_cols: Optional[List[str]] = None,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """
    Quét và loại bỏ các cột có chất lượng kém:
    1. Cột có tỷ lệ khuyết thiếu (Missing Ratio) > high_null_threshold (mặc định > 95%).
    2. Cột hằng số (Zero Variance): chỉ có đúng 1 giá trị duy nhất (không mang lại thông tin phân loại).

    Bảo toàn tính nhất quán giữa Train và Test:
    - Khi is_train=True: Quét dữ liệu train, xác định các cột cần loại bỏ và lưu vào dropped_cols_state.
    - Khi is_train=False: Sử dụng danh sách dropped_cols_state đã học từ train để drop đồng bộ (chống Data Leakage).

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame cần lọc đặc trưng.
    is_train : bool, default True
        Xác định chế độ chạy (True: huấn luyện / tính toán; False: kiểm định / biến đổi theo thông số train).
    high_null_threshold : float, default 0.95
        Ngưỡng tỷ lệ khuyết thiếu tối đa cho phép (0.95 = 95%).
    dropped_cols_state : Optional[Dict[str, List[str]]], default None
        Dictionary lưu vết các cột bị loại: {'high_null': [...], 'zero_var': [...], 'all': [...]}.
    protected_cols : Optional[List[str]], default None
        Danh sách các cột được bảo vệ (không bao giờ bị drop, ví dụ 'SK_ID_CURR', 'TARGET').
    verbose : bool, default True
        Nếu True, in chi tiết log kiểm tra và danh sách cột bị loại bỏ.

    Returns:
    --------
    Tuple[pd.DataFrame, Dict[str, List[str]]]
        (DataFrame sau khi drop, Dictionary danh sách các cột đã loại bỏ)
    """
    if protected_cols is None:
        protected_cols = ["SK_ID_CURR", "TARGET", "DAYS_EMPLOYED_ANOM"]

    cols_before = df.shape[1]

    if is_train:
        # =====================================================================
        # 1. Quét cột có tỷ lệ missing > 95%
        # =====================================================================
        null_ratios = df.isnull().mean()
        high_null_mask = null_ratios > high_null_threshold
        high_null_candidates = null_ratios[high_null_mask].index.tolist()
        high_null_cols = [c for c in high_null_candidates if c not in protected_cols]

        # =====================================================================
        # 2. Quét cột hằng số (Zero Variance)
        # =====================================================================
        # Loại trừ các cột đã bị gắn mác high_null và các cột được bảo vệ
        remaining_candidates = [
            c for c in df.columns if c not in high_null_cols and c not in protected_cols
        ]

        zero_var_cols = []
        for col in remaining_candidates:
            # nunique(dropna=True) <= 1 nghĩa là tất cả các dòng có giá trị đều cùng 1 giá trị (phương sai = 0)
            if df[col].nunique(dropna=True) <= 1:
                zero_var_cols.append(col)

        all_dropped = list(dict.fromkeys(high_null_cols + zero_var_cols))
        dropped_cols_state = {
            "high_null": high_null_cols,
            "zero_var": zero_var_cols,
            "all": all_dropped,
        }
    else:
        # Chế độ Test: Sử dụng danh sách đã fit từ Train
        if dropped_cols_state is None:
            raise ValueError(
                "Khi is_train=False, bắt buộc phải truyền dropped_cols_state đã học từ tập train."
            )
        all_dropped = [c for c in dropped_cols_state.get("all", []) if c in df.columns]

    # Thực hiện loại bỏ cột
    df = df.drop(columns=all_dropped, errors="ignore")
    cols_after = df.shape[1]

    # In log nghiệm thu (Validation Summary)
    if verbose:
        high_null_list = dropped_cols_state.get("high_null", [])
        zero_var_list = dropped_cols_state.get("zero_var", [])

        print("\n" + "=" * 80)
        print("BƯỚC 2: LỌC BỎ ĐẶC TRƯNG KÉM CHẤT LƯỢNG (LOW/ZERO VARIANCE & HIGH NULL FILTER)")
        print("=" * 80)
        print(f"• Chế độ xử lý                     : {'TRAIN (Khám phá & Xác lập)' if is_train else 'TEST (Áp dụng bộ lọc từ Train)'}")
        print(f"• Số lượng cột ban đầu             : {cols_before:,d} cột")
        print(f"• Tổng số cột bị loại bỏ           : {len(all_dropped):,d} cột")
        print(f"  ├ 1. Cột tỷ lệ missing > {high_null_threshold*100:.0f}%  : {len(high_null_list):,d} cột")
        if high_null_list:
            for idx, col in enumerate(high_null_list, 1):
                # Tính tỷ lệ null nếu cột còn trong df ban đầu
                ratio = null_ratios[col] * 100 if is_train and col in null_ratios else 0.0
                print(f"     [{idx:02d}] {col:<45} (Missing: {ratio:.2f}%)")

        print(f"  └ 2. Cột hằng số (Zero Variance) : {len(zero_var_list):,d} cột")
        if zero_var_list:
            for idx, col in enumerate(zero_var_list, 1):
                print(f"     [{idx:02d}] {col:<45} (Chỉ chứa 1 giá trị duy nhất)")

        print(f"• Số lượng cột còn lại             : {cols_after:,d} cột")
        print(f"• Xác nhận không còn cột >{high_null_threshold*100:.0f}% null: {'✓ ĐẠT' if (df.isnull().mean() <= high_null_threshold).all() else '✗ CÒN CỘT LỖI'}")
        print("-" * 80)

    return df, dropped_cols_state
