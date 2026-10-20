"""
Module: structural_imputer.py
Package: source.data_cleaning
Dự án: PRJ-01 - Dự đoán Rủi ro Vỡ nợ Tín dụng (Home Credit Default Risk)
Mô tả: Bước 5 trong pipeline làm sạch dữ liệu tối ưu cho LightGBM & CatBoost:
       - Phân loại và xử lý khuyết thiếu có cấu trúc (Structural Missingness):
         + Với 92+ biến thẻ tín dụng ('CC_'): Tạo cờ NO_CC = 1 cho khách chưa mở thẻ tín dụng.
         + Điền giá trị 0.0 cho các biến số 'CC_' (vì không có thẻ = 0 dư nợ, 0 phát sinh, 0 trễ hạn).
       - BẢO TOÀN NATIVE MISSING CHO TOÀN BỘ BIẾN SỐ CÒN LẠI:
         + LOẠI BỎ hoàn toàn việc điền Median cho các biến định lượng khác.
         + Giữ nguyên np.nan để LightGBM và CatBoost tự động học hướng phân nhánh tối ưu (Missing Value Split Gain).
       - Với biến định tính (Categorical): Điền nhãn 'Missing' cho các giá trị khuyết thiếu.
"""

import sys
from typing import Any, Dict, List, Optional, Tuple
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


def handle_structural_missing(
    df: pd.DataFrame,
    is_train: bool = True,
    cc_prefix: str = "CC_",
    cc_flag_col: str = "NO_CC",
    imputer_state: Optional[Dict[str, Any]] = None,
    fill_categorical_missing: bool = True,
    categorical_fill_value: str = "Missing",
    protected_cols: Optional[List[str]] = None,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Xử lý khuyết thiếu có cấu trúc và bảo toàn Native Missing Values:
    1. Khuyết thiếu có cấu trúc (Thẻ tín dụng 'CC_'):
       - Khách hàng không có thẻ tín dụng (tất cả các cột CC_ đều NaN) -> tạo cờ NO_CC = 1 (ngược lại 0).
       - Điền 0.0 cho toàn bộ các biến số thuộc nhóm CC_ (phản ánh đúng bản chất kinh tế).
    2. Khuyết thiếu ngẫu nhiên:
       - BIẾN ĐỊNH LƯỢNG (Numerical): GIỮ NGUYÊN np.nan (KHÔNG điền Median/Mean).
       - BIẾN ĐỊNH TÍNH (Categorical): Điền chuỗi 'Missing' để tạo nhóm danh mục riêng.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame cần xử lý khuyết thiếu.
    is_train : bool, default True
        True: Chế độ huấn luyện; False: Chế độ kiểm định/test.
    cc_prefix : str, default 'CC_'
        Tiền tố nhận diện các cột thẻ tín dụng.
    cc_flag_col : str, default 'NO_CC'
        Tên cột cờ không có thẻ tín dụng.
    imputer_state : Optional[Dict[str, Any]], default None
        Dictionary lưu cấu hình các cột CC_ và cột phân loại để áp dụng cho test set.
    fill_categorical_missing : bool, default True
        Nếu True, điền 'Missing' cho các giá trị null ở cột phân loại.
    categorical_fill_value : str, default 'Missing'
        Giá trị điền cho cột phân loại.
    protected_cols : Optional[List[str]], default None
        Danh sách cột bảo vệ (không can thiệp).
    verbose : bool, default True
        Nếu True, in chi tiết log kiểm tra.

    Returns:
    --------
    Tuple[pd.DataFrame, Dict[str, Any]]
        (DataFrame đã xử lý, Dictionary imputer_state)
    """
    if protected_cols is None:
        protected_cols = ["SK_ID_CURR", "TARGET", "DAYS_EMPLOYED_ANOM"]

    total_rows = len(df)
    nan_before = int(df.isnull().sum().sum())

    # =========================================================================
    # 1. Khuyết thiếu có cấu trúc: Nhóm thẻ tín dụng (CC_)
    # =========================================================================
    if is_train:
        cc_cols = [c for c in df.columns if c.startswith(cc_prefix) and c not in protected_cols]
        # Xác định các cột phân loại
        cat_cols = [
            c for c in df.select_dtypes(include=["object", "category"]).columns
            if c not in protected_cols
        ]
        imputer_state = {
            "cc_cols": cc_cols,
            "cat_cols": cat_cols,
            "categorical_fill_value": categorical_fill_value,
        }
    else:
        if imputer_state is None:
            raise ValueError("Khi is_train=False, bắt buộc phải truyền imputer_state từ tập train.")
        cc_cols = [c for c in imputer_state.get("cc_cols", []) if c in df.columns]
        cat_cols = [c for c in imputer_state.get("cat_cols", []) if c in df.columns]

    no_cc_count = 0
    if cc_cols:
        # Nhận diện khách hàng không có thẻ tín dụng: tất cả cột CC_ đều NaN
        no_cc_mask = df[cc_cols].isnull().all(axis=1)
        no_cc_count = int(no_cc_mask.sum())
        df[cc_flag_col] = no_cc_mask.astype(np.int8)

        # Điền 0.0 cho các cột số thuộc nhóm CC_
        num_cc_cols = [c for c in cc_cols if pd.api.types.is_numeric_dtype(df[c])]
        df[num_cc_cols] = df[num_cc_cols].fillna(0.0)

    # =========================================================================
    # 2. Xử lý cột phân loại (Categorical Features)
    # =========================================================================
    cat_filled_count = 0
    if fill_categorical_missing and cat_cols:
        for c in cat_cols:
            if df[c].isnull().any():
                cat_filled_count += 1
                if pd.api.types.is_categorical_dtype(df[c]):
                    if categorical_fill_value not in df[c].cat.categories:
                        df[c] = df[c].cat.add_categories([categorical_fill_value])
                    df[c] = df[c].fillna(categorical_fill_value)
                else:
                    df[c] = df[c].fillna(categorical_fill_value).astype(str)

    nan_after = int(df.isnull().sum().sum())
    remaining_nan_numeric = int(df.select_dtypes(include=["number"]).isnull().sum().sum())

    if verbose:
        pct_no_cc = (no_cc_count / total_rows) * 100 if total_rows > 0 else 0
        print("\n" + "=" * 80)
        print("BƯỚC 5: XỬ LÝ KHUYẾT THIẾU CẤU TRÚC & BẢO TOÀN NATIVE MISSING CHO CÂY")
        print("=" * 80)
        print(f"• Tổng số ô NaN ban đầu             : {nan_before:,d} ô")
        print(f"• Xử lý có cấu trúc nhóm 'CC_'      : {len(cc_cols)} cột")
        print(f"  ├ Khách hàng chưa mở thẻ (NO_CC) : {no_cc_count:,d} dòng ({pct_no_cc:.2f}%)")
        print(f"  └ Gán 0.0 cho các biến số CC_    : Đã điền 0.0 hoàn tất")
        print(f"• Xử lý biến phân loại              : Đã điền '{categorical_fill_value}' cho {cat_filled_count} cột")
        print(f"• Cơ chế Native Missing cho biến số : BẢO TOÀN TUYỆT ĐỐI {remaining_nan_numeric:,d} ô np.nan")
        print(f"  └ Không dùng Median/Mean Imputation (LightGBM/CatBoost tự động tìm nhánh tối ưu)")
        print(f"• Tổng số ô NaN còn lại             : {nan_after:,d} ô (Chuẩn định dạng cho Tree-based)")
        print("-" * 80)

    return df, imputer_state
