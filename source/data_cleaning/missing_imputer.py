"""
Module: missing_imputer.py
Package: source.data_cleaning
Dự án: PRJ-01 - Dự đoán Rủi ro Vỡ nợ Tín dụng (Home Credit Default Risk)
Mô tả: Bước 4 trong pipeline làm sạch dữ liệu:
       - Phân loại & Xử lý dữ liệu khuyết thiếu (Missing Data Strategy).
       - Khuyết thiếu có cấu trúc (Structural Missingness):
         + Với nhóm biến thẻ tín dụng (tiền tố 'CC_'), khách hàng không mở thẻ -> tạo flag 'NO_CC'.
         + Gán giá trị 0 cho toàn bộ các biến số thuộc nhóm 'CC_'.
       - Khuyết thiếu ngẫu nhiên:
         + Biến định lượng (Numerical): Điền bằng giá trị Trung vị (Median) học từ tập train.
         + Biến định tính (Categorical): Điền bằng nhãn 'Missing'.
       - Chống Data Leakage bằng cách lưu lại các giá trị Median từ tập train.
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


def handle_missing_data(
    df: pd.DataFrame,
    is_train: bool = True,
    cc_prefix: str = "CC_",
    cc_flag_col: str = "NO_CC",
    imputer_state: Optional[Dict[str, Any]] = None,
    protected_cols: Optional[List[str]] = None,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Phân loại và xử lý toàn diện dữ liệu khuyết thiếu:

    1. Khuyết thiếu có cấu trúc (Structural Missingness):
       - Nhận diện các cột liên quan đến Thẻ tín dụng (tiền tố 'CC_').
       - Khách hàng không phát sinh lịch sử thẻ tín dụng (tất cả các cột CC_ đều NaN)
         sẽ được đánh dấu cờ NO_CC = 1 (ngược lại có thẻ tín dụng thì NO_CC = 0).
       - Điền giá trị 0 cho tất cả các cột CC_ (vì không có thẻ tương đương với
         dư nợ = 0, phát sinh = 0, số ngày quá hạn = 0).

    2. Khuyết thiếu ngẫu nhiên (Random Missingness):
       - Biến phân loại (Categorical: object / category): Điền bằng chuỗi 'Missing'.
       - Biến định lượng (Numerical): Điền bằng giá trị Trung vị (Median).
         * Khi is_train=True: Tính median trên tập train và lưu vào imputer_state.
         * Khi is_train=False: Dùng các median đã học để điền cho tập test (chống Data Leakage).

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame cần xử lý dữ liệu khuyết thiếu.
    is_train : bool, default True
        Xác định chế độ chạy (True: huấn luyện; False: kiểm định/test).
    cc_prefix : str, default 'CC_'
        Tiền tố định danh các biến thuộc bảng thẻ tín dụng.
    cc_flag_col : str, default 'NO_CC'
        Tên cột biến cờ chỉ báo khách hàng không có thẻ tín dụng.
    imputer_state : Optional[Dict[str, Any]], default None
        Dictionary lưu trữ trạng thái điền khuyết:
        {'medians': {col: float}, 'categorical_cols': [...], 'cc_cols': [...]}.
    protected_cols : Optional[List[str]], default None
        Danh sách các cột đặc biệt không áp dụng điền median/missing (ví dụ: 'SK_ID_CURR', 'TARGET').
    verbose : bool, default True
        Nếu True, in chi tiết log kiểm tra (Validation Summary).

    Returns:
    --------
    Tuple[pd.DataFrame, Dict[str, Any]]
        (DataFrame đã được điền khuyết hoàn chỉnh, Dictionary trạng thái imputer_state)
    """
    if protected_cols is None:
        protected_cols = ["SK_ID_CURR", "TARGET", "DAYS_EMPLOYED_ANOM"]

    total_nulls_before = int(df.isnull().sum().sum())
    n_rows = len(df)

    # Khởi tạo hoặc nạp imputer_state
    if is_train:
        imputer_state = {
            "medians": {},
            "categorical_cols": [],
            "cc_cols": [],
        }
    else:
        if imputer_state is None:
            raise ValueError(
                "Khi is_train=False, bắt buộc phải truyền imputer_state đã lưu từ tập train."
            )

    # =========================================================================
    # 1. KHUYẾT THIẾU CÓ CẤU TRÚC: NHÓM THẺ TÍN DỤNG (CC_)
    # =========================================================================
    if is_train:
        cc_cols = [c for c in df.columns if c.startswith(cc_prefix)]
        imputer_state["cc_cols"] = cc_cols
    else:
        cc_cols = [c for c in imputer_state.get("cc_cols", []) if c in df.columns]

    no_cc_count = 0
    if cc_cols:
        # Nếu tất cả các cột CC_ đều NaN -> Khách hàng chưa từng mở thẻ tín dụng
        # (hoặc cột CC_ đầu tiên là null)
        is_no_cc = df[cc_cols].isnull().all(axis=1)
        no_cc_count = int(is_no_cc.sum())
        df[cc_flag_col] = is_no_cc.astype(np.int32)

        # Gán giá trị 0 cho nhóm biến thẻ tín dụng
        df[cc_cols] = df[cc_cols].fillna(0.0)

    # Thêm cờ NO_CC vào danh sách bảo vệ
    effective_protected = list(set(protected_cols + [cc_flag_col] + cc_cols))

    # =========================================================================
    # 2. KHUYẾT THIẾU NGẪU NHIÊN: BIẾN ĐỊNH TÍNH (CATEGORICAL)
    # =========================================================================
    if is_train:
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        cat_cols = [c for c in cat_cols if c not in effective_protected]
        imputer_state["categorical_cols"] = cat_cols
    else:
        cat_cols = [
            c for c in imputer_state.get("categorical_cols", []) if c in df.columns
        ]

    for col in cat_cols:
        if df[col].isnull().any():
            if isinstance(df[col].dtype, pd.CategoricalDtype):
                if "Missing" not in df[col].cat.categories:
                    df[col] = df[col].cat.add_categories("Missing")
            df[col] = df[col].fillna("Missing")

    # =========================================================================
    # 3. KHUYẾT THIẾU NGẪU NHIÊN: BIẾN ĐỊNH LƯỢNG (NUMERICAL)
    # =========================================================================
    if is_train:
        num_cols = df.select_dtypes(include=["number"]).columns.tolist()
        num_cols = [c for c in num_cols if c not in effective_protected]

        # Tính median cho các cột số có chứa giá trị khuyết thiếu (hoặc toàn bộ num_cols)
        medians_dict = {}
        cols_with_null = [c for c in num_cols if df[c].isnull().any()]
        if cols_with_null:
            calc_medians = df[cols_with_null].median()
            for col, val in calc_medians.items():
                medians_dict[col] = float(val)
        imputer_state["medians"] = medians_dict
    else:
        medians_dict = imputer_state.get("medians", {})

    # Điền giá trị trung vị
    if medians_dict:
        # Lọc các cột có trong DataFrame hiện tại
        active_medians = {k: v for k, v in medians_dict.items() if k in df.columns}
        df.fillna(value=active_medians, inplace=True)

    total_nulls_after = int(df.isnull().sum().sum())

    # In log nghiệm thu (Validation Summary)
    if verbose:
        cc_dist = df[cc_flag_col].value_counts().to_dict() if cc_flag_col in df.columns else {}
        print("\n" + "=" * 80)
        print("BƯỚC 4: PHÂN LOẠI & XỬ LÝ DỮ LIỆU KHUYẾT THIẾU (MISSING DATA STRATEGY)")
        print("=" * 80)
        print(f"• Tổng số ô khuyết thiếu trước xử lý: {total_nulls_before:,d} ô")
        print("\n▶ 1. KHUYẾT THIẾU CÓ CẤU TRÚC (STRUCTURAL MISSINGNESS):")
        print(f"  • Nhóm biến thẻ tín dụng ('{cc_prefix}*')     : {len(cc_cols):,d} cột")
        print(f"  • Tạo biến cờ phân loại          : '{cc_flag_col}' (1: Chưa từng mở thẻ, 0: Đã có thẻ)")
        print(f"  • Số khách hàng không có thẻ     : {no_cc_count:,d} / {n_rows:,d} hồ sơ ({no_cc_count/n_rows*100:.2f}%)")
        print(f"  • Giá trị gán cho các cột {cc_prefix:<6} : 0.0 (Dư nợ, chi tiêu, quá hạn đều bằng 0)")

        print("\n▶ 2. KHUYẾT THIẾU NGẪU NHIÊN (RANDOM MISSINGNESS):")
        print(f"  • Biến định tính (Categorical)    : {len(cat_cols):,d} cột (Điền nhãn 'Missing')")
        print(f"  • Biến định lượng (Numerical)      : {len(medians_dict):,d} cột (Điền giá trị Trung vị - Median)")

        print("\n▶ 3. KẾT QUẢ NGHIỆM THU KHUYẾT THIẾU:")
        print(f"  • Tổng số ô NaN còn lại          : {total_nulls_after:,d} ô")
        print(f"  • Đánh giá hoàn tất điền khuyết  : {'✓ ĐẠT - DỮ LIỆU SẠCH 100%' if total_nulls_after == 0 else '✗ CÒN Ô KHUYẾT'}")
        print("-" * 80)

    return df, imputer_state
