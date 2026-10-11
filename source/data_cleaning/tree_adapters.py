"""
Module: tree_adapters.py
Package: source.data_cleaning
Dự án: PRJ-01 - Dự đoán Rủi ro Vỡ nợ Tín dụng (Home Credit Default Risk)
Mô tả: Bộ điều hợp dữ liệu chuyên biệt (Adapters) cho LightGBM và CatBoost:
       - HỦY BỎ One-Hot Encoding (pd.get_dummies / OneHotEncoder): Không làm bùng nổ số cột.
       - HỦY BỎ Scaler (StandardScaler / RobustScaler): Bảo toàn miền giá trị gốc.
       - Chuẩn hóa tên đặc trưng (Feature Name Sanitization): Khử bỏ ký tự đặc biệt
         (khoảng trắng, dấu hai chấm, dấu ngoặc...) chống lỗi JSON parser của LightGBM.
       - Nhánh LightGBM: Ép 16 biến phân loại sang định dạng pandas 'category'.
       - Nhánh CatBoost: Giữ nguyên biến phân loại ở định dạng chuỗi (string) và xuất
         danh sách cat_features (tên cột) để truyền vào Pool / fit().
       - Cung cấp hàm get_lightgbm_data() và get_catboost_data() trả về các tập
         X_train, y_train, X_val, y_val được phân tầng (Stratified Split).
"""

import re
import sys
from typing import List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Đảm bảo console Windows in tiếng Việt chuẩn UTF-8
if sys.platform.startswith("win"):
    try:
        if sys.stdout and hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if sys.stderr and hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Danh sách 16 biến phân loại gốc trong bài toán Home Credit Default Risk
CATEGORICAL_COLS: List[str] = [
    "NAME_CONTRACT_TYPE",
    "CODE_GENDER",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
    "NAME_TYPE_SUITE",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "OCCUPATION_TYPE",
    "WEEKDAY_APPR_PROCESS_START",
    "ORGANIZATION_TYPE",
    "FONDKAPREMONT_MODE",
    "HOUSETYPE_MODE",
    "WALLSMATERIAL_MODE",
    "EMERGENCYSTATE_MODE",
]


def sanitize_feature_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Chuẩn hóa tên cột để triệt tiêu lỗi cú pháp trong LightGBM:
    - LightGBM không chấp nhận các ký tự đặc biệt của JSON (khoảng trắng, [, ], {, }, :, \", ,)
      trong tên đặc trưng.
    - Hàm tự động thay thế mọi ký tự không phải chữ cái, chữ số hoặc gạch dưới bằng '_'.
    """
    clean_cols = {c: re.sub(r"[^A-Za-z0-9_]+", "_", str(c)) for c in df.columns}
    return df.rename(columns=clean_cols)


def detect_categorical_columns(df: pd.DataFrame) -> List[str]:
    """
    Tự động nhận diện các cột phân loại có trong DataFrame:
    Kết hợp giữa danh sách chuẩn CATEGORICAL_COLS và các cột có kiểu object / category.
    """
    found = [c for c in CATEGORICAL_COLS if c in df.columns]
    other_obj = [
        c for c in df.columns
        if c not in found
        and c not in ["SK_ID_CURR", "TARGET"]
        and (isinstance(df[c].dtype, pd.CategoricalDtype) or df[c].dtype == "object")
    ]
    return list(dict.fromkeys(found + other_obj))


def get_lightgbm_data(
    df: pd.DataFrame,
    target_col: str = "TARGET",
    id_col: Optional[str] = "SK_ID_CURR",
    test_size: float = 0.2,
    random_state: int = 42,
    sanitize_names: bool = True,
    return_splits: bool = True,
    verbose: bool = True,
) -> Union[Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series], pd.DataFrame]:
    """
    Chuẩn bị dữ liệu đầu ra chuyên biệt cho LightGBM:
    - Chuẩn hóa tên đặc trưng sạch (Sanitize Feature Names), loại bỏ khoảng trắng và ký tự đặc biệt.
    - Ép toàn bộ biến phân loại sang định dạng pandas `category` dtype.
      (LightGBM tận dụng thuật toán phân tách Fisher tối ưu O(K log K) trên histogram).
    - Giữ nguyên các giá trị np.nan ở các biến định lượng (Native Missing Handling).
    - Phân chia tập huấn luyện và kiểm định theo tỷ lệ phân tầng (Stratified Split)
      nhằm bảo toàn tuyệt đối tỷ lệ mất cân bằng nhãn vỡ nợ ~8.07%.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame đã qua các bước làm sạch cơ bản (Clean DataFrame).
    target_col : str, default 'TARGET'
        Tên cột nhãn mục tiêu.
    id_col : Optional[str], default 'SK_ID_CURR'
        Tên cột khóa chính (sẽ được loại khỏi tập đặc trưng X).
    test_size : float, default 0.2
        Tỷ lệ tập kiểm định (Validation set).
    random_state : int, default 42
        Seed ngẫu nhiên bảo đảm tính tái lập.
    sanitize_names : bool, default True
        Nếu True, chuẩn hóa tên cột để triệt tiêu lỗi JSON của LightGBM.
    return_splits : bool, default True
        Nếu True, trả về (X_train, y_train, X_val, y_val). Nếu False, trả về df đã chuyển đổi.
    verbose : bool, default True
        Nếu True, in chi tiết thông số dữ liệu đầu ra.

    Returns:
    --------
    Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series] HOẶC pd.DataFrame
    """
    df_lgb = df.copy()

    # Chuẩn hóa tên đặc trưng chống lỗi JSON của LightGBM
    if sanitize_names:
        df_lgb = sanitize_feature_names(df_lgb)

    # Nhận diện và ép kiểu các cột phân loại sang 'category'
    cat_cols = detect_categorical_columns(df_lgb)
    for c in cat_cols:
        df_lgb[c] = df_lgb[c].astype("category")

    if not return_splits:
        return df_lgb

    # Tách X và y
    drop_cols = [c for c in [target_col, id_col] if c and c in df_lgb.columns]
    X = df_lgb.drop(columns=drop_cols)
    y = df_lgb[target_col] if target_col in df_lgb.columns else None

    if y is None:
        raise ValueError(f"Không tìm thấy cột mục tiêu '{target_col}' trong DataFrame.")

    # Phân tầng dữ liệu Stratified Split
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    if verbose:
        print("\n" + "=" * 80)
        print("ADAPTER 1: XUẤT DỮ LIỆU ĐẦU RA CHO MÔ HÌNH LIGHTGBM (LIGHTGBM ADAPTER)")
        print("=" * 80)
        print(f"• Chuẩn hóa tên đặc trưng           : ĐÃ HOÀN TẤT (Khử bỏ khoảng trắng, dấu hai chấm)")
        print(f"• Số lượng cột phân loại (category) : {len(cat_cols)} cột")
        print(f"• Kích thước tập Train              : X_train = {X_train.shape}, y_train = {y_train.shape}")
        print(f"• Kích thước tập Validation         : X_val   = {X_val.shape}, y_val   = {y_val.shape}")
        print(f"• Tỷ lệ nhãn vỡ nợ (TARGET = 1)     : Train = {y_train.mean()*100:.2f}% | Val = {y_val.mean()*100:.2f}%")
        print(f"• Cơ chế xử lý khuyết thiếu         : Native Missing Value (Giữ nguyên np.nan)")
        print(f"• Trạng thái tương thích            : Sẵn sàng nạp vào lgb.Dataset hoặc LGBMClassifier.fit()")
        print("-" * 80)

    return X_train, y_train, X_val, y_val


def get_catboost_data(
    df: pd.DataFrame,
    target_col: str = "TARGET",
    id_col: Optional[str] = "SK_ID_CURR",
    test_size: float = 0.2,
    random_state: int = 42,
    sanitize_names: bool = True,
    return_splits: bool = True,
    verbose: bool = True,
) -> Union[Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, List[str]], Tuple[pd.DataFrame, List[str]]]:
    """
    Chuẩn bị dữ liệu đầu ra chuyên biệt cho CatBoost:
    - Chuẩn hóa tên đặc trưng an toàn (Sanitize Feature Names).
    - Giữ nguyên các biến phân loại ở định dạng chuỗi (string) và điền nhãn 'Missing'
      cho các giá trị rỗng để CatBoost xử lý tối ưu qua Ordered Target Statistics.
    - Xuất kèm danh sách `cat_features` (danh sách tên cột) để truyền vào Pool(..., cat_features=...).
    - Giữ nguyên các giá trị np.nan ở các biến định lượng (CatBoost NanMode: 'Min' hoặc 'Max').
    - Phân chia tập huấn luyện và kiểm định phân tầng (Stratified Split).

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame đã qua các bước làm sạch cơ bản (Clean DataFrame).
    target_col : str, default 'TARGET'
        Tên cột nhãn mục tiêu.
    id_col : Optional[str], default 'SK_ID_CURR'
        Tên cột khóa chính.
    test_size : float, default 0.2
        Tỷ lệ tập kiểm định.
    random_state : int, default 42
        Seed ngẫu nhiên.
    sanitize_names : bool, default True
        Nếu True, chuẩn hóa tên đặc trưng đồng bộ.
    return_splits : bool, default True
        Nếu True, trả về (X_train, y_train, X_val, y_val, cat_features).
    verbose : bool, default True
        Nếu True, in chi tiết thông số.

    Returns:
    --------
    Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, List[str]]
    """
    df_cb = df.copy()

    # Chuẩn hóa tên cột đồng bộ
    if sanitize_names:
        df_cb = sanitize_feature_names(df_cb)

    # Nhận diện cột phân loại và đưa về chuỗi an toàn
    cat_cols = detect_categorical_columns(df_cb)
    for c in cat_cols:
        df_cb[c] = df_cb[c].fillna("Missing").astype(str)

    if not return_splits:
        return df_cb, cat_cols

    # Tách X và y
    drop_cols = [c for c in [target_col, id_col] if c and c in df_cb.columns]
    X = df_cb.drop(columns=drop_cols)
    y = df_cb[target_col] if target_col in df_cb.columns else None

    if y is None:
        raise ValueError(f"Không tìm thấy cột mục tiêu '{target_col}' trong DataFrame.")

    # Phân tầng dữ liệu Stratified Split
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    if verbose:
        print("\n" + "=" * 80)
        print("ADAPTER 2: XUẤT DỮ LIỆU ĐẦU RA CHO MÔ HÌNH CATBOOST (CATBOOST ADAPTER)")
        print("=" * 80)
        print(f"• Chuẩn hóa tên đặc trưng           : ĐÃ HOÀN TẤT")
        print(f"• Số lượng cột phân loại (string)   : {len(cat_cols)} cột")
        print(f"• Danh sách cat_features            : {cat_cols[:5]} ... (+ {len(cat_cols)-5} cột)")
        print(f"• Kích thước tập Train              : X_train = {X_train.shape}, y_train = {y_train.shape}")
        print(f"• Kích thước tập Validation         : X_val   = {X_val.shape}, y_val   = {y_val.shape}")
        print(f"• Tỷ lệ nhãn vỡ nợ (TARGET = 1)     : Train = {y_train.mean()*100:.2f}% | Val = {y_val.mean()*100:.2f}%")
        print(f"• Trạng thái tương thích            : Sẵn sàng nạp vào cb.Pool(..., cat_features=cat_features)")
        print("-" * 80)

    return X_train, y_train, X_val, y_val, cat_cols
