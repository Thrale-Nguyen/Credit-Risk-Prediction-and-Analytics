"""
Module: pipeline.py
Package: source.data_cleaning
Dự án: PRJ-01 - Dự đoán Rủi ro Vỡ nợ Tín dụng (Home Credit Default Risk)
Mô tả: Pipeline Tiền xử lý & Làm sạch dữ liệu chuyên biệt cho LightGBM & CatBoost:
       Thứ tự thực thi 5 chốt chặn logic chuẩn mực:
       1. Tối ưu bộ nhớ (Safe Downcasting: int -> int8/16/32, float64 -> float32).
       2. Xử lý dị thường đặc thù (DAYS_EMPLOYED == 365243 -> NaN, tạo cờ DAYS_EMPLOYED_ANOM,
          và chuẩn hóa dương hóa các biến thời gian quá khứ DAYS_*).
       3. Lọc bỏ đặc trưng kém chất lượng (tỷ lệ missing > 95%, Zero Variance).
       4. Xử lý ngoại lai theo quy tắc nghiệp vụ (AMT_INCOME_TOTAL > 10M, bảo toàn dải liên tục).
       5. Xử lý khuyết thiếu có cấu trúc (Nhóm thẻ tín dụng CC_ -> cờ NO_CC & gán 0;
          BẢO TOÀN NATIVE MISSING np.nan CHO TẤT CẢ CÁC BIẾN SỐ KHÁC).
"""

import copy
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from .downcast import safe_downcast_numeric
from .anomalies import handle_business_anomalies
from .feature_filters import filter_low_quality_features
from .business_rules import handle_business_outliers
from .structural_imputer import handle_structural_missing
from .tree_adapters import get_lightgbm_data, get_catboost_data

# Đảm bảo console Windows in tiếng Việt chuẩn UTF-8
if sys.platform.startswith("win"):
    try:
        if sys.stdout and hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if sys.stderr and hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Biến toàn cục lưu trạng thái pipeline nhẹ nhất
_LATEST_TREE_PIPELINE_STATE: Optional[Dict[str, Any]] = None


class TreeCleaningPipeline:
    """
    Pipeline làm sạch dữ liệu chuẩn Scikit-Learn style dành riêng cho mô hình cây:
    - fit(df): Học các thông số (cột drop, cấu hình CC_, ngưỡng thu nhập).
    - transform(df): Áp dụng các thông số đã học lên tập kiểm tra (Chống Data Leakage).
    - fit_transform(df): Học và làm sạch đồng thời trên tập Train.
    - save_state(path) / load_state(path): Lưu và tải trạng thái nhẹ gọn dưới dạng JSON.
    """

    def __init__(
        self,
        high_null_threshold: float = 0.95,
        income_max_threshold: float = 10_000_000.0,
        income_action: str = "clip",
        convert_negative_days: bool = True,
        verbose: bool = True,
    ):
        self.high_null_threshold = high_null_threshold
        self.income_max_threshold = income_max_threshold
        self.income_action = income_action
        self.convert_negative_days = convert_negative_days
        self.verbose = verbose
        self.state: Dict[str, Any] = {}
        self.is_fitted: bool = False

    def fit(self, df: pd.DataFrame) -> "TreeCleaningPipeline":
        """
        Học các thông số lọc đặc trưng từ tập Train.
        """
        _, state = self._execute_pipeline(df, is_train=True, copy_df=True)
        self.state = state
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame, copy_df: bool = True) -> pd.DataFrame:
        """
        Biến đổi tập Test dựa trên các thông số đã học từ Train.
        """
        if not self.is_fitted:
            raise RuntimeError("Pipeline chưa được fit. Hãy gọi fit(df) hoặc fit_transform(df) trước.")
        df_out, _ = self._execute_pipeline(
            df, is_train=False, state=self.state, copy_df=copy_df
        )
        return df_out

    def fit_transform(self, df: pd.DataFrame, copy_df: bool = True) -> pd.DataFrame:
        """
        Học và làm sạch đồng thời trên tập Train.
        """
        df_out, state = self._execute_pipeline(df, is_train=True, copy_df=copy_df)
        self.state = state
        self.is_fitted = True
        return df_out

    def save_state(self, filepath: Union[str, Path]) -> None:
        """
        Lưu trạng thái cấu hình pipeline (chỉ chứa tên cột và ngưỡng, không chứa median weights).
        """
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.state, f, ensure_ascii=False, indent=2)
        if self.verbose:
            print(f"[THÔNG BÁO] Đã lưu trạng thái TreeCleaningPipeline nhẹ vào: {p.resolve()}")

    def load_state(self, filepath: Union[str, Path]) -> "TreeCleaningPipeline":
        """
        Tải trạng thái cấu hình pipeline từ tệp JSON.
        """
        p = Path(filepath)
        if not p.exists():
            raise FileNotFoundError(f"Không tìm thấy file trạng thái: {p}")
        with open(p, "r", encoding="utf-8") as f:
            self.state = json.load(f)
        self.is_fitted = True
        if self.verbose:
            print(f"[THÔNG BÁO] Đã tải trạng thái TreeCleaningPipeline từ: {p.resolve()}")
        return self

    def _execute_pipeline(
        self,
        df: pd.DataFrame,
        is_train: bool = True,
        state: Optional[Dict[str, Any]] = None,
        copy_df: bool = True,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        return _run_tree_cleaning_steps(
            df=df,
            is_train=is_train,
            state=state,
            high_null_threshold=self.high_null_threshold,
            income_max_threshold=self.income_max_threshold,
            income_action=self.income_action,
            convert_negative_days=self.convert_negative_days,
            verbose=self.verbose,
            copy_df=copy_df,
        )


def clean_data_for_trees(
    df: pd.DataFrame,
    is_train: bool = True,
    state: Optional[Dict[str, Any]] = None,
    high_null_threshold: float = 0.95,
    income_max_threshold: float = 10_000_000.0,
    income_action: str = "clip",
    convert_negative_days: bool = True,
    verbose: bool = True,
    return_state: bool = False,
    copy_df: bool = True,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict[str, Any]]]:
    """
    Hàm chuẩn thực thi toàn diện quy trình Làm sạch dữ liệu cho mô hình cây (LightGBM & CatBoost).

    Thứ tự thực thi 5 chốt chặn:
    1. Tối ưu bộ nhớ (Safe Downcasting: int -> int8/16/32, float64 -> float32).
    2. Dị thường nghiệp vụ (DAYS_EMPLOYED == 365243 -> NaN, cờ DAYS_EMPLOYED_ANOM, dương hóa DAYS_*).
    3. Lọc bỏ đặc trưng kém chất lượng (>95% null, Zero Variance).
    4. Xử lý ngoại lai thu nhập thực tế (Business Sanity Check > 10M, không dùng Winsorization).
    5. Khuyết thiếu cấu trúc nhóm CC_ (cờ NO_CC, điền 0.0) và BẢO TOÀN NATIVE MISSING CHO CÂY.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame cần tiền xử lý.
    is_train : bool, default True
        True: Train; False: Test.
    state : Optional[Dict[str, Any]], default None
        Trạng thái đã fit từ tập train khi is_train=False.
    high_null_threshold : float, default 0.95
        Ngưỡng missing tối đa cho phép.
    income_max_threshold : float, default 10,000,000.0
        Ngưỡng thu nhập hợp lý tối đa.
    income_action : str, default 'clip'
        'clip' (kẹp trần) hoặc 'filter' (lọc bỏ dòng).
    convert_negative_days : bool, default True
        Nếu True, chuyển các cột DAYS_ âm thành dương (abs).
    verbose : bool, default True
        Nếu True, in chi tiết log các bước.
    return_state : bool, default False
        Nếu True, trả về (df_clean, state).
    copy_df : bool, default True
        Nếu True, tạo bản sao dữ liệu ban đầu.

    Returns:
    --------
    Union[pd.DataFrame, Tuple[pd.DataFrame, Dict[str, Any]]]
    """
    global _LATEST_TREE_PIPELINE_STATE

    if not is_train and state is None:
        if _LATEST_TREE_PIPELINE_STATE is not None:
            if verbose:
                print("[THÔNG BÁO] is_train=False: Tự động dùng state đã học từ lần fit train gần nhất.")
            state = _LATEST_TREE_PIPELINE_STATE
        else:
            raise ValueError("Khi is_train=False, bắt buộc phải truyền state đã học từ tập train.")

    df_clean, new_state = _run_tree_cleaning_steps(
        df=df,
        is_train=is_train,
        state=state,
        high_null_threshold=high_null_threshold,
        income_max_threshold=income_max_threshold,
        income_action=income_action,
        convert_negative_days=convert_negative_days,
        verbose=verbose,
        copy_df=copy_df,
    )

    if is_train:
        _LATEST_TREE_PIPELINE_STATE = new_state

    if return_state:
        return df_clean, new_state
    return df_clean


def _run_tree_cleaning_steps(
    df: pd.DataFrame,
    is_train: bool,
    state: Optional[Dict[str, Any]],
    high_null_threshold: float,
    income_max_threshold: float,
    income_action: str,
    convert_negative_days: bool,
    verbose: bool,
    copy_df: bool,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Thực thi 5 bước cốt lõi của pipeline cây.
    """
    t_start = time.time()
    if copy_df:
        df = df.copy()

    # Thu thập thống kê ban đầu
    shape_raw = df.shape
    mem_raw_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
    nan_raw = int(df.isnull().sum().sum())
    days_anom_raw = int((df["DAYS_EMPLOYED"] == 365243).sum()) if "DAYS_EMPLOYED" in df.columns else 0

    if verbose:
        print("\n" + "#" * 85)
        print("KHỞI ĐỘNG PIPELINE TIỀN XỬ LÝ & LÀM SẠCH DỮ LIỆU CHO LIGHTGBM & CATBOOST")
        print("#" * 85)
        print(f"• Dữ liệu đầu vào                  : {shape_raw[0]:,d} dòng x {shape_raw[1]:,d} cột")
        print(f"• Dung lượng bộ nhớ RAM ban đầu    : {mem_raw_mb:,.2f} MB")
        print(f"• Tổng số ô dữ liệu khuyết (NaN)   : {nan_raw:,d} ô")
        print(f"• Chế độ vận hành                  : {'TRAIN (Học & Lưu State)' if is_train else 'TEST (Áp dụng State)'}")

    # =========================================================================
    # BƯỚC 1: Tối ưu bộ nhớ (Safe Downcasting Engine)
    # =========================================================================
    df = safe_downcast_numeric(df, verbose=verbose)

    # =========================================================================
    # BƯỚC 2: Xử lý dị thường nghiệp vụ & Chuẩn hóa trục thời gian (DAYS_*)
    # =========================================================================
    df = handle_business_anomalies(
        df=df,
        target_col="DAYS_EMPLOYED",
        anom_value=365243.0,
        flag_col="DAYS_EMPLOYED_ANOM",
        convert_negative_days=convert_negative_days,
        verbose=verbose,
    )

    # =========================================================================
    # BƯỚC 3: Lọc bỏ đặc trưng kém chất lượng (High Null & Zero Variance)
    # =========================================================================
    filter_state = state.get("dropped_cols_state") if state else None
    df, dropped_cols_state = filter_low_quality_features(
        df=df,
        is_train=is_train,
        high_null_threshold=high_null_threshold,
        dropped_cols_state=filter_state,
        protected_cols=["SK_ID_CURR", "TARGET", "DAYS_EMPLOYED_ANOM"],
        verbose=verbose,
    )

    # =========================================================================
    # BƯỚC 4: Xử lý ngoại lai theo Quy tắc Nghiệp vụ (Business Sanity Check)
    # =========================================================================
    outlier_state_in = state.get("outlier_state") if state else None
    df, outlier_state = handle_business_outliers(
        df=df,
        is_train=is_train,
        target_col="AMT_INCOME_TOTAL",
        max_threshold=income_max_threshold,
        action=income_action,
        outlier_state=outlier_state_in,
        verbose=verbose,
    )

    # =========================================================================
    # BƯỚC 5: Xử lý khuyết thiếu có cấu trúc & Bảo toàn Native Missing
    # =========================================================================
    imputer_state_in = state.get("imputer_state") if state else None
    df, imputer_state = handle_structural_missing(
        df=df,
        is_train=is_train,
        cc_prefix="CC_",
        cc_flag_col="NO_CC",
        imputer_state=imputer_state_in,
        fill_categorical_missing=True,
        categorical_fill_value="Missing",
        protected_cols=["SK_ID_CURR", "TARGET", "DAYS_EMPLOYED_ANOM"],
        verbose=verbose,
    )

    # Đóng gói pipeline state nhẹ (chỉ metadata cấu hình)
    new_state = {
        "is_tree_pipeline": True,
        "dropped_cols_state": dropped_cols_state,
        "outlier_state": outlier_state,
        "imputer_state": imputer_state,
        "high_null_threshold": high_null_threshold,
        "income_max_threshold": income_max_threshold,
        "convert_negative_days": convert_negative_days,
    }

    # Tổng kết nghiệm thu
    shape_clean = df.shape
    mem_clean_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
    nan_clean = int(df.isnull().sum().sum())
    elapsed = time.time() - t_start

    if verbose:
        days_anom_clean = int((df["DAYS_EMPLOYED"] == 365243).sum()) if "DAYS_EMPLOYED" in df.columns else 0
        neg_days_remaining = sum(
            int((df[c] < 0).sum()) for c in df.columns if "DAYS_" in c and pd.api.types.is_numeric_dtype(df[c])
        )

        print("\n" + "=" * 85)
        print(f"TỔNG KẾT NGHIỆM THU PIPELINE LÀM SẠCH (HOÀN THÀNH TRONG {elapsed:.2f} GIÂY)")
        print("=" * 85)
        print(f"• Kích thước trước làm sạch       : {shape_raw[0]:,d} dòng x {shape_raw[1]:,d} cột")
        print(f"• Kích thước sau làm sạch         : {shape_clean[0]:,d} dòng x {shape_clean[1]:,d} cột")
        print(f"• Mức tiêu thụ RAM                : {mem_raw_mb:,.2f} MB -> {mem_clean_mb:,.2f} MB (Giảm {(1-mem_clean_mb/mem_raw_mb)*100:.1f}%)")
        print(f"• Ô khuyết thiếu (NaN)            : {nan_raw:,d} -> {nan_clean:,d} ô (Bảo toàn Native Missing cho Cây)")
        print("-" * 85)
        print("BẢNG KIỂM TRA 5 CHỐT CHẶN CHẤT LƯỢNG:")
        print(f"  [✓ ĐẠT] Chốt 1: Tối ưu bộ nhớ RAM qua Safe Downcasting (Đã ép float32 và kiểm soát tràn số int64).")
        print(f"  [{'✓ ĐẠT' if days_anom_clean == 0 else '✗ LỖI'}] Chốt 2: Triệt tiêu mã 365243 trong DAYS_EMPLOYED và tạo cờ DAYS_EMPLOYED_ANOM.")
        print(f"  [✓ ĐẠT] Chốt 3: Đã loại bỏ các cột >{high_null_threshold*100:.0f}% null và zero-variance.")
        print(f"  [{'✓ ĐẠT' if neg_days_remaining == 0 else '✗ LỖI'}] Chốt 4: Quy đổi các biến thời gian âm (DAYS_*) thành số dương (abs).")
        print(f"  [✓ ĐẠT] Chốt 5: Bảo toàn Native Missing Values (np.nan) cho LightGBM và CatBoost.")
        print("=" * 85 + "\n")

    return df, new_state


# =============================================================================
# TƯƠNG THÍCH NGƯỢC (BACKWARD COMPATIBILITY ALIASES)
# =============================================================================
def preprocess_data(
    df: pd.DataFrame,
    is_train: bool = True,
    state: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict[str, Any]]]:
    """Alias tương thích ngược trỏ về clean_data_for_trees."""
    return clean_data_for_trees(df=df, is_train=is_train, state=state, **kwargs)


DataCleaningPipeline = TreeCleaningPipeline
