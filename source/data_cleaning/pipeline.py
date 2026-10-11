"""
Module: pipeline.py
Package: source.data_cleaning
Dự án: PRJ-01 - Dự đoán Rủi ro Vỡ nợ Tín dụng (Home Credit Default Risk)
Mô tả: Đóng gói toàn bộ quy trình Tiền xử lý & Làm sạch dữ liệu trong hàm preprocess_data:
       Thứ tự thực thi 4 bước logic:
       1. Xử lý giá trị bất thường đặc thù (DAYS_EMPLOYED = 365243 -> NaN, cờ DAYS_EMPLOYED_ANOM)
       2. Lọc bỏ đặc trưng kém chất lượng (tỷ lệ missing > 95%, Zero Variance)
       3. Xử lý ngoại lai tài chính (Winsorization 99th percentile / Log1p, lưu cận trên cho test)
       4. Phân loại & Xử lý khuyết thiếu (Thẻ tín dụng CC_ -> cờ NO_CC & gán 0; Median cho số; 'Missing' cho phân loại)

       Cung cấp bảng tóm tắt Shape và Log kiểm tra chốt chặn toàn diện.
"""

import copy
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from .anomalies import handle_business_anomalies
from .feature_filters import filter_low_quality_features
from .outlier_handler import handle_financial_outliers
from .missing_imputer import handle_missing_data

# Đảm bảo console Windows in tiếng Việt chuẩn UTF-8
if sys.platform.startswith("win"):
    try:
        if sys.stdout and hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if sys.stderr and hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Lưu cache state mới nhất phục vụ transform tiện lợi
_LATEST_PIPELINE_STATE: Optional[Dict[str, Any]] = None


class DataCleaningPipeline:
    """
    Lớp điều phối Pipeline Làm sạch & Tiền xử lý dữ liệu chuẩn Scikit-Learn style:
    - fit(df): Học các thông số (cột drop, cận 99th ngoại lai, median, danh mục).
    - transform(df): Áp dụng các thông số đã học lên dữ liệu (chống Data Leakage).
    - fit_transform(df): Học và biến đổi đồng thời trên tập train.
    - save_state(path) / load_state(path): Lưu và tải trạng thái pipeline dạng JSON.
    """

    def __init__(
        self,
        high_null_threshold: float = 0.95,
        percentile_cap: float = 0.99,
        outlier_method: str = "winsorize",
        financial_cols: Optional[List[str]] = None,
        auto_detect_skewed: bool = False,
        skew_threshold: float = 10.0,
        verbose: bool = True,
    ):
        self.high_null_threshold = high_null_threshold
        self.percentile_cap = percentile_cap
        self.outlier_method = outlier_method
        self.financial_cols = financial_cols or ["AMT_INCOME_TOTAL"]
        self.auto_detect_skewed = auto_detect_skewed
        self.skew_threshold = skew_threshold
        self.verbose = verbose
        self.state: Dict[str, Any] = {}
        self.is_fitted: bool = False

    def fit(self, df: pd.DataFrame) -> "DataCleaningPipeline":
        """
        Học toàn bộ các tham số thống kê từ tập huấn luyện (Train Set).
        """
        _, state = self._execute_pipeline(df, is_train=True, copy_df=True)
        self.state = state
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame, copy_df: bool = True) -> pd.DataFrame:
        """
        Biến đổi tập dữ liệu (Test Set / Validation Set) dựa trên các tham số đã học.
        """
        if not self.is_fitted:
            raise RuntimeError("Pipeline chưa được fit. Hãy gọi fit(df) hoặc fit_transform(df) trước.")
        df_out, _ = self._execute_pipeline(
            df, is_train=False, state=self.state, copy_df=copy_df
        )
        return df_out

    def fit_transform(
        self, df: pd.DataFrame, copy_df: bool = True
    ) -> pd.DataFrame:
        """
        Học các tham số và trả về DataFrame đã làm sạch trên tập huấn luyện.
        """
        df_out, state = self._execute_pipeline(
            df, is_train=True, copy_df=copy_df
        )
        self.state = state
        self.is_fitted = True
        return df_out

    def save_state(self, filepath: Union[str, Path]) -> None:
        """
        Lưu trạng thái pipeline ra file JSON để tái sử dụng trong production/serving.
        """
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.state, f, ensure_ascii=False, indent=2)
        if self.verbose:
            print(f"[THÔNG BÁO] Đã lưu trạng thái DataCleaningPipeline vào: {p.resolve()}")

    def load_state(self, filepath: Union[str, Path]) -> "DataCleaningPipeline":
        """
        Tải trạng thái pipeline từ file JSON.
        """
        p = Path(filepath)
        if not p.exists():
            raise FileNotFoundError(f"Không tìm thấy file trạng thái: {p}")
        with open(p, "r", encoding="utf-8") as f:
            self.state = json.load(f)
        self.is_fitted = True
        if self.verbose:
            print(f"[THÔNG BÁO] Đã nạp thành công trạng thái DataCleaningPipeline từ: {p.resolve()}")
        return self

    def _execute_pipeline(
        self,
        df: pd.DataFrame,
        is_train: bool = True,
        state: Optional[Dict[str, Any]] = None,
        copy_df: bool = True,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        return _run_preprocessing_steps(
            df=df,
            is_train=is_train,
            state=state,
            high_null_threshold=self.high_null_threshold,
            percentile_cap=self.percentile_cap,
            outlier_method=self.outlier_method,
            financial_cols=self.financial_cols,
            auto_detect_skewed=self.auto_detect_skewed,
            skew_threshold=self.skew_threshold,
            verbose=self.verbose,
            copy_df=copy_df,
        )


def preprocess_data(
    df: pd.DataFrame,
    is_train: bool = True,
    state: Optional[Dict[str, Any]] = None,
    high_null_threshold: float = 0.95,
    percentile_cap: float = 0.99,
    outlier_method: str = "winsorize",
    financial_cols: Optional[List[str]] = None,
    auto_detect_skewed: bool = False,
    skew_threshold: float = 10.0,
    verbose: bool = True,
    return_state: bool = False,
    copy_df: bool = True,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict[str, Any]]]:
    """
    Hàm chuẩn thực thi toàn diện quy trình Tiền xử lý & Làm sạch dữ liệu.

    Thứ tự thực thi 4 bước logic:
    1. Xử lý giá trị bất thường đặc thù (DAYS_EMPLOYED = 365243 -> NaN, tạo cờ DAYS_EMPLOYED_ANOM)
    2. Lọc bỏ đặc trưng kém chất lượng (tỷ lệ missing > 95%, Zero Variance)
    3. Xử lý ngoại lai tài chính (Cắt tỉa phân vị 99th Winsorization / Log1p, giữ cận trên cho test)
    4. Phân loại & Xử lý dữ liệu khuyết thiếu (Thẻ tín dụng CC_ -> cờ NO_CC & gán 0; Median cho số; 'Missing' cho phân loại)

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame cần tiền xử lý (bảng sau ghép nối).
    is_train : bool, default True
        True: Huấn luyện (tính toán thống kê và lưu state); False: Áp dụng state từ train lên test.
    state : Optional[Dict[str, Any]], default None
        Dictionary chứa các tham số đã học từ train khi is_train=False.
        Nếu is_train=False và state=None, tự động dùng _LATEST_PIPELINE_STATE nếu có.
    high_null_threshold : float, default 0.95
        Ngưỡng missing để lọc cột kém chất lượng.
    percentile_cap : float, default 0.99
        Phân vị cắt tỉa ngoại lai tài chính.
    outlier_method : str, default 'winsorize'
        Phương pháp xử lý ngoại lai ('winsorize', 'log1p', 'both').
    financial_cols : Optional[List[str]], default None
        Danh sách cột tài chính cần xử lý (mặc định ['AMT_INCOME_TOTAL']).
    auto_detect_skewed : bool, default False
        Tự động phát hiện thêm các cột tài chính lệch phải cực đoan.
    skew_threshold : float, default 10.0
        Ngưỡng độ lệch Skewness để kích hoạt auto-detect.
    verbose : bool, default True
        In log kiểm tra chi tiết từng bước và bảng tổng kết.
    return_state : bool, default False
        Nếu True, trả về tuple (df_clean, state). Nếu False, trả về df_clean.
    copy_df : bool, default True
        Nếu True, tạo bản sao DataFrame ở đầu pipeline để tránh làm thay đổi dữ liệu gốc.

    Returns:
    --------
    Union[pd.DataFrame, Tuple[pd.DataFrame, Dict[str, Any]]]
        DataFrame đã được làm sạch hoàn chỉnh (hoặc kèm state nếu return_state=True).
    """
    global _LATEST_PIPELINE_STATE

    if not is_train and state is None:
        if _LATEST_PIPELINE_STATE is not None:
            if verbose:
                print("[THÔNG BÁO] is_train=False và state=None: Tự động sử dụng state đã học từ lần fit train gần nhất.")
            state = _LATEST_PIPELINE_STATE
        else:
            raise ValueError(
                "Khi is_train=False, bắt buộc phải truyền state đã học từ tập train."
            )

    df_clean, new_state = _run_preprocessing_steps(
        df=df,
        is_train=is_train,
        state=state,
        high_null_threshold=high_null_threshold,
        percentile_cap=percentile_cap,
        outlier_method=outlier_method,
        financial_cols=financial_cols or ["AMT_INCOME_TOTAL"],
        auto_detect_skewed=auto_detect_skewed,
        skew_threshold=skew_threshold,
        verbose=verbose,
        copy_df=copy_df,
    )

    if is_train:
        _LATEST_PIPELINE_STATE = new_state

    if return_state:
        return df_clean, new_state
    return df_clean


def _run_preprocessing_steps(
    df: pd.DataFrame,
    is_train: bool,
    state: Optional[Dict[str, Any]],
    high_null_threshold: float,
    percentile_cap: float,
    outlier_method: str,
    financial_cols: List[str],
    auto_detect_skewed: bool,
    skew_threshold: float,
    verbose: bool,
    copy_df: bool,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Hàm nội bộ thực thi 4 bước tuần tự và in validation tổng hợp.
    """
    start_time = time.time()
    if copy_df:
        df = df.copy()

    # Lưu trữ thông tin trạng thái ban đầu
    n_rows_init, n_cols_init = df.shape
    mem_init_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
    total_nulls_init = int(df.isnull().sum().sum())
    income_skew_init = (
        float(df["AMT_INCOME_TOTAL"].skew()) if "AMT_INCOME_TOTAL" in df.columns else np.nan
    )
    days_anom_init = (
        int((df["DAYS_EMPLOYED"] == 365243).sum())
        if "DAYS_EMPLOYED" in df.columns
        else 0
    )

    state_out = copy.deepcopy(state) if state is not None else {}

    if verbose:
        print("\n" + "=" * 85)
        print("BẮT ĐẦU QUY TRÌNH TIỀN XỬ LÝ & LÀM SẠCH DỮ LIỆU (DATA CLEANING PIPELINE)")
        print("=" * 85)
        print(f"• Dữ liệu đầu vào                  : {n_rows_init:,d} dòng x {n_cols_init:,d} cột")
        print(f"• Bộ nhớ RAM tiêu thụ              : {mem_init_mb:,.2f} MB (~{mem_init_mb/1024:.2f} GB)")
        print(f"• Tổng số ô khuyết thiếu ban đầu   : {total_nulls_init:,d} ô")
        print(f"• Chế độ xử lý                     : {'TRAIN PIPELINE (Fit & Transform)' if is_train else 'TEST PIPELINE (Transform only)'}")

    # =========================================================================
    # BƯỚC 1: XỬ LÝ GIÁ TRỊ BẤT THƯỜNG ĐẶC THÙ (BUSINESS ANOMALIES)
    # =========================================================================
    df = handle_business_anomalies(
        df=df,
        target_col="DAYS_EMPLOYED",
        anom_value=365243.0,
        flag_col="DAYS_EMPLOYED_ANOM",
        verbose=verbose,
    )

    # =========================================================================
    # BƯỚC 2: LỌC BỎ ĐẶC TRƯNG KÉM CHẤT LƯỢNG (LOW/ZERO VARIANCE & HIGH NULL)
    # =========================================================================
    dropped_state = state_out.get("dropped_cols", None)
    df, dropped_state = filter_low_quality_features(
        df=df,
        is_train=is_train,
        high_null_threshold=high_null_threshold,
        dropped_cols_state=dropped_state,
        protected_cols=["SK_ID_CURR", "TARGET", "DAYS_EMPLOYED_ANOM"],
        verbose=verbose,
    )
    state_out["dropped_cols"] = dropped_state

    # =========================================================================
    # BƯỚC 3: XỬ LÝ NGOẠI LAI TÀI CHÍNH (FINANCIAL OUTLIER HANDLING)
    # =========================================================================
    caps_state = state_out.get("outlier_caps", None)
    df, caps_state = handle_financial_outliers(
        df=df,
        is_train=is_train,
        financial_cols=financial_cols,
        percentile=percentile_cap,
        method=outlier_method,
        outlier_caps_state=caps_state,
        auto_detect_skewed=auto_detect_skewed,
        skew_threshold=skew_threshold,
        verbose=verbose,
    )
    state_out["outlier_caps"] = caps_state

    # =========================================================================
    # BƯỚC 4: PHÂN LOẠI & XỬ LÝ DỮ LIỆU KHUYẾT THIẾU (MISSING DATA STRATEGY)
    # =========================================================================
    imputer_st = state_out.get("imputer_state", None)
    df, imputer_st = handle_missing_data(
        df=df,
        is_train=is_train,
        cc_prefix="CC_",
        cc_flag_col="NO_CC",
        imputer_state=imputer_st,
        protected_cols=["SK_ID_CURR", "TARGET", "DAYS_EMPLOYED_ANOM"],
        verbose=verbose,
    )
    state_out["imputer_state"] = imputer_st

    # =========================================================================
    # TỔNG KẾT & KIỂM TRA CHỐT CHẶN (VALIDATION SUMMARY & SHAPE COMPARISON)
    # =========================================================================
    n_rows_final, n_cols_final = df.shape
    mem_final_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
    total_nulls_final = int(df.isnull().sum().sum())
    income_skew_final = (
        float(df["AMT_INCOME_TOTAL"].skew()) if "AMT_INCOME_TOTAL" in df.columns else np.nan
    )
    days_anom_final = (
        int((df["DAYS_EMPLOYED"] == 365243).sum())
        if "DAYS_EMPLOYED" in df.columns
        else 0
    )
    high_null_cols_rem = int((df.isnull().mean() > high_null_threshold).sum())
    elapsed = time.time() - start_time

    if verbose:
        print("\n" + "=" * 85)
        print("TỔNG KẾT NGHIỆM THU PIPELINE (DATA PREPROCESSING VALIDATION SUMMARY)")
        print("=" * 85)

        # 1. Bảng so sánh Shape
        print("\n▶ 1. BẢNG TÓM TẮT THAY ĐỔI SHAPE & BỘ NHỚ TRƯỚC VÀ SAU KHI XỬ LÝ:")
        print("-" * 85)
        shape_table = pd.DataFrame(
            [
                {
                    "Chỉ số đánh giá": "Số dòng (Rows / Grain)",
                    "Trước xử lý": f"{n_rows_init:,d}",
                    "Sau xử lý": f"{n_rows_final:,d}",
                    "Chênh lệch": f"{n_rows_final - n_rows_init:+d} (Bảo toàn)",
                },
                {
                    "Chỉ số đánh giá": "Số cột đặc trưng (Features)",
                    "Trước xử lý": f"{n_cols_init:,d}",
                    "Sau xử lý": f"{n_cols_final:,d}",
                    "Chênh lệch": f"{n_cols_final - n_cols_init:+d} cột",
                },
                {
                    "Chỉ số đánh giá": "  ├ Cột bị loại bỏ (High null & Zero var)",
                    "Trước xử lý": "0",
                    "Sau xử lý": f"{len(dropped_state.get('all', [])):,d}",
                    "Chênh lệch": f"-{len(dropped_state.get('all', []))} cột",
                },
                {
                    "Chỉ số đánh giá": "  └ Cột mới tạo (DAYS_EMPLOYED_ANOM, NO_CC)",
                    "Trước xử lý": "0",
                    "Sau xử lý": "2",
                    "Chênh lệch": "+2 cột",
                },
                {
                    "Chỉ số đánh giá": "Tổng ô khuyết thiếu (Missing Cells)",
                    "Trước xử lý": f"{total_nulls_init:,d}",
                    "Sau xử lý": f"{total_nulls_final:,d}",
                    "Chênh lệch": f"{total_nulls_final - total_nulls_init:,d} (Sạch 100%)",
                },
                {
                    "Chỉ số đánh giá": "Dung lượng bộ nhớ RAM",
                    "Trước xử lý": f"{mem_init_mb:,.2f} MB",
                    "Sau xử lý": f"{mem_final_mb:,.2f} MB",
                    "Chênh lệch": f"{mem_final_mb - mem_init_mb:+,.2f} MB",
                },
            ]
        )
        print(shape_table.to_string(index=False))

        # 2. Bảng kiểm định 5 chốt chặn cốt lõi
        print("\n▶ 2. KIỂM ĐỊNH 5 CHỐT CHẶN CHẤT LƯỢNG (ASSERTION & VALIDATION CHECKS):")
        print("-" * 85)
        check1 = high_null_cols_rem == 0
        check2 = days_anom_final == 0
        check3 = abs(income_skew_final) < 3.0 if pd.notnull(income_skew_final) else True
        check4 = total_nulls_final == 0
        check5 = n_rows_final == n_rows_init

        status_fmt = lambda ok: "✓ ĐẠT CHUẨN" if ok else "✗ KHÔNG ĐẠT"

        print(f"[{status_fmt(check1)}] Chốt chặn 1: Không còn cột nào có tỷ lệ missing > {high_null_threshold*100:.0f}% (Số cột còn lại: {high_null_cols_rem})")
        print(f"[{status_fmt(check2)}] Chốt chặn 2: Không còn giá trị 365243 trong DAYS_EMPLOYED (Số giá trị: {days_anom_final})")
        print(f"[{status_fmt(check3)}] Chốt chặn 3: Triệt tiêu lệch phải cực đoan AMT_INCOME_TOTAL (Skewness: {income_skew_init:.2f} -> {income_skew_final:.2f})")
        print(f"[{status_fmt(check4)}] Chốt chặn 4: Toàn bộ dữ liệu sạch 100% khuyết thiếu (Tổng NaN còn lại: {total_nulls_final})")
        print(f"[{status_fmt(check5)}] Chốt chặn 5: Bảo toàn grain 1 khách hàng = 1 dòng (Zero fan-out / Row explosion)")

        print("-" * 85)
        all_passed = all([check1, check2, check3, check4, check5])
        print(f"KẾT QUẢ CUỐI CÙNG: {'✓ TẤT CẢ TIÊU CHUẨN ĐÃ ĐƯỢC NGHIỆM THU THÀNH CÔNG!' if all_passed else '✗ CÓ LỖI XẢY RA TRONG PIPELINE!'}")
        print(f"Thời gian thực thi toàn bộ pipeline: {elapsed:.2f} giây")
        print("=" * 85 + "\n")

    return df, state_out
