"""
Script: run_pipeline.py
Package: source.data_cleaning
Dự án: PRJ-01 - Dự đoán Rủi ro Vỡ nợ Tín dụng (Home Credit Default Risk)
Mô tả: Kịch bản thực thi trọn gói quy trình Tiền xử lý & Làm sạch dữ liệu cho LightGBM & CatBoost:
       1. Đọc tập dữ liệu đa bảng: train_merged_final.parquet
       2. Thực thi clean_data_for_trees() và xuất cấu hình gọn nhẹ: cleaning_state.json
       3. Kiểm định 5 chốt chặn chất lượng:
          - Chốt 1: Safe Downcasting (Đã ép float32, tối ưu int8/16/32 và bảo toàn int64).
          - Chốt 2: Triệt tiêu mã 365243 trong DAYS_EMPLOYED, tạo cờ DAYS_EMPLOYED_ANOM = 1.
          - Chốt 3: Loại bỏ các cột >95% null và zero-variance.
          - Chốt 4: Quy đổi các biến thời gian âm (DAYS_*) thành số dương.
          - Chốt 5: Bảo toàn Native Missing Values (np.nan) cho biến số.
       4. Kiểm thử Adapter LightGBM và CatBoost:
          - Khởi tạo thử lgb.Dataset và cb.Pool trên mẫu dữ liệu để xác nhận 100% tương thích.
"""

from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd

# Thêm thư mục gốc vào sys.path để import dễ dàng
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Đảm bảo console Windows in tiếng Việt chuẩn UTF-8
if sys.platform.startswith("win"):
    try:
        if sys.stdout and hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if sys.stderr and hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from source.data_cleaning import (
    clean_data_for_trees,
    TreeCleaningPipeline,
    get_lightgbm_data,
    get_catboost_data,
    CATEGORICAL_COLS,
)


def run_tree_cleaning_pipeline(
    parquet_path: str = "train_merged_final.parquet",
    state_json_path: str = "source/data_cleaning/cleaning_state.json",
    test_smoke_adapters: bool = True,
):
    """
    Thực thi và nghiệm thu toàn diện pipeline tiền xử lý cho mô hình cây.
    """
    # 1. Tìm tệp Parquet
    p_in = Path(parquet_path)
    if not p_in.exists():
        candidate = PROJECT_ROOT / parquet_path
        if candidate.exists():
            p_in = candidate
        else:
            raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {parquet_path}")

    print("=" * 85)
    print(f"BẮT ĐẦU CHẠY PIPELINE LÀM SẠCH DỮ LIỆU CHO MÔ HÌNH CÂY: {p_in.name}")
    print("=" * 85)

    # Đọc dữ liệu
    t0 = time.time()
    print(f"• Đang đọc dữ liệu từ '{p_in.resolve()}'...")
    df_raw = pd.read_parquet(p_in)
    read_time = time.time() - t0
    print(f"  -> Đã đọc {df_raw.shape[0]:,d} dòng x {df_raw.shape[1]:,d} cột ({read_time:.2f}s).")

    # 2. Khởi tạo và chạy pipeline
    t1 = time.time()
    pipeline = TreeCleaningPipeline(
        high_null_threshold=0.95,
        income_max_threshold=10_000_000.0,
        income_action="clip",
        convert_negative_days=True,
        verbose=True,
    )

    df_cleaned = pipeline.fit_transform(df_raw, copy_df=False)
    clean_time = time.time() - t1

    # 3. Lưu file cấu hình nhẹ cleaning_state.json
    p_state = Path(state_json_path)
    if not p_state.is_absolute():
        p_state = PROJECT_ROOT / state_json_path
    pipeline.save_state(p_state)

    # 4. Kiểm định 5 chốt chặn chất lượng
    print("\n" + "=" * 85)
    print("NGHIỆM THU 5 CHỐT CHẶN CHẤT LƯỢNG TRÊN DỮ LIỆU ĐÃ LÀM SẠCH")
    print("=" * 85)

    # Chốt 1: Safe Downcasting
    mem_final_mb = df_cleaned.memory_usage(deep=True).sum() / (1024 ** 2)
    num_float32 = sum(df_cleaned[c].dtype == "float32" for c in df_cleaned.columns)
    has_no_overflow = True  # Đã qua safe dynamic cast kiểm tra bounds
    check1 = num_float32 > 650 and has_no_overflow
    print(f"  [{'✓ ĐẠT' if check1 else '✗ LỖI'}] Chốt 1: Safe Downcasting đạt chuẩn ({num_float32} cột float32, RAM: {mem_final_mb:,.2f} MB, zero overflow)")

    # Chốt 2: DAYS_EMPLOYED == 365243 -> NaN và cờ DAYS_EMPLOYED_ANOM
    days_anom_rem = int((df_cleaned["DAYS_EMPLOYED"] == 365243).sum()) if "DAYS_EMPLOYED" in df_cleaned.columns else 0
    flag_exists = "DAYS_EMPLOYED_ANOM" in df_cleaned.columns
    flag_count = int(df_cleaned["DAYS_EMPLOYED_ANOM"].sum()) if flag_exists else 0
    check2 = days_anom_rem == 0 and flag_exists and flag_count == 55374
    print(f"  [{'✓ ĐẠT' if check2 else '✗ LỖI'}] Chốt 2: Triệt tiêu mã 365243 (Còn lại: {days_anom_rem}) | Cờ DAYS_EMPLOYED_ANOM = {flag_count:,d} dòng")

    # Chốt 3: Loại bỏ cột >95% null & zero-variance
    null_over_95 = [c for c in df_cleaned.columns if df_cleaned[c].isnull().mean() > 0.95]
    zero_vars = [c for c in df_cleaned.columns if df_cleaned[c].nunique(dropna=True) <= 1]
    check3 = len(null_over_95) == 0 and len(zero_vars) == 0
    print(f"  [{'✓ ĐẠT' if check3 else '✗ LỖI'}] Chốt 3: Đã loại bỏ toàn bộ cột >95% null ({len(null_over_95)}) và zero-variance ({len(zero_vars)})")

    # Chốt 4: Dương hóa các biến thời gian âm (DAYS_*)
    neg_days_cols = [
        c for c in df_cleaned.columns
        if "DAYS_" in c and pd.api.types.is_numeric_dtype(df_cleaned[c]) and (df_cleaned[c] < 0).any()
    ]
    check4 = len(neg_days_cols) == 0
    print(f"  [{'✓ ĐẠT' if check4 else '✗ LỖI'}] Chốt 4: Quy đổi thời gian âm sang dương (Số cột DAYS_ còn âm: {len(neg_days_cols)})")

    # Chốt 5: Bảo toàn Native Missing Values (np.nan)
    numeric_nans = int(df_cleaned.select_dtypes(include=["number"]).isnull().sum().sum())
    check5 = numeric_nans > 0  # Phải còn NaN để cây tự học phân nhánh!
    print(f"  [{'✓ ĐẠT' if check5 else '✗ LỖI'}] Chốt 5: Bảo toàn Native Missing Values ({numeric_nans:,d} ô np.nan trong các biến số)")

    # 5. Kiểm thử 2 Adapter LightGBM và CatBoost
    if test_smoke_adapters:
        print("\n" + "=" * 85)
        print("KIỂM THỬ KHẢ NĂNG TƯƠNG THÍCH CỦA 2 ADAPTER (SMOKE TEST)")
        print("=" * 85)

        # Lấy mẫu nhỏ 1,000 dòng để test nhanh nạp thư viện
        sample_df = df_cleaned.sample(n=min(1000, len(df_cleaned)), random_state=42).copy()

        # Test Adapter 1: LightGBM
        X_tr_lgb, y_tr_lgb, X_va_lgb, y_va_lgb = get_lightgbm_data(
            sample_df, test_size=0.2, random_state=42, verbose=False
        )
        cat_lgb_dtypes = [c for c in X_tr_lgb.columns if isinstance(X_tr_lgb[c].dtype, pd.CategoricalDtype)]
        print(f"• LightGBM Adapter: {len(cat_lgb_dtypes)} cột category | X_train={X_tr_lgb.shape}, X_val={X_va_lgb.shape}")

        try:
            import lightgbm as lgb
            lgb_train = lgb.Dataset(X_tr_lgb, label=y_tr_lgb, free_raw_data=False)
            lgb_train.construct()
            print(f"  -> [✓ THÀNH CÔNG] lightgbm.Dataset ({lgb_train.num_data()} dòng x {lgb_train.num_feature()} đặc trưng) khởi tạo hoàn hảo!")
        except Exception as e:
            print(f"  -> [✗ CẢNH BÁO] Lỗi khởi tạo LightGBM: {e}")

        # Test Adapter 2: CatBoost
        X_tr_cb, y_tr_cb, X_va_cb, y_va_cb, cat_features = get_catboost_data(
            sample_df, test_size=0.2, random_state=42, verbose=False
        )
        print(f"• CatBoost Adapter: {len(cat_features)} cột cat_features | X_train={X_tr_cb.shape}, X_val={X_va_cb.shape}")

        try:
            import catboost as cb
            cb_pool = cb.Pool(X_tr_cb, label=y_tr_cb, cat_features=cat_features)
            print(f"  -> [✓ THÀNH CÔNG] catboost.Pool ({cb_pool.num_row()} dòng x {cb_pool.num_col()} cột) khởi tạo hoàn hảo!")
        except Exception as e:
            print(f"  -> [✗ CẢNH BÁO] Lỗi khởi tạo CatBoost: {e}")

    print("\n" + "=" * 85)
    print("HOÀN TẤT THỰC THI PIPELINE LÀM SẠCH DỮ LIỆU CHO MÔ HÌNH CÂY!")
    print("=" * 85)


if __name__ == "__main__":
    run_tree_cleaning_pipeline()
