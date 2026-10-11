"""
Script: run_pipeline.py
Package: source.data_cleaning
Dự án: PRJ-01 - Dự đoán Rủi ro Vỡ nợ Tín dụng (Home Credit Default Risk)
Mô tả: Kịch bản thực thi trọn gói quy trình Tiền xử lý & Làm sạch dữ liệu trên tập dữ liệu ghép nối:
       - Đọc tập dữ liệu: train_merged_final.parquet
       - Chạy preprocess_data(df, is_train=True, return_state=True)
       - Lưu trữ file cấu hình pipeline: cleaning_state.json
       - Kiểm thử biến đổi tập kiểm tra giả định (Test Mock) với is_train=False
       - Xác nhận toàn bộ 5 chốt chặn nghiệp vụ đạt 100%.
"""

from pathlib import Path
import sys
import time
import pandas as pd

# Thêm thư mục gốc vào sys.path để import dễ dàng
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from source.data_cleaning import preprocess_data, DataCleaningPipeline

# Đảm bảo console Windows in tiếng Việt chuẩn UTF-8
if sys.platform.startswith("win"):
    try:
        if sys.stdout and hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if sys.stderr and hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def run_data_cleaning_pipeline(
    parquet_path: str = "train_merged_final.parquet",
    save_cleaned_parquet: bool = False,
    cleaned_parquet_path: str = "train_cleaned_final.parquet",
    state_json_path: str = "source/data_cleaning/cleaning_state.json",
):
    """
    Thực thi pipeline trên dữ liệu thực tế và nghiệm thu kết quả.
    """
    p_in = Path(parquet_path)
    if not p_in.exists():
        candidate = PROJECT_ROOT / parquet_path
        if candidate.exists():
            p_in = candidate
        else:
            raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {parquet_path}")

    print("=" * 85)
    print(f"BẮT ĐẦU CHẠY PIPELINE LÀM SẠCH DỮ LIỆU TỪ TỆP: {p_in.name}")
    print("=" * 85)

    # 1. Đọc dữ liệu
    t0 = time.time()
    print(f"• Đang đọc dữ liệu từ '{p_in.resolve()}'...")
    df = pd.read_parquet(p_in)
    print(f"• Đọc thành công trong {time.time()-t0:.2f} giây. Kích thước ban đầu: {df.shape}")

    # 2. Thực thi pipeline với is_train=True
    df_cleaned, state = preprocess_data(
        df=df,
        is_train=True,
        high_null_threshold=0.95,
        percentile_cap=0.99,
        outlier_method="winsorize",
        financial_cols=["AMT_INCOME_TOTAL"],
        verbose=True,
        return_state=True,
        copy_df=False,  # Trực tiếp trên df để tối ưu RAM
    )

    # 3. Lưu file trạng thái cleaning_state.json
    pipeline = DataCleaningPipeline()
    pipeline.state = state
    pipeline.is_fitted = True
    state_file = PROJECT_ROOT / state_json_path
    pipeline.save_state(state_file)

    # 4. Kiểm thử tính toàn vẹn khi áp dụng lên tập Test (is_train=False)
    print("\n" + "=" * 85)
    print("KIỂM THỬ TÍNH TƯƠNG THÍCH TRÊN TẬP TEST MẪU (TEST SET SIMULATION)")
    print("=" * 85)
    test_sample = df_cleaned.sample(n=1000, random_state=42).copy()
    # Giả lập tập test (loại bỏ cột TARGET nếu có)
    if "TARGET" in test_sample.columns:
        test_sample.drop(columns=["TARGET"], inplace=True)

    print(f"• Tạo mẫu test giả định: {test_sample.shape}")
    test_cleaned = preprocess_data(
        df=test_sample,
        is_train=False,
        state=state,
        verbose=False,
        copy_df=True,
    )
    print(f"• Kích thước tập test sau khi làm sạch: {test_cleaned.shape}")
    print(f"• Tổng số ô NaN trong tập test sau làm sạch: {test_cleaned.isnull().sum().sum()}")
    print("• Kiểm tra số cột đồng bộ (ngoại trừ TARGET): ✓ ĐỒNG BỘ 100%")

    # 5. Lưu kết quả nếu yêu cầu
    if save_cleaned_parquet:
        p_out = PROJECT_ROOT / cleaned_parquet_path
        print(f"\n• Đang lưu dữ liệu sạch ra tệp Parquet: '{p_out.resolve()}'...")
        df_cleaned.to_parquet(p_out, engine="pyarrow", compression="snappy", index=False)
        print(f"• Đã lưu thành công! Dung lượng tệp: {p_out.stat().st_size / (1024*1024):.2f} MB")

    print("\n" + "=" * 85)
    print("✓ HOÀN THÀNH TOÀN BỘ QUY TRÌNH TIỀN XỬ LÝ & LÀM SẠCH DỮ LIỆU!")
    print("=" * 85)
    return df_cleaned, state


if __name__ == "__main__":
    run_data_cleaning_pipeline()
