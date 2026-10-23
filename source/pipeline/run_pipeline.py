"""
Script: run_pipeline.py
Package: source.pipeline
Dự án: PRJ-01 - Dự đoán Rủi ro Vỡ nợ Tín dụng (Home Credit Default Risk)
Mô tả: Trình điều phối liên hoàn toàn bộ quy trình xử lý dữ liệu (1-Click Pipeline Runner).
       Tích hợp cơ chế Smart Cache Skip:
       - Tự động kiểm tra file đầu ra của từng bước. Nếu đã tồn tại thì bỏ qua (Skip) để tiết kiệm thời gian.
       - Hỗ trợ tham số --force để ghi đè và ép chạy lại toàn bộ từ đầu.
       - Hỗ trợ tham số --step [1|2|3|all] để chạy riêng lẻ từng bước mong muốn.

Cách sử dụng:
    # 1. Chạy thông minh (bỏ qua các bước đã hoàn thành):
    python source/pipeline/run_pipeline.py

    # 2. Ép chạy lại toàn bộ từ đầu (ghi đè kết quả cũ):
    python source/pipeline/run_pipeline.py --force

    # 3. Chạy riêng một bước cụ thể:
    python source/pipeline/run_pipeline.py --step 3
"""
 
import argparse
import sys
import time
from pathlib import Path

# Đảm bảo thư mục gốc dự án nằm trong PYTHONPATH
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from source.pipeline.table_sanitizer import run_table_sanitizer
from source.pipeline.merge_pipeline import run_merge_pipeline
from source.pipeline.tree_preprocessing import run_tree_preprocessing


CLEANED_TABLE_FILES = [
    "application_train.parquet",
    "bureau.parquet",
    "previous_application.parquet",
    "POS_CASH_balance.parquet",
    "installments_payments.parquet",
    "credit_card_balance.parquet",
]


def check_step_1_completed(data_dir: Path) -> bool:
    """Kiểm tra xem toàn bộ 6 bảng Parquet sạch của Bước 1 đã tồn tại hay chưa."""
    for f in CLEANED_TABLE_FILES:
        if not (data_dir / f).exists():
            return False
    return True


def check_step_2_completed(data_dir: Path) -> bool:
    """Kiểm tra xem ma trận ghép nối train_merged.parquet của Bước 2 đã tồn tại hay chưa."""
    return (data_dir / "train_merged.parquet").exists()


def check_step_3_completed(data_dir: Path) -> bool:
    """Kiểm tra xem các tệp huấn luyện train_lgb/val_lgb của Bước 3 đã tồn tại hay chưa."""
    f1 = data_dir / "train_lgb.parquet"
    f2 = data_dir / "val_lgb.parquet"
    f3 = data_dir / "pipeline_metadata.json"
    return f1.exists() and f2.exists() and f3.exists()


def main():
    parser = argparse.ArgumentParser(
        description="Quy trình Xử lý Dữ liệu Tuyến tính 3 Bước (Home Credit Default Risk)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Bắt buộc chạy lại toàn bộ các bước, không dùng bộ nhớ cache.",
    )
    parser.add_argument(
        "--step",
        choices=["1", "2", "3", "all"],
        default="all",
        help="Chỉ định bước cần chạy (1: Sanitizer, 2: Merge, 3: Tree Preprocessing, all: Toàn bộ quy trình).",
    )
    args = parser.parse_args()

    processed_dir = project_root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 85)
    print("🚀 KHỞI ĐỘNG PIPELINE DỮ LIỆU TUYẾN TÍNH - HOME CREDIT DEFAULT RISK")
    print("=" * 85)
    print(f"• Thư mục đích           : {processed_dir}")
    print(f"• Chế độ cưỡng chế (Force): {'BẬT (Ghi đè tất cả)' if args.force else 'TẮT (Sử dụng Smart Cache Skip)'}")
    print(f"• Phạm vi thực thi       : Bước {args.step}")
    print("=" * 85)

    pipeline_start = time.time()
    steps_run = []

    # -------------------------------------------------------------------------
    # BƯỚC 1: LÀM SẠCH CẤP BẢNG ĐỘC LẬP
    # -------------------------------------------------------------------------
    if args.step in ["1", "all"]:
        print("\n" + "#" * 85)
        print(">>> [BƯỚC 1/3] LÀM SẠCH CẤP BẢNG ĐỘC LẬP (TABLE SANITIZER)")
        print("#" * 85)
        if not args.force and check_step_1_completed(processed_dir):
            print("⚡ [SMART CACHE SKIP] Toàn bộ 6 bảng parquet sạch đã tồn tại.")
            print("   -> Bỏ qua Bước 1. Dùng cờ --force nếu muốn chạy lại từ file CSV thô.")
        else:
            t0 = time.time()
            run_table_sanitizer(output_dir=processed_dir)
            steps_run.append(f"Bước 1 ({time.time() - t0:.1f}s)")

    # -------------------------------------------------------------------------
    # BƯỚC 2: NÉN 1:N VÀ GHÉP NỐI AN TOÀN
    # -------------------------------------------------------------------------
    if args.step in ["2", "all"]:
        print("\n" + "#" * 85)
        print(">>> [BƯỚC 2/3] NÉN 1:N VÀ GHÉP NỐI ĐA BẢNG AN TOÀN (MERGE PIPELINE)")
        print("#" * 85)
        if not args.force and check_step_2_completed(processed_dir):
            print("⚡ [SMART CACHE SKIP] Tệp 'train_merged.parquet' đã tồn tại.")
            print("   -> Bỏ qua Bước 2. Dùng cờ --force nếu muốn nén và ghép nối lại.")
        else:
            t0 = time.time()
            run_merge_pipeline(input_dir=processed_dir, output_file=processed_dir / "train_merged.parquet")
            steps_run.append(f"Bước 2 ({time.time() - t0:.1f}s)")

    # -------------------------------------------------------------------------
    # BƯỚC 3: TIỀN XỬ LÝ CHUYÊN BIỆT CHO LIGHTGBM
    # -------------------------------------------------------------------------
    if args.step in ["3", "all"]:
        print("\n" + "#" * 85)
        print(">>> [BƯỚC 3/3] TIỀN XỬ LÝ CHUYÊN BIỆT CHO LIGHTGBM (TREE PREPROCESSING)")
        print("#" * 85)
        if not args.force and check_step_3_completed(processed_dir):
            print("⚡ [SMART CACHE SKIP] Tệp 'train_lgb.parquet' và 'val_lgb.parquet' đã tồn tại.")
            print("   -> Bỏ qua Bước 3. Dùng cờ --force nếu muốn tiền xử lý lại.")
        else:
            t0 = time.time()
            run_tree_preprocessing(
                merged_parquet=processed_dir / "train_merged.parquet",
                output_dir=processed_dir
            )
            steps_run.append(f"Bước 3 ({time.time() - t0:.1f}s)")

    total_time = time.time() - pipeline_start
    print("\n" + "=" * 85)
    print("✨ TOÀN BỘ QUY TRÌNH PIPELINE HOÀN TẤT THÀNH CÔNG!")
    print("=" * 85)
    print(f"• Các bước đã thực thi: {', '.join(steps_run) if steps_run else 'Tất cả các bước đều đã được Cache'}")
    print(f"• Tổng thời gian      : {total_time:.1f} giây ({total_time / 60:.2f} phút)")
    print(f"• Dữ liệu LightGBM sẵn sàng tại: {processed_dir}/train_lgb.parquet & val_lgb.parquet")
    print("=" * 85)


if __name__ == "__main__":
    main()
