"""
Module: build_train_merged.py
Dự án: PRJ-01 - Dự đoán Rủi ro Vỡ nợ Tín dụng (Home Credit Default Risk)
Mô tả: Kịch bản thực thi trọn gói (End-to-End Execution):
       1. Đọc application_train.csv
       2. Tuần tự aggregate, làm phẳng và merge 5 bảng phụ:
          - bureau.csv (BUREAU_)
          - previous_application.csv (PREV_)
          - POS_CASH_balance.csv (POS_)
          - installments_payments.csv (INSTAL_)
          - credit_card_balance.csv (CC_)
       3. Chốt chặn bảo toàn 307,511 dòng và không sinh cột _x, _y
       4. Xuất file kết quả chuẩn: train_merged_final.parquet
"""

import os
import gc
import sys
import time
from pathlib import Path
from typing import Optional, Union
import pandas as pd

# Đảm bảo console Windows in tiếng Việt chuẩn Unicode
if sys.platform.startswith("win"):
    try:
        if sys.stdout and hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if sys.stderr and hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Import các module cốt lõi đã xây dựng
try:
    from .aggregate_secondary_table import aggregate_secondary_table
    from .flatten_aggregated_df import flatten_aggregated_df
except ImportError:
    from aggregate_secondary_table import aggregate_secondary_table
    from flatten_aggregated_df import flatten_aggregated_df


def run_build_train_merged(
    data_dir: Optional[Union[str, Path]] = None,
    output_parquet: Optional[Union[str, Path]] = None,
    expected_rows: int = 307511
) -> Path:
    pipeline_start = time.time()

    # 1. Xác định thư mục dữ liệu
    if data_dir is None:
        candidate_paths = [
            Path("Dataset - Home Credit Default Risk"),
            Path("../Dataset - Home Credit Default Risk"),
            Path("PRJ-01/Dataset - Home Credit Default Risk"),
            Path(__file__).resolve().parent.parent / "Dataset - Home Credit Default Risk" if "__file__" in globals() else None
        ]
        for p in candidate_paths:
            if p and p.exists() and p.is_dir():
                data_dir = p.resolve()
                break
        if data_dir is None:
            raise FileNotFoundError("Không tìm thấy thư mục 'Dataset - Home Credit Default Risk'!")
    else:
        data_dir = Path(data_dir).resolve()

    if output_parquet is None:
        output_parquet = data_dir.parent / "train_merged_final.parquet"
    else:
        output_parquet = Path(output_parquet).resolve()

    print("=" * 85)
    print("KHỞI CHẠY PIPELINE TẠO TẬP DỮ LIỆU HUẤN LUYỆN HOÀN CHỈNH (CÁCH 1)")
    print("=" * 85)
    print(f"• Thư mục nguồn : {data_dir}")
    print(f"• File đích     : {output_parquet}")
    print(f"• Số dòng chuẩn : {expected_rows:,d} dòng")
    print("-" * 85)

    # 2. Đọc bảng chính application_train.csv
    train_path = data_dir / "application_train.csv"
    if not train_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {train_path}")

    t0 = time.time()
    print(f"⏳ [0/5] Đang đọc bảng chính: {train_path.name}...")
    df_train = pd.read_csv(train_path)
    
    # Downcast bảng chính ban đầu để tiết kiệm RAM
    for col in df_train.select_dtypes(include=["float64"]).columns:
        df_train[col] = df_train[col].astype("float32")
    for col in df_train.select_dtypes(include=["int64"]).columns:
        if col not in ["SK_ID_CURR", "TARGET"]:
            df_train[col] = df_train[col].astype("int32")

    mem_train_mb = df_train.memory_usage(deep=True).sum() / (1024 * 1024)
    print(f"  ✓ Đã nạp application_train trong {time.time() - t0:.2f}s | Shape: {df_train.shape} | RAM: {mem_train_mb:.2f} MB")
    print("-" * 85)

    assert len(df_train) == expected_rows, f"Lỗi số dòng application_train ban đầu: {len(df_train)} != {expected_rows}"

    # 3. Cấu hình 5 bảng phụ tuần tự
    configs = [
        {
            "name": "bureau",
            "file": "bureau.csv",
            "prefix": "BUREAU",
            "exclude": ["SK_ID_BUREAU"]
        },
        {
            "name": "previous_application",
            "file": "previous_application.csv",
            "prefix": "PREV",
            "exclude": ["SK_ID_PREV"]
        },
        {
            "name": "POS_CASH_balance",
            "file": "POS_CASH_balance.csv",
            "prefix": "POS",
            "exclude": ["SK_ID_PREV"]
        },
        {
            "name": "installments_payments",
            "file": "installments_payments.csv",
            "prefix": "INSTAL",
            "exclude": ["SK_ID_PREV"]
        },
        {
            "name": "credit_card_balance",
            "file": "credit_card_balance.csv",
            "prefix": "CC",
            "exclude": ["SK_ID_PREV"]
        }
    ]

    # 4. Vòng lặp xử lý tuần tự từng bảng phụ
    for step, cfg in enumerate(configs, 1):
        step_start = time.time()
        file_path = data_dir / cfg["file"]
        if not file_path.exists():
            print(f"⚠️ Bỏ qua {cfg['file']} (không tìm thấy file)")
            continue

        print(f"⏳ [{step}/5] Đang xử lý bảng: {cfg['file']} ...")
        # Đọc dữ liệu bảng phụ
        t_read = time.time()
        df_sec_raw = pd.read_csv(file_path)
        print(f"  • Đọc CSV ({len(df_sec_raw):,d} dòng) mất {time.time() - t_read:.2f}s")

        # Bước A: Aggregate về cấp độ SK_ID_CURR
        t_agg = time.time()
        df_agg = aggregate_secondary_table(
            df=df_sec_raw,
            group_key="SK_ID_CURR",
            exclude_keys=cfg["exclude"],
            downcast=True,
            verbose=False
        )
        # Giải phóng dữ liệu thô ngay lập tức
        del df_sec_raw
        gc.collect()
        print(f"  • Aggregate về {len(df_agg):,d} khách hàng mất {time.time() - t_agg:.2f}s")

        # Bước B: Làm phẳng MultiIndex & Phục hồi khóa chính
        df_flat = flatten_aggregated_df(
            df_agg=df_agg,
            prefix=cfg["prefix"],
            group_key="SK_ID_CURR",
            verbose=False
        )
        del df_agg
        gc.collect()

        # Bước C: Merge vào df_train & Downcast cột mới
        cols_before = set(df_train.columns)
        df_train = pd.merge(df_train, df_flat, on="SK_ID_CURR", how="left")

        # Chốt chặn toàn vẹn (Assertion)
        assert len(df_train) == expected_rows, (
            f"LỖI BÙNG NỔ DÒNG (FAN-OUT) TẠI {cfg['name']}! "
            f"Số dòng hiện tại là {len(df_train):,d} != {expected_rows:,d}."
        )

        collision_cols = [c for c in df_train.columns if c.endswith("_x") or c.endswith("_y")]
        assert len(collision_cols) == 0, (
            f"LỖI XUNG ĐỘT CỘT TẠI {cfg['name']}! Phát hiện đuôi _x/_y: {collision_cols[:5]}"
        )

        # Downcast các cột mới
        new_cols = [c for c in df_train.columns if c not in cols_before]
        for col in new_cols:
            dtype_str = str(df_train[col].dtype)
            if dtype_str == "float64":
                df_train[col] = df_train[col].astype("float32")
            elif dtype_str == "int64":
                df_train[col] = df_train[col].astype("int32")

        # Giải phóng df_flat
        del df_flat
        gc.collect()

        current_ram = df_train.memory_usage(deep=True).sum() / (1024 * 1024)
        print(
            f"  ✓ Ghép thành công! +{len(new_cols)} đặc trưng | "
            f"Shape: {df_train.shape} | RAM: {current_ram:.2f} MB | "
            f"Tổng thời gian bước: {time.time() - step_start:.2f}s"
        )
        print("-" * 85)

    # 5. Xuất file Parquet hoàn chỉnh
    print(f"⏳ Đang nén và xuất file Parquet: {output_parquet.name} ...")
    t_save = time.time()
    df_train.to_parquet(output_parquet, index=False, engine="pyarrow")
    file_size_mb = output_parquet.stat().st_size / (1024 * 1024)
    print(f"  ✓ Xuất file thành công trong {time.time() - t_save:.2f}s!")

    # 6. Nghiệm thu đọc kiểm chứng
    print("⏳ Kiểm chứng đọc lại 5 dòng từ file Parquet vừa tạo...")
    t_verify = time.time()
    df_check = pd.read_parquet(output_parquet)
    print(f"  ✓ Đọc kiểm chứng thành công trong {time.time() - t_verify:.2f}s!")
    print(f"  ✓ Kích thước xác thực: {df_check.shape} (Dòng: {len(df_check):,d}, Cột: {df_check.shape[1]:,d})")
    del df_check
    gc.collect()

    total_pipeline_time = time.time() - pipeline_start
    print("=" * 85)
    print("✓ HOÀN TẤT THÀNH CÔNG TOÀN BỘ QUY TRÌNH!")
    print("=" * 85)
    print(f"• Đường dẫn file Parquet : {output_parquet}")
    print(f"• Dung lượng file ổ đĩa  : {file_size_mb:.2f} MB (Siêu nén, tiết kiệm >80% so với CSV)")
    print(f"• Số dòng dữ liệu        : {df_train.shape[0]:,d} (Bảo toàn 100% tập huấn luyện)")
    print(f"• Tổng số cột đặc trưng  : {df_train.shape[1]:,d} cột")
    print(f"• Tổng thời gian xử lý   : {total_pipeline_time:.2f}s (~{total_pipeline_time / 60:.1f} phút)")
    print("=" * 85)

    return output_parquet


if __name__ == "__main__":
    run_build_train_merged()
