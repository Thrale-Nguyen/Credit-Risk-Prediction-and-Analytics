"""
Module: build_train_merged.py
Package: source.data_merging
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

# Import các module cốt lõi trong cùng package
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
            Path(__file__).resolve().parent.parent.parent / "Dataset - Home Credit Default Risk" if "__file__" in globals() else None,
            Path(__file__).resolve().parent.parent / "Dataset - Home Credit Default Risk" if "__file__" in globals() else None,
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
        if col != "SK_ID_CURR":
            df_train[col] = df_train[col].astype("int32")

    mem_init_mb = df_train.memory_usage(deep=True).sum() / (1024 * 1024)
    print(f"✓ Đọc xong bảng chính trong {time.time()-t0:.2f}s | Shape: {df_train.shape} | RAM: {mem_init_mb:.2f} MB")
    assert len(df_train) == expected_rows, f"Bảng chính không đúng {expected_rows:,d} dòng!"

    # 3. Cấu hình 5 bảng phụ cần ghép
    configs = [
        {
            "name": "bureau",
            "filename": "bureau.csv",
            "prefix": "BUREAU",
            "exclude": ["SK_ID_BUREAU"],
            "num_agg": ["min", "max", "mean", "sum"],
            "cat_agg": ["mean", "sum"]
        },
        {
            "name": "previous_application",
            "filename": "previous_application.csv",
            "prefix": "PREV",
            "exclude": ["SK_ID_PREV"],
            "num_agg": ["min", "max", "mean", "sum"],
            "cat_agg": ["mean", "sum"]
        },
        {
            "name": "POS_CASH_balance",
            "filename": "POS_CASH_balance.csv",
            "prefix": "POS",
            "exclude": ["SK_ID_PREV"],
            "num_agg": ["min", "max", "mean", "sum"],
            "cat_agg": ["mean", "sum"]
        },
        {
            "name": "installments_payments",
            "filename": "installments_payments.csv",
            "prefix": "INSTAL",
            "exclude": ["SK_ID_PREV"],
            "num_agg": ["min", "max", "mean", "sum"],
            "cat_agg": ["mean", "sum"]
        },
        {
            "name": "credit_card_balance",
            "filename": "credit_card_balance.csv",
            "prefix": "CC",
            "exclude": ["SK_ID_PREV"],
            "num_agg": ["min", "max", "mean", "sum"],
            "cat_agg": ["mean", "sum"]
        }
    ]

    # 4. Vòng lặp xử lý streaming từng bảng phụ
    for idx, cfg in enumerate(configs, 1):
        step_t0 = time.time()
        file_path = data_dir / cfg["filename"]
        print(f"\n⏳ [{idx}/5] Đang xử lý: {cfg['filename']} (Prefix: {cfg['prefix']}_)...")
        if not file_path.exists():
            raise FileNotFoundError(f"Không tìm thấy file: {file_path}")

        # 4.1 Đọc bảng phụ
        t_read = time.time()
        df_sec = pd.read_csv(file_path)
        print(f"  • Nạp CSV: {len(df_sec):,d} dòng x {df_sec.shape[1]} cột ({time.time()-t_read:.2f}s)")

        # 4.2 Aggregate
        t_agg = time.time()
        df_agg = aggregate_secondary_table(
            df_sec,
            group_key="SK_ID_CURR",
            exclude_keys=cfg["exclude"],
            num_aggregations=cfg["num_agg"],
            cat_aggregations=cfg["cat_agg"],
            downcast=True,
            in_place=True,
            verbose=False
        )
        print(f"  • Aggregate: {df_agg.shape[0]:,d} unique khách hàng x {df_agg.shape[1]} metrics ({time.time()-t_agg:.2f}s)")

        # Dọn rác df_sec ngay lập tức
        del df_sec
        gc.collect()

        # 4.3 Flatten
        df_flat = flatten_aggregated_df(
            df_agg,
            prefix=cfg["prefix"],
            group_key="SK_ID_CURR",
            verbose=False
        )
        del df_agg
        gc.collect()

        # 4.4 Merge an toàn vào df_train
        t_merge = time.time()
        cols_before = set(df_train.columns)
        df_train = pd.merge(df_train, df_flat, on="SK_ID_CURR", how="left")

        # Kiểm tra chốt chặn
        assert len(df_train) == expected_rows, f"Bùng nổ dòng tại bảng {cfg['name']}! ({len(df_train):,d} dòng)"
        collision_cols = [c for c in df_train.columns if c.endswith("_x") or c.endswith("_y")]
        assert len(collision_cols) == 0, f"Xung đột cột tại {cfg['name']}: {collision_cols}"

        # Downcast cột mới
        new_cols = [c for c in df_train.columns if c not in cols_before]
        for col in new_cols:
            if df_train[col].dtype == "float64":
                df_train[col] = df_train[col].astype("float32")
            elif df_train[col].dtype == "int64":
                df_train[col] = df_train[col].astype("int32")

        del df_flat
        gc.collect()

        current_ram_mb = df_train.memory_usage(deep=True).sum() / (1024 * 1024)
        print(f"  ✓ Ghép thành công (+{len(new_cols)} cột) trong {time.time()-t_merge:.2f}s")
        print(f"  • Shape hiện tại: {df_train.shape} | RAM: {current_ram_mb:.2f} MB | Tổng bước: {time.time()-step_t0:.2f}s")

    # 5. Lưu ra file Parquet
    print("\n" + "=" * 85)
    print("⏳ Đang nén và xuất file Parquet (snappy)...")
    save_t0 = time.time()
    df_train.to_parquet(output_parquet, engine="pyarrow", compression="snappy", index=False)
    file_size_mb = output_parquet.stat().st_size / (1024 * 1024)
    print(f"✓ Đã lưu file: {output_parquet}")
    print(f"• Dung lượng file trên đĩa : {file_size_mb:.2f} MB")
    print(f"• Thời gian ghi Parquet    : {time.time()-save_t0:.2f}s")

    # 6. Kiểm tra lại tính toàn vẹn bằng cách đọc lại tệp
    print("\n⏳ Đang kiểm tra tính toàn vẹn của tệp đã lưu...")
    df_verify = pd.read_parquet(output_parquet, columns=["SK_ID_CURR"])
    assert len(df_verify) == expected_rows, "Lỗi: File parquet đọc lại có số dòng không khớp!"
    del df_verify
    gc.collect()

    total_time = time.time() - pipeline_start
    print("=" * 85)
    print("✓ PIPELINE HOÀN TẤT THÀNH CÔNG 100%!")
    print("=" * 85)
    print(f"• Kích thước cuối cùng : {df_train.shape[0]:,d} dòng x {df_train.shape[1]:,d} cột")
    print(f"• Bảo toàn Grain       : Đúng {expected_rows:,d} dòng (Zero fan-out)")
    print(f"• Dung lượng Parquet   : {file_size_mb:.2f} MB")
    print(f"• Tổng thời gian chạy  : {total_time:.2f}s ({total_time/60:.2f} phút)")
    print("=" * 85)

    return output_parquet


if __name__ == "__main__":
    run_build_train_merged()
