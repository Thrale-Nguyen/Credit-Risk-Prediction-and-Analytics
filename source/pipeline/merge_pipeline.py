"""
Module: merge_pipeline.py
Package: source.pipeline
Dự án: PRJ-01 - Dự đoán Rủi ro Vỡ nợ Tín dụng (Home Credit Default Risk)
Mô tả: [Bước 2 trong Pipeline Tuyến tính]
       Nén dữ liệu 1:N và Ghép nối An toàn (Feature Aggregation & Safe Left Join):
       1. Đọc 6 bảng sạch đã được chuẩn hóa từ data/processed/*.parquet (hoặc data/cleaned_tables/).
       2. Nén 1:N về 1:1 cho 5 bảng phụ (bureau, previous_application, POS, installments, credit_card):
          - Biến định lượng: Tính toán các thước đo thống kê tổng hợp (mean, max, min, sum, std).
          - Biến định tính: One-Hot Encoding trước khi nén để đếm tần suất và tỷ lệ hành vi (mean, sum).
          - Loại trừ các khóa phụ (SK_ID_BUREAU, SK_ID_PREV) khỏi tính toán thống kê.
       3. Làm phẳng MultiIndex và gắn tiền tố (Prefixing):
          - BUREAU_*, PREV_*, POS_*, INSTAL_*, CC_* chống xung đột tên cột ngoài khóa chính.
       4. Ghép nối an toàn (Safe Left Join):
          - Chốt chặn kiểm định bảo toàn đúng 234,072 dòng (Zero Fan-out / 1 khách hàng = đúng 1 dòng).
          - Chốt chặn ngăn chặn sinh các cột xung đột đuôi _x, _y.
       5. Xuất ma trận đặc trưng hợp nhất vào `data/processed/train_merged.parquet`.
"""

import os
import gc
import sys
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any, Union
import numpy as np
import pandas as pd

# Thiết lập encoding UTF-8 cho Windows console
if sys.platform.startswith("win"):
    try:
        if sys.stdout and hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if sys.stderr and hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


# =============================================================================
# CẤU HÌNH BẢNG & TIỀN TỐ (PREFIX CONFIGURATION)
# =============================================================================

DEFAULT_GROUP_KEY = "SK_ID_CURR"
DEFAULT_EXPECTED_ROWS: Optional[int] = None

# Danh sách 5 bảng phụ kèm tiền tố phân định tương ứng
SECONDARY_SPECS = [
    ("bureau", "BUREAU_"),
    ("previous_application", "PREV_"),
    ("POS_CASH_balance", "POS_"),
    ("installments_payments", "INSTAL_"),
    ("credit_card_balance", "CC_"),
]

# Khóa phụ cần loại trừ khỏi tính toán thống kê
EXCLUDE_KEYS = ["SK_ID_PREV", "SK_ID_BUREAU"]


def resolve_data_paths(
    input_dir: Optional[Union[str, Path]] = None,
    output_file: Optional[Union[str, Path]] = None
) -> Tuple[Path, Path]:
    """
    Xác định đường dẫn thư mục chứa bảng parquet sạch và tệp train_merged đích.
    Ưu tiên tìm trong `data/processed` trước, sau đó tới `data/cleaned_tables`.
    """
    if input_dir is None:
        candidates = [
            Path("data/processed"),
            Path("data/cleaned_tables"),
            Path("../data/processed"),
            Path("../data/cleaned_tables"),
        ]
        found_dir = None
        for c in candidates:
            if c.exists() and c.is_dir() and (c / "application_train.parquet").exists():
                found_dir = c.resolve()
                break
        if found_dir is None:
            raise FileNotFoundError(
                "Không tìm thấy thư mục chứa 'application_train.parquet' "
                "(đã quét 'data/processed' và 'data/cleaned_tables'). "
                "Vui lòng chạy Bước 1 (01_table_sanitizer.py) trước!"
            )
        in_path = found_dir
    else:
        in_path = Path(input_dir).resolve()

    if output_file is None:
        out_path = in_path.parent / "processed" / "train_merged.parquet"
    else:
        out_path = Path(output_file).resolve()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    return in_path, out_path


# =============================================================================
# HÀM NÉN 1:N VỀ 1:1 (AGGREGATION ENGINE)
# =============================================================================

def aggregate_secondary_table(
    df: pd.DataFrame,
    table_name: str,
    group_key: str = DEFAULT_GROUP_KEY,
    exclude_cols: Optional[List[str]] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Nén bảng phụ từ quan hệ 1:N về quan hệ 1:1 theo `group_key` (SK_ID_CURR):
    - Biến định lượng: Tính [mean, max, min, sum, std].
    - Biến định tính: One-Hot Encoding (float32) và tính [mean, sum].
    - Ép kiểu bộ nhớ tối ưu (float32, int32) trước khi gom nhóm để giảm tải RAM.
    """
    t0 = time.time()
    n_rows_orig = len(df)
    n_unique_keys = df[group_key].nunique()

    if exclude_cols is None:
        exclude_cols = EXCLUDE_KEYS
    else:
        exclude_cols = list(set(exclude_cols + EXCLUDE_KEYS))

    # 1. Lọc bớt cột không cần thiết & Downcast nhanh
    cols_to_use = [c for c in df.columns if c not in exclude_cols or c == group_key]
    df_work = df[cols_to_use].copy()

    for col in df_work.select_dtypes(include=["float64"]).columns:
        df_work[col] = df_work[col].astype("float32")
    for col in df_work.select_dtypes(include=["int64"]).columns:
        if col != group_key:
            df_work[col] = df_work[col].astype("int32")

    # 2. Phân loại biến định lượng và biến định tính
    cat_cols = df_work.select_dtypes(include=["object", "category"]).columns.tolist()
    if group_key in cat_cols:
        cat_cols.remove(group_key)

    num_cols = [c for c in df_work.select_dtypes(include=["number"]).columns if c != group_key]

    if verbose:
        print(f"  • Gom nhóm {n_rows_orig:,} dòng về {n_unique_keys:,} khách hàng duy nhất...")
        print(f"    - {len(num_cols)} biến định lượng -> agg=['mean', 'max', 'min', 'sum', 'std']")
        print(f"    - {len(cat_cols)} biến định tính -> One-Hot Encoding + agg=['mean', 'sum']")

    # 3. Tính toán tổng hợp cho biến định lượng
    num_agg = None
    if num_cols:
        num_agg = df_work[[group_key] + num_cols].groupby(group_key).agg(["mean", "max", "min", "sum", "std"])

    # 4. One-Hot Encoding & tính toán tổng hợp cho biến định tính
    cat_agg = None
    if cat_cols:
        # Chuẩn hóa chữ hoa và cắt khoảng trắng để gom các biến cùng nghĩa (tránh sinh trùng cột do khác hoa/thường)
        for c in cat_cols:
            df_work[c] = df_work[c].astype(str).str.strip().str.upper()
            df_work[c] = df_work[c].replace({"NAN": np.nan, "NONE": np.nan, "NULL": np.nan})

        # One-Hot Encoding nhẹ nhàng
        cat_dummies = pd.get_dummies(df_work[cat_cols], dummy_na=False, dtype="float32")
        cat_dummies[group_key] = df_work[group_key]
        cat_agg = cat_dummies.groupby(group_key).agg(["mean", "sum"])
        del cat_dummies
        gc.collect()

    # 5. Kết hợp các đặc trưng tổng hợp
    if num_agg is not None and cat_agg is not None:
        df_agg = pd.concat([num_agg, cat_agg], axis=1)
    elif num_agg is not None:
        df_agg = num_agg
    elif cat_agg is not None:
        df_agg = cat_agg
    else:
        raise ValueError(f"Bảng {table_name} không có cột định lượng hoặc định tính khả dụng!")

    del df_work
    gc.collect()

    elapsed = time.time() - t0
    if verbose:
        print(f"  ✓ Nén 1:N hoàn tất trong {elapsed:.1f}s | Ma trận nén: {df_agg.shape}")

    return df_agg


# =============================================================================
# HÀM LÀM PHẲNG SCHEMA & GẮN TIỀN TỐ (PREFIXING & FLATTENING)
# =============================================================================

def flatten_and_prefix(
    df_agg: pd.DataFrame,
    prefix: str,
    group_key: str = DEFAULT_GROUP_KEY,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Làm phẳng MultiIndex columns thành: [PREFIX]_[TÊN_CỘT]_[PHÉP_TOÁN]
    và đưa group_key (SK_ID_CURR) trở lại thành cột dữ liệu phẳng độc nhất.
    """
    clean_prefix = prefix.strip().upper()
    if not clean_prefix.endswith("_"):
        clean_prefix = f"{clean_prefix}_"

    df = df_agg.copy()

    # 1. Làm phẳng MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        flat_columns = []
        for col in df.columns:
            base_col, stat_op = str(col[0]), str(col[1])
            col_name = f"{clean_prefix}{base_col}_{stat_op}".upper()
            flat_columns.append(col_name)
        df.columns = flat_columns
    else:
        df.columns = [
            f"{clean_prefix}{col}".upper() if not str(col).upper().startswith(clean_prefix) else str(col).upper()
            for col in df.columns
        ]

    # Loại bỏ bất kỳ cột nào bị trùng lặp tên do ép hoa
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()]

    # 2. Phục hồi khóa chính từ index
    if df.index.name == group_key or group_key not in df.columns:
        df = df.reset_index()
        if df.columns[0] != group_key and group_key in df.columns:
            pass
        elif df.columns[0] != group_key:
            df.rename(columns={df.columns[0]: group_key}, inplace=True)

    # 3. Chốt chặn toàn vẹn 1:1
    assert group_key in df.columns, f"Thiếu khóa chính '{group_key}' sau khi reset index!"
    dup_count = df[group_key].duplicated().sum()
    assert dup_count == 0, f"Lỗi: Phát hiện {dup_count} khóa '{group_key}' bị trùng lặp!"

    if verbose:
        print(f"  ✓ Gắn tiền tố [{clean_prefix}] thành công | {df.shape[1]-1} đặc trưng mới")

    return df


# =============================================================================
# HÀM GHÉP NỐI AN TOÀN (SAFE LEFT JOIN VỚI CHỐT CHẶN BẢO TOÀN DÒNG)
# =============================================================================

def safe_left_join(
    df_main: pd.DataFrame,
    df_sec: pd.DataFrame,
    table_name: str,
    group_key: str = DEFAULT_GROUP_KEY,
    expected_rows: int = DEFAULT_EXPECTED_ROWS,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Thực hiện Left Join với 2 chốt chặn an toàn (Guardrails):
    1. Chống bùng nổ dòng: len(df_merged) == expected_rows (Zero Fan-out).
    2. Chống xung đột tên cột: 0 cột mang hậu tố _x, _y.
    """
    t0 = time.time()
    n_before = len(df_main)
    n_cols_sec = df_sec.shape[1] - 1

    # Left Join
    df_merged = pd.merge(df_main, df_sec, on=group_key, how="left")

    # Guardrail 1: Số dòng phải giữ nguyên tuyệt đối
    n_after = len(df_merged)
    assert n_after == expected_rows, (
        f"LỖI BÙNG NỔ DÒNG (ROW EXPLOSION) TẠI BẢNG [{table_name}]!\n"
        f"Số dòng bị tăng/thay đổi từ {n_before:,} thành {n_after:,} (Kỳ vọng: {expected_rows:,})."
    )

    # Guardrail 2: Tuyệt đối không sinh cột _x, _y
    collision_cols = [c for c in df_merged.columns if c.endswith("_x") or c.endswith("_y")]
    assert len(collision_cols) == 0, (
        f"LỖI XUNG ĐỘT TÊN CỘT TẠI BẢNG [{table_name}]!\n"
        f"Phát hiện {len(collision_cols)} cột mang đuôi _x/_y: {collision_cols[:8]}"
    )

    # Downcast nhanh các cột float64 sang float32 để giữ RAM mát mẻ
    for c in df_merged.select_dtypes(include=["float64"]).columns:
        df_merged[c] = df_merged[c].astype("float32")

    elapsed = time.time() - t0
    if verbose:
        print(f"  ✓ Ghép an toàn [{table_name}] trong {elapsed:.1f}s | "
              f"+{n_cols_sec} cột mới | Tổng: {df_merged.shape[1]} cột | Bảo toàn {n_after:,} dòng (Zero Fan-out)")

    del df_sec
    gc.collect()
    return df_merged


# =============================================================================
# HÀM ĐIỀU PHỐI CHÍNH BƯỚC 2 (MAIN RUNNER)
# =============================================================================

def run_merge_pipeline(
    input_dir: Optional[Union[str, Path]] = None,
    output_file: Optional[Union[str, Path]] = None,
    expected_rows: int = DEFAULT_EXPECTED_ROWS
) -> Dict[str, Any]:
    """
    Thực thi trọn gói kịch bản Nén 1:N và Ghép nối đa bảng:
    - Đọc application_train.parquet (234,072 dòng).
    - Tuần tự nén và ghép 5 bảng phụ (bureau, previous_application, POS, installments, credit_card).
    - Kiểm soát nghiêm ngặt 234,072 dòng qua từng bước ghép nối.
    - Xuất file Parquet ma trận đặc trưng hợp nhất: train_merged.parquet.
    """
    total_start = time.time()
    in_path, out_file = resolve_data_paths(input_dir, output_file)

    print("=" * 85)
    print("KHỞI CHẠY BƯỚC 2: NÉN 1:N & GHÉP NỐI ĐA BẢNG AN TOÀN (SAFE MERGE PIPELINE)")
    print("=" * 85)
    print(f"• Thư mục bảng sạch đầu vào (Input) : {in_path}")
    print(f"• Ma trận đặc trưng đầu ra (Output): {out_file}")
    # 1. Đọc bảng chính application_train.parquet
    app_file = in_path / "application_train.parquet"
    if not app_file.exists():
        raise FileNotFoundError(f"Không tìm thấy file bảng chính: {app_file}")

    print(f"⏳ [0/5] Đang nạp bảng chính: {app_file.name}...")
    t_app = time.time()
    df_main = pd.read_parquet(app_file, engine="pyarrow")
    
    # Downcast nhanh bảng chính
    for col in df_main.select_dtypes(include=["float64"]).columns:
        df_main[col] = df_main[col].astype("float32")
    for col in df_main.select_dtypes(include=["int64"]).columns:
        if col != DEFAULT_GROUP_KEY:
            df_main[col] = df_main[col].astype("int32")

    mem_main_mb = df_main.memory_usage(deep=True).sum() / (1024 * 1024)
    print(f"✓ Nạp xong bảng chính trong {time.time()-t_app:.1f}s | "
          f"Shape: {df_main.shape} | RAM: {mem_main_mb:.1f} MB")
    
    # Thiết lập số dòng chuẩn cần bảo toàn (mặc định lấy theo độ dài bảng chính)
    if expected_rows is None:
        expected_rows = len(df_main)
    else:
        assert len(df_main) == expected_rows, (
            f"Lỗi: Số dòng của bảng chính ({len(df_main):,}) không khớp mục tiêu {expected_rows:,}!"
        )

    print(f"• Số dòng bảo toàn bắt buộc        : {expected_rows:,} dòng (Grain 1:1)")
    print("-" * 85)

    # 2. Tuần tự nén, làm phẳng và ghép nối từng bảng phụ
    merge_history: List[Dict[str, Any]] = []

    for step_idx, (tbl_name, prefix) in enumerate(SECONDARY_SPECS, 1):
        tbl_file = in_path / f"{tbl_name}.parquet"
        print(f"\n{'-'*75}")
        print(f"▶ [{step_idx}/5] XỬ LÝ & GHÉP BẢNG: {tbl_name}.parquet (Tiền tố: '{prefix}')")
        print(f"{'-'*75}")

        if not tbl_file.exists():
            print(f"⚠️ Cảnh báo: Không tìm thấy {tbl_file.name}, bỏ qua bước này.")
            continue

        # 2.1 Đọc bảng phụ parquet
        t_sec = time.time()
        df_sec_raw = pd.read_parquet(tbl_file, engine="pyarrow")
        print(f"  • Đọc xong file trong {time.time()-t_sec:.1f}s (Shape gốc: {df_sec_raw.shape})")

        # 2.2 Nén 1:N về 1:1
        df_agg = aggregate_secondary_table(
            df=df_sec_raw,
            table_name=tbl_name,
            group_key=DEFAULT_GROUP_KEY,
            verbose=True
        )
        del df_sec_raw
        gc.collect()

        # 2.3 Làm phẳng schema và gắn tiền tố
        df_flat = flatten_and_prefix(
            df_agg=df_agg,
            prefix=prefix,
            group_key=DEFAULT_GROUP_KEY,
            verbose=True
        )
        del df_agg
        gc.collect()

        # 2.4 Safe Left Join vào bảng chính
        cols_added = df_flat.shape[1] - 1
        df_main = safe_left_join(
            df_main=df_main,
            df_sec=df_flat,
            table_name=tbl_name,
            group_key=DEFAULT_GROUP_KEY,
            expected_rows=expected_rows,
            verbose=True
        )

        merge_history.append({
            "table": tbl_name,
            "prefix": prefix,
            "cols_added": cols_added,
            "total_cols": df_main.shape[1],
            "current_rows": len(df_main)
        })

    # 3. Xuất file kết quả train_merged.parquet
    print("\n" + "=" * 85)
    print("⏳ ĐANG XUẤT MA TRẬN ĐẶC TRƯNG HỢP NHẤT RA ĐĨA...")
    t_save = time.time()
    
    # Downcast toàn bộ float64 lần cuối
    for col in df_main.select_dtypes(include=["float64"]).columns:
        df_main[col] = df_main[col].astype("float32")

    df_main.to_parquet(out_file, index=False, engine="pyarrow")
    file_size_mb = out_file.stat().st_size / (1024 * 1024)
    final_mem_mb = df_main.memory_usage(deep=True).sum() / (1024 * 1024)
    total_elapsed = time.time() - total_start

    print("=" * 85)
    print("✓ HOÀN TẤT THÀNH CÔNG BƯỚC 2: GHÉP NỐI ĐA BẢNG AN TOÀN")
    print("=" * 85)
    print(f"• Số dòng cuối cùng : {len(df_main):,d} dòng (Khớp 100% mục tiêu, 0 dòng nhân đôi)")
    print(f"• Tổng số đặc trưng : {df_main.shape[1]:,d} cột")
    print(f"• Bộ nhớ RAM chiếm  : {final_mem_mb:.2f} MB (~{final_mem_mb/1024:.2f} GB)")
    print(f"• Dung lượng Parquet: {file_size_mb:.2f} MB trên đĩa ({out_file})")
    print(f"• Thời gian hoàn tất: {total_elapsed:.1f} giây ({total_elapsed/60:.2f} phút)")
    print("-" * 85)
    print("CHI TIẾT ĐÓNG GÓP ĐẶC TRƯNG TỪNG BẢNG:")
    hist_df = pd.DataFrame(merge_history)
    print(hist_df.to_string(index=False))
    print("=" * 85)

    del df_main
    gc.collect()

    return {
        "status": "success",
        "output_file": str(out_file),
        "final_rows": expected_rows,
        "history": merge_history,
        "elapsed_seconds": total_elapsed
    }


if __name__ == "__main__":
    run_merge_pipeline()
