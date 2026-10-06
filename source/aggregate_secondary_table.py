"""
Module: aggregate_secondary_table.py
Dự án: PRJ-01 - Dự đoán Rủi ro Vỡ nợ Tín dụng (Home Credit Default Risk)
Mô tả: Hàm aggregate dữ liệu từ bảng phụ (bureau, previous_application, POS_CASH,
       installments, credit_card) về cấp độ khách hàng SK_ID_CURR (quan hệ 1 : 1).
"""

import gc
import sys
import time
from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd

# Đảm bảo console Windows hỗ trợ in Unicode tiếng Việt mượt mà
if sys.platform.startswith("win"):
    try:
        if sys.stdout and hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if sys.stderr and hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def aggregate_secondary_table(
    df: pd.DataFrame,
    group_key: str = "SK_ID_CURR",
    exclude_keys: Optional[List[str]] = None,
    num_aggregations: Optional[Union[List[str], Dict[str, List[str]]]] = None,
    cat_aggregations: Optional[List[str]] = None,
    downcast: bool = True,
    in_place: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Aggregate một DataFrame phụ về cấp độ `group_key` (mặc định 'SK_ID_CURR')
    để đảm bảo quan hệ 1 : 1 với bảng chính application_train/application_test.

    Quy trình kỹ thuật bắt buộc:
    1. Tách loại biến:
       - Biến định danh: Loại trừ hoàn toàn các khóa phụ (SK_ID_PREV, SK_ID_BUREAU)
         khỏi danh sách tính toán thống kê; chỉ giữ SK_ID_CURR làm khóa gom nhóm.
       - Biến phân loại: Dùng pd.get_dummies(..., dummy_na=False) để One-Hot Encode trước,
         sau đó aggregate bằng các phép ['mean', 'sum'].
       - Biến định lượng: Aggregate bằng ['min', 'max', 'mean', 'sum'] (với các cột đo lường
         tài chính) hoặc ['count'] để đếm tần suất.
    2. Kiểm soát tài nguyên:
       - Chuyển đổi kiểu dữ liệu (float64 -> float32, int64 -> int32) trước khi aggregate
         để giảm ~50% bộ nhớ RAM, chống tràn RAM trên các bảng hàng triệu dòng.
       - Dọn rác bộ nhớ tức thì bằng `gc.collect()`.
    3. Output:
       - Trả về DataFrame đã gom nhóm theo SK_ID_CURR, giữ nguyên index là SK_ID_CURR
         và MultiIndex columns (chưa reset index vội) phục vụ bước làm phẳng tên cột.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame phụ cần aggregate (vd: previous_application, bureau, installments...).
    group_key : str, default 'SK_ID_CURR'
        Khóa chính dùng để gom nhóm (liên kết với bảng application_train).
    exclude_keys : List[str], optional
        Danh sách các cột khóa phụ cần loại trừ khỏi tính toán thống kê.
        Mặc định: ['SK_ID_PREV', 'SK_ID_BUREAU'].
    num_aggregations : List[str] | Dict[str, List[str]], optional
        Các phép tính aggregate cho biến định lượng.
        Mặc định: ['min', 'max', 'mean', 'sum'].
    cat_aggregations : List[str], optional
        Các phép tính aggregate cho biến phân loại sau khi One-Hot Encoding.
        Mặc định: ['mean', 'sum'].
    downcast : bool, default True
        Nếu True, chuyển đổi kiểu dữ liệu float64 -> float32, int64 -> int32.
    in_place : bool, default False
        Nếu True, thao tác ép kiểu trực tiếp trên DataFrame đầu vào (tiết kiệm RAM tối đa).
        Nếu False, tạo bản sao chọn lọc của các cột cần dùng.
    verbose : bool, default True
        Nếu True, in thông tin tiến trình thực hiện, thời gian và kích thước đầu ra.

    Returns:
    --------
    pd.DataFrame
        DataFrame đã aggregate theo `group_key` với MultiIndex columns, index là `group_key`.
    """
    start_time = time.time()

    # Kiểm tra khóa gom nhóm
    if group_key not in df.columns:
        raise KeyError(
            f"Không tìm thấy khóa gom nhóm '{group_key}' trong DataFrame. "
            f"Các cột hiện có: {df.columns.tolist()}"
        )

    # Cấu hình danh sách khóa phụ cần loại trừ
    default_exclude = ["SK_ID_PREV", "SK_ID_BUREAU"]
    if exclude_keys is None:
        exclude_cols = default_exclude
    else:
        exclude_cols = list(set(exclude_keys + default_exclude))

    # Cấu hình các phép aggregate mặc định
    if num_aggregations is None:
        num_aggregations = ["min", "max", "mean", "sum"]
    if cat_aggregations is None:
        cat_aggregations = ["mean", "sum"]

    # -------------------------------------------------------------
    # 1. Lọc cột và Kiểm soát tài nguyên (Memory Management & Downcast)
    # -------------------------------------------------------------
    # Loại trừ các khóa phụ khỏi tập dữ liệu tính toán
    cols_to_keep = [c for c in df.columns if c not in exclude_cols or c == group_key]

    if in_place:
        work_df = df[cols_to_keep]
    else:
        work_df = df[cols_to_keep].copy()

    # Đo bộ nhớ trước khi downcast
    mem_before_mb = work_df.memory_usage(deep=True).sum() / (1024 * 1024)

    if downcast:
        # float64 --> float32
        float_cols = work_df.select_dtypes(include=["float64"]).columns
        for col in float_cols:
            work_df[col] = work_df[col].astype("float32")

        # int64 --> int32
        int_cols = work_df.select_dtypes(include=["int64"]).columns
        for col in int_cols:
            work_df[col] = work_df[col].astype("int32")

    mem_after_mb = work_df.memory_usage(deep=True).sum() / (1024 * 1024)

    # -------------------------------------------------------------
    # 2. Tách loại biến (Categorical vs Numerical)
    # -------------------------------------------------------------
    # Biến phân loại (object, category)
    cat_cols = work_df.select_dtypes(include=["object", "category"]).columns.tolist()
    # Loại trừ group_key nếu vô tình là object
    if group_key in cat_cols:
        cat_cols.remove(group_key)

    # Biến định lượng (number)
    num_cols = [
        c for c in work_df.select_dtypes(include=["number"]).columns if c != group_key
    ]

    if verbose:
        print(f"--- Bắt đầu Aggregate theo '{group_key}' ---")
        print(f"• Số dòng gốc: {len(df):,d} | Unique {group_key}: {df[group_key].nunique():,d}")
        print(f"• Bộ nhớ DataFrame làm việc: {mem_before_mb:.2f} MB --> {mem_after_mb:.2f} MB (giảm {((mem_before_mb - mem_after_mb) / mem_before_mb * 100):.1f}%)" if mem_before_mb > 0 else "")
        print(f"• Khóa phụ đã loại trừ: {[c for c in exclude_cols if c in df.columns]}")
        print(f"• Số biến định lượng: {len(num_cols)} cột -> agg={num_aggregations}")
        print(f"• Số biến phân loại: {len(cat_cols)} cột -> OHE + agg={cat_aggregations}")

    # -------------------------------------------------------------
    # 3. Aggregate biến định lượng (Numerical Features)
    # -------------------------------------------------------------
    num_agg = None
    if num_cols:
        t_num = time.time()
        if isinstance(num_aggregations, dict):
            num_agg = work_df[[group_key] + list(num_aggregations.keys())].groupby(group_key).agg(num_aggregations)
        else:
            num_agg = work_df[[group_key] + num_cols].groupby(group_key).agg(num_aggregations)
        if verbose:
            print(f"  ✓ Aggregate định lượng hoàn tất trong {time.time() - t_num:.2f}s (Shape: {num_agg.shape})")

    # -------------------------------------------------------------
    # 4. One-Hot Encode & Aggregate biến phân loại (Categorical Features)
    # -------------------------------------------------------------
    cat_agg = None
    if cat_cols:
        t_cat = time.time()
        # One-Hot Encoding bằng pd.get_dummies(..., dummy_na=False) với kiểu float32 tiết kiệm RAM
        cat_dummies = pd.get_dummies(work_df[cat_cols], dummy_na=False, dtype="float32")
        cat_dummies[group_key] = work_df[group_key]

        cat_agg = cat_dummies.groupby(group_key).agg(cat_aggregations)

        # Dọn dẹp dummies trung gian lập tức
        del cat_dummies
        gc.collect()

        if verbose:
            print(f"  ✓ OHE & Aggregate phân loại hoàn tất trong {time.time() - t_cat:.2f}s (Shape: {cat_agg.shape})")

    # -------------------------------------------------------------
    # 5. Ghép nối kết quả (chưa reset index, giữ MultiIndex columns)
    # -------------------------------------------------------------
    if num_agg is not None and cat_agg is not None:
        df_agg = pd.concat([num_agg, cat_agg], axis=1)
    elif num_agg is not None:
        df_agg = num_agg
    elif cat_agg is not None:
        df_agg = cat_agg
    else:
        raise ValueError("Không tìm thấy biến định lượng hoặc phân loại nào khả dụng để aggregate.")

    total_time = time.time() - start_time
    if verbose:
        print(f"✓ Hoàn tất toàn bộ pipeline trong {total_time:.2f}s!")
        print(f"• Kích thước đầu ra: {df_agg.shape} (1 dòng / 1 {group_key})")
        print(f"• Index name: '{df_agg.index.name}' | Columns type: MultiIndex ({isinstance(df_agg.columns, pd.MultiIndex)})")
        print("-" * 50)

    return df_agg


if __name__ == "__main__":
    # Demo thử nghiệm với bảng previous_application nếu có sẵn
    from pathlib import Path

    sample_csv = Path(__file__).resolve().parent.parent / "Dataset - Home Credit Default Risk" / "bureau.csv"
    if sample_csv.exists():
        print(f"Thử nghiệm đọc 1,000 dòng từ {sample_csv.name}...")
        sample_df = pd.read_csv(sample_csv, nrows=1000)
        res = aggregate_secondary_table(sample_df, verbose=True)
        print("\nMultiIndex Columns mẫu:")
        print(res.columns[:5])
        print("\nDữ liệu mẫu:")
        print(res.iloc[:3, :4])
