"""
Module: merge_pipeline.py
Dự án: PRJ-01 - Dự đoán Rủi ro Vỡ nợ Tín dụng (Home Credit Default Risk)
Mô tả: Pipeline ghép nối tuần tự 5 bảng phụ vào application_train với kiểm soát
       toàn vẹn dòng (chống fan-out), kiểm tra xung đột cột (_x, _y), downcast bộ nhớ
       và giải phóng RAM tức thì.
"""

import gc
import sys
import time
from typing import Dict, List, Optional, Tuple, Union
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


def sequential_merge_pipeline(
    df_main: pd.DataFrame,
    secondary_tables: List[Tuple[str, pd.DataFrame]],
    group_key: str = "SK_ID_CURR",
    expected_rows: int = 307511,
    downcast_new_cols: bool = True,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Thực hiện ghép nối tuần tự các bảng đặc trưng phụ vào bảng chính.

    Parameters:
    -----------
    df_main : pd.DataFrame
        Bảng chính ban đầu (application_train).
    secondary_tables : List[Tuple[str, pd.DataFrame]]
        Danh sách các cặp (tên_bảng, DataFrame_đặc_trưng_đã_aggregate).
        Thứ tự chuẩn:
        1. ('bureau_agg', bureau_agg)
        2. ('prev_agg', prev_agg)
        3. ('pos_agg', pos_agg)
        4. ('install_agg', install_agg)
        5. ('credit_agg', credit_agg)
    group_key : str, default 'SK_ID_CURR'
        Khóa ghép nối chính.
    expected_rows : int, default 307511
        Số dòng mục tiêu bắt buộc phải giữ nguyên (ngăn fan-out).
    downcast_new_cols : bool, default True
        Tự động ép kiểu float64 -> float32, int64 -> int32 cho các cột mới.
    verbose : bool, default True
        In chi tiết tiến trình từng bước.

    Returns:
    --------
    pd.DataFrame
        DataFrame chính đã ghép toàn diện 5 bảng phụ, giữ nguyên 307,511 dòng.
    """
    start_time = time.time()
    n_initial = len(df_main)
    assert n_initial == expected_rows, (
        f"Lỗi: Bảng chính ban đầu có {n_initial:,d} dòng, khác kỳ vọng {expected_rows:,d} dòng!"
    )

    if verbose:
        print("=" * 80)
        print("BẮT ĐẦU PIPELINE GHÉP NỐI TUẦN TỰ 5 BẢNG ĐẶC TRƯNG PHỤ")
        print("=" * 80)
        print(f"• Bảng chính: application_train (Shape gốc: {df_main.shape})")
        print(f"• Khóa ghép: '{group_key}' | Số dòng bảo toàn bắt buộc: {expected_rows:,d}")
        print(f"• Số bảng phụ cần ghép: {len(secondary_tables)}")
        print("-" * 80)

    for step, (tbl_name, df_sec) in enumerate(secondary_tables, 1):
        step_start = time.time()
        cols_before = set(df_main.columns)
        n_cols_sec = df_sec.shape[1] - (1 if group_key in df_sec.columns else 0)

        # 1. Thao tác Merge (Left Join)
        df_main = pd.merge(df_main, df_sec, on=group_key, how="left")

        # 2. Kiểm tra tính toàn vẹn (Integrity Assertions)
        # Check A: Chống bùng nổ dòng (Row Explosion / Fan-out)
        assert len(df_main) == expected_rows, (
            f"LỖI BÙNG NỔ DÒNG (FAN-OUT) TẠI BƯỚC [{tbl_name}]!\n"
            f"Số dòng hiện tại là {len(df_main):,d}, không khớp mục tiêu {expected_rows:,d} dòng. "
            f"Bảng phụ {tbl_name} chưa đạt quan hệ 1:1 trên khóa {group_key}."
        )

        # Check B: Chống sinh hậu tố _x, _y (Column Collision)
        collision_cols = [c for c in df_main.columns if c.endswith("_x") or c.endswith("_y")]
        assert len(collision_cols) == 0, (
            f"LỖI XUNG ĐỘT TÊN CỘT TẠI BƯỚC [{tbl_name}]!\n"
            f"Phát hiện {len(collision_cols)} cột sinh đuôi _x/_y: {collision_cols[:10]}."
        )

        # 3. Downcasting dữ liệu cho các cột mới thêm vào
        new_cols = [c for c in df_main.columns if c not in cols_before]
        if downcast_new_cols:
            for col in new_cols:
                col_dtype = df_main[col].dtype
                if col_dtype == "float64":
                    df_main[col] = df_main[col].astype("float32")
                elif col_dtype == "int64":
                    df_main[col] = df_main[col].astype("int32")

        # 4. Quản lý bộ nhớ: Xóa bảng phụ tạm thời và thu gom rác bộ nhớ
        del df_sec
        gc.collect()

        # Đo dung lượng bộ nhớ RAM hiện tại
        mem_mb = df_main.memory_usage(deep=True).sum() / (1024 * 1024)
        step_elapsed = time.time() - step_start

        if verbose:
            print(
                f"[{step}/{len(secondary_tables)}] Ghép thành công: {tbl_name:<16} | "
                f"+{len(new_cols):>3d} cột | "
                f"Shape: ({df_main.shape[0]:,d}, {df_main.shape[1]:,d}) | "
                f"RAM: {mem_mb:>8.2f} MB | "
                f"Thời gian: {step_elapsed:>5.2f}s"
            )

    total_time = time.time() - start_time
    if verbose:
        print("=" * 80)
        print("✓ HOÀN TẤT TOÀN BỘ PIPELINE GHÉP NỐI AN TOÀN")
        print("=" * 80)
        print(f"• Shape cuối cùng : ({df_main.shape[0]:,d} dòng, {df_main.shape[1]:,d} cột)")
        print(f"• Tổng số cột mới : {df_main.shape[1] - len(cols_before) + len(new_cols):,d} đặc trưng")
        print(f"• Bảo toàn số dòng: {len(df_main):,d} == {expected_rows:,d} (100% không bùng nổ dòng)")
        print(f"• Tổng RAM tiêu thụ: {df_main.memory_usage(deep=True).sum() / (1024 * 1024):.2f} MB")
        print(f"• Tổng thời gian  : {total_time:.2f}s")
        print("=" * 80)

    return df_main
