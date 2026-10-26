"""
Module: orphan_filter.py
Package: source.pipeline.table_sanitizer.secondary
Mô tả: Chuyên trách khử trùng lặp và loại bỏ các bản ghi mang khóa ngoại mồ côi (Orphan Foreign Keys)
       trong 5 bảng phụ mà không tồn tại trong danh sách khách hàng hợp lệ của application_train.
"""

from typing import Set, Tuple
import pandas as pd


def filter_orphan_foreign_keys(
    df: pd.DataFrame,
    valid_sk_ids: Set[int],
    table_name: str
) -> Tuple[pd.DataFrame, int]:
    """
    Lọc bỏ các bản ghi trong bảng phụ có khóa ngoại SK_ID_CURR không tồn tại trong valid_sk_ids.
    
    Parameters:
        df: DataFrame bảng phụ cần lọc.
        valid_sk_ids: Tập hợp SK_ID_CURR hợp lệ từ application_train.
        table_name: Tên bảng (để ghi log).
        
    Returns:
        Tuple[pd.DataFrame, int]: (DataFrame_đã_lọc_hợp_lệ, số_dòng_mồ_côi_bị_loại)
    """
    initial_rows = len(df)
    if "SK_ID_CURR" not in df.columns:
        print(f"ℹ Bảng {table_name} không chứa cột SK_ID_CURR trực tiếp.")
        return df, 0

    before_orphan = len(df)
    # Ép SK_ID_CURR sang số để so khớp chính xác
    df["SK_ID_CURR"] = pd.to_numeric(df["SK_ID_CURR"], errors="coerce")
    df = df[df["SK_ID_CURR"].notna()].copy()
    df["SK_ID_CURR"] = df["SK_ID_CURR"].astype(int)

    # Lọc theo valid_sk_ids
    df_clean = df[df["SK_ID_CURR"].isin(valid_sk_ids)].copy()
    orphan_removed = before_orphan - len(df_clean)

    print(f"• Đã lọc bỏ {orphan_removed:,} dòng mang khóa ngoại mồ côi (không tồn tại trong application_train)")
    print(f"• Số dòng hợp lệ còn lại: {len(df_clean):,} dòng ({(len(df_clean)/initial_rows*100):.2f}%)")

    return df_clean, orphan_removed


if __name__ == "__main__":
    import sys
    if sys.platform.startswith("win"):
        try:
            if hasattr(sys.stdout, "reconfigure"):
                sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    print("\n[DEMO KIỂM THỬ: secondary/orphan_filter.py]")
    print("-" * 50)
    df_sample = pd.DataFrame({
        "SK_ID_CURR": [1001, 1002, 9999, 1001, 8888, "1002", "null"],
        "VAL": [10, 20, 30, 40, 50, 60, 70]
    })
    valid_ids = {1001, 1002}
    print(f"Danh sách khách hàng hợp lệ valid_sk_ids = {valid_ids}")
    print("Dữ liệu bảng phụ trước khi lọc khóa ngoại mồ côi:")
    print(df_sample)
    df_clean, orphan_count = filter_orphan_foreign_keys(df_sample, valid_ids, "demo_bureau")
    print(f"\nDữ liệu bảng phụ sau khi lọc (Đã xóa {orphan_count} dòng mồ côi):")
    print(df_clean)
