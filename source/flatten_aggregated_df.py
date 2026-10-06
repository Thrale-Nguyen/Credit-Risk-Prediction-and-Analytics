"""
Module: flatten_aggregated_df.py
Dự án: PRJ-01 - Dự đoán Rủi ro Vỡ nợ Tín dụng (Home Credit Default Risk)
Mô tả: Chuẩn hóa schema sau khi aggregate: Làm phẳng MultiIndex, gắn tiền tố bảng
       và phục hồi khóa chính SK_ID_CURR về cột dữ liệu 1:1.
"""

import sys
from typing import Optional
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


def flatten_aggregated_df(
    df_agg: pd.DataFrame,
    prefix: str,
    group_key: str = "SK_ID_CURR",
    verbose: bool = True
) -> pd.DataFrame:
    """
    Chuẩn hóa schema sau khi aggregate bảng phụ:
    1. Làm phẳng MultiIndex thành: [PREFIX]_[TÊN_CỘT_GỐC]_[PHÉP_TOÁN]
    2. Phục hồi khóa chính `group_key` (SK_ID_CURR) từ index trở lại cột thông thường không dính prefix.
    3. Validation check: Assert kiểm tra tính duy nhất 1:1 và nghiệm thu shape, cột.

    Parameters:
    -----------
    df_agg : pd.DataFrame
        DataFrame từ hàm aggregate_secondary_table (MultiIndex columns, index là group_key).
    prefix : str
        Tiền tố đại diện cho bảng nguồn (vd: 'BUREAU', 'PREV', 'POS', 'INSTAL', 'CC').
    group_key : str, default 'SK_ID_CURR'
        Tên khóa chính cần phục hồi.
    verbose : bool, default True
        Nếu True, in kết quả nghiệm thu sau chuẩn hóa.

    Returns:
    --------
    pd.DataFrame
        DataFrame với schema phẳng, cột đầu tiên là SK_ID_CURR, các cột còn lại viết hoa kèm prefix.
    """
    df = df_agg.copy()

    # Chuẩn hóa tiền tố (đảm bảo luôn kết thúc bằng dấu gạch dưới '_')
    clean_prefix = prefix.strip().upper()
    if not clean_prefix.endswith("_"):
        clean_prefix = f"{clean_prefix}_"

    # 1. Làm phẳng MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            f"{clean_prefix}{col[0]}_{col[1]}".upper()
            for col in df.columns
        ]
    else:
        # Trường hợp cột đã phẳng, chỉ gắn thêm prefix nếu chưa có
        df.columns = [
            f"{clean_prefix}{col}".upper() if not col.upper().startswith(clean_prefix) else col.upper()
            for col in df.columns
        ]

    # 2. Phục hồi khóa chính từ index trở lại cột dữ liệu thông thường
    # Vì SK_ID_CURR đang ở index khi đổi tên cột ở trên, nó hoàn toàn KHÔNG bị dính tiền tố
    if df.index.name == group_key or group_key not in df.columns:
        df = df.reset_index()
        # Đảm bảo tên cột khóa chính luôn chuẩn xác
        if df.columns[0] != group_key and group_key in df.columns:
            pass
        elif df.columns[0] != group_key:
            df.rename(columns={df.columns[0]: group_key}, inplace=True)

    # 3. Validation Check
    assert group_key in df.columns, f"Lỗi: Không tìm thấy khóa chính '{group_key}' trong DataFrame!"
    assert df[group_key].duplicated().sum() == 0, (
        f"Lỗi vi phạm toàn vẹn: Tồn tại {df[group_key].duplicated().sum()} giá trị "
        f"'{group_key}' bị trùng lặp! Quan hệ không đạt 1 : 1."
    )

    if verbose:
        print("=" * 65)
        print(f"✓ CHUẨN HÓA SCHEMA THÀNH CÔNG CHO TIỀN TỐ [{clean_prefix}]")
        print("=" * 65)
        print(f"• Số dòng (Unique {group_key}): {len(df):,d} dòng")
        print(f"• Số lượng cột mới: {df.shape[1]:,d} cột")
        print(f"• Khóa chính phục hồi: '{group_key}' (Độc nhất 100%, 0 trùng lặp)")
        print(f"• 5 cột đầu tiên:\n  {df.columns[:5].tolist()}")
        print("=" * 65)

    return df


if __name__ == "__main__":
    # Demo kiểm thử trực tiếp
    import numpy as np

    sample_index = pd.Index([100001, 100002, 100003], name="SK_ID_CURR")
    sample_cols = pd.MultiIndex.from_tuples([
        ("DAYS_CREDIT", "min"),
        ("DAYS_CREDIT", "mean"),
        ("AMT_CREDIT_SUM", "sum"),
        ("CREDIT_ACTIVE_Active", "mean")
    ])
    sample_df = pd.DataFrame(np.random.randn(3, 4), index=sample_index, columns=sample_cols)

    flat_df = flatten_aggregated_df(sample_df, prefix="BUREAU", verbose=True)
