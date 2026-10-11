"""
Module: business_rules.py
Package: source.data_cleaning
Dự án: PRJ-01 - Dự đoán Rủi ro Vỡ nợ Tín dụng (Home Credit Default Risk)
Mô tả: Bước 4 trong pipeline làm sạch dữ liệu:
       - Xử lý ngoại lai theo Quy tắc Nghiệp vụ thực tế (Business Logic Sanity Check):
         + HỦY BỎ hoàn toàn cơ chế Winsorization / kẹp trần cứng tại mốc phân vị 99% (472,500).
         + Bảo toàn tính liên tục và độ đơn điệu của dải thu nhập cho các mô hình dạng cây
           (LightGBM & CatBoost) tự do phân tách điểm cắt (split point).
         + Chỉ áp dụng kiểm soát lỗi khai báo / lỗi gõ phím hệ thống phi lý (ví dụ: > 10,000,000).
         + Hỗ trợ action='clip' (giữ nguyên số dòng) hoặc action='filter' (chỉ áp dụng trên train).
"""

import sys
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

# Đảm bảo console Windows in tiếng Việt chuẩn UTF-8
if sys.platform.startswith("win"):
    try:
        if sys.stdout and hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if sys.stderr and hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def handle_business_outliers(
    df: pd.DataFrame,
    is_train: bool = True,
    target_col: str = "AMT_INCOME_TOTAL",
    max_threshold: float = 10_000_000.0,
    action: str = "clip",
    outlier_state: Optional[Dict[str, float]] = None,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Xử lý ngoại lai tài chính theo Quy tắc Nghiệp vụ Thực tế (Business Sanity Rule):
    - Thay thế hoàn toàn phương pháp Winsorization 99th percentile (472,500) vốn làm mất
      thông tin phân hóa của 3,075 khách hàng có thu nhập cao chính đáng.
    - Áp dụng kiểm tra ngưỡng phi lý thực tế (mặc định > 10,000,000), nơi tập trung
      các bản ghi lỗi hệ thống hoặc khai báo sai lệch quá mức.
    - Duy trì dải giá trị liên tục từ 25k đến 10 triệu để mô hình cây học phân loại rủi ro.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame cần xử lý.
    is_train : bool, default True
        True: Huấn luyện; False: Áp dụng ngưỡng từ Train cho Test.
    target_col : str, default 'AMT_INCOME_TOTAL'
        Tên cột thu nhập cần kiểm soát lỗi ngoại lai.
    max_threshold : float, default 10,000,000.0
        Ngưỡng trần tối đa hợp lý cho thu nhập (10 triệu CZK).
    action : str, default 'clip'
        Hành vi xử lý:
        - 'clip': Kẹp giá trị vượt ngưỡng về đúng max_threshold (Bảo toàn 100% số dòng Grain).
        - 'filter': Lọc bỏ các dòng vượt ngưỡng (chỉ thực hiện khi is_train=True).
    outlier_state : Optional[Dict[str, float]], default None
        Dictionary lưu vết ngưỡng max_threshold để áp dụng cho tập test chống rò rỉ dữ liệu.
    verbose : bool, default True
        Nếu True, in chi tiết log nghiệm thu.

    Returns:
    --------
    Tuple[pd.DataFrame, Dict[str, float]]
        (DataFrame sau xử lý, Dictionary trạng thái ngưỡng)
    """
    if target_col not in df.columns:
        if verbose:
            print(f"[CẢNH BÁO] Không tìm thấy cột '{target_col}' trong DataFrame.")
        return df, outlier_state or {}

    rows_before = len(df)
    if is_train:
        threshold = float(max_threshold)
        outlier_state = {"col": target_col, "max_threshold": threshold, "action": action}
    else:
        threshold = outlier_state.get("max_threshold", max_threshold) if outlier_state else max_threshold

    # Phát hiện các dòng vượt ngưỡng
    extreme_mask = df[target_col] > threshold
    num_extreme = int(extreme_mask.sum())
    max_val_before = float(df[target_col].max())

    if num_extreme > 0:
        if action == "clip" or not is_train:
            # Kẹp trần giá trị về max_threshold để bảo toàn số dòng
            df[target_col] = df[target_col].clip(upper=threshold)
            processed_desc = f"Đã kẹp trần {num_extreme} bản ghi (> {threshold:,.0f}) về đúng {threshold:,.0f}"
        elif action == "filter" and is_train:
            # Lọc bỏ dòng lỗi trên tập train
            df = df.loc[~extreme_mask].copy()
            processed_desc = f"Đã lọc bỏ {num_extreme} bản ghi lỗi hệ thống (> {threshold:,.0f})"
        else:
            processed_desc = f"Không can thiệp ({num_extreme} bản ghi)"
    else:
        processed_desc = f"Không có bản ghi nào vượt ngưỡng {threshold:,.0f}"

    max_val_after = float(df[target_col].max())

    if verbose:
        print("\n" + "=" * 80)
        print("BƯỚC 4: XỬ LÝ NGOẠI LAI THEO QUY TẮC NGHIỆP VỤ THỰC TẾ (BUSINESS SANITY RULE)")
        print("=" * 80)
        print(f"• Cột kiểm soát                     : {target_col}")
        print(f"• Cơ chế xử lý                     : Nghiệp vụ (Ngưỡng tối đa hợp lý: {threshold:,.0f})")
        print(f"• Giá trị cực đại ban đầu           : {max_val_before:,.2f}")
        print(f"• Số lượng bản ghi vượt ngưỡng      : {num_extreme:,d} dòng")
        print(f"• Kết quả xử lý                    : {processed_desc}")
        print(f"• Giá trị cực đại sau xử lý         : {max_val_after:,.2f}")
        print(f"• Số dòng dữ liệu (Grain bảo toàn)  : {rows_before:,d} -> {len(df):,d} dòng")
        print("-" * 80)

    return df, outlier_state
