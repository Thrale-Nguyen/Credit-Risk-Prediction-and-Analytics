"""
Module: column_sanitizer.py
Package: source.pipeline.table_sanitizer
Mô tả: Hàm làm sạch cấp cột:
       - Tự động phát hiện và bóc tách đơn vị tiền tệ/đo lường, dấu âm và phẩy thập phân
         để ép 104 cột object bị pha tạp về dạng số thực (float64).
       - Chuẩn hóa các biến phân loại thực sự (cắt khoảng trắng, chuyển missing ẩn -> np.nan,
         Title Case và ánh xạ giá trị chuẩn).
"""

from typing import Tuple
import numpy as np
import pandas as pd

try:
    from .config import (
        UNIT_CLEAN_REGEX,
        HIDDEN_MISSING_STRINGS,
        SENTINEL_TEXT,
        NUMERIC_KEYWORDS,
        TEXT_MAPPING_RULES,
    )
except (ImportError, ValueError):
    import sys
    from pathlib import Path
    _current = Path(__file__).resolve()
    for _parent in _current.parents:
        if (_parent / "source").exists():
            if str(_parent) not in sys.path:
                sys.path.insert(0, str(_parent))
            break
    from source.pipeline.table_sanitizer.config import (
        UNIT_CLEAN_REGEX,
        HIDDEN_MISSING_STRINGS,
        SENTINEL_TEXT,
        NUMERIC_KEYWORDS,
        TEXT_MAPPING_RULES,
    )


def sanitize_column_series(series: pd.Series, col_name: str) -> Tuple[pd.Series, str]:
    """
    Xử lý làm sạch một Series cột dữ liệu:
    - Nếu là cột số bị ép kiểu sang object: Bóc tách đơn vị, chuyển về số thực float64.
    - Nếu là cột phân loại thực sự: Dọn dẹp khoảng trắng thừa, chuyển missing ẩn về np.nan.
    
    Parameters:
        series: Pandas Series cần làm sạch.
        col_name: Tên cột (dùng để kiểm tra từ khóa ngữ cảnh số).
        
    Returns:
        Tuple[pd.Series, str]: (Series_sau_làm_sạch, loại_hành_động: 'converted_to_numeric' | 'cleaned_text' | 'kept_as_is')
    """
    # Nếu không phải kiểu object/string thì giữ nguyên
    if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
        return series, "kept_as_is"

    # Lấy mẫu giá trị không null để kiểm tra
    non_null = series.dropna()
    if non_null.empty:
        return series, "kept_as_is"

    col_upper = col_name.upper()
    has_numeric_name = any(kw in col_upper for kw in NUMERIC_KEYWORDS) or col_upper.endswith(("_AVG", "_MODE", "_MEDI"))

    # Thử bóc tách Regex đơn vị tiền tệ & đo lường, chuyển dấu gạch chéo đầu số âm thành dấu trừ, và đổi phẩy thập phân thành chấm
    sample_size = min(len(non_null), 2000)
    sample_vals = non_null.sample(n=sample_size, random_state=42) if len(non_null) > sample_size else non_null
    sample_str = (
        sample_vals.astype(str)
        .str.strip()
        .str.replace(r"^/\s*", "-", regex=True)
        .str.replace(",", ".")
    )
    sample_stripped = sample_str.str.replace(UNIT_CLEAN_REGEX, "", regex=True)
    
    parsed_sample = pd.to_numeric(sample_stripped, errors="coerce")
    parse_success_rate = parsed_sample.notna().mean()

    # Tiêu chí quyết định ép về dạng số:
    # 1. Tỷ lệ parse thành công >= 50%
    # 2. Hoặc tên cột có từ khóa số và tỷ lệ parse >= 20%
    if parse_success_rate >= 0.50 or (has_numeric_name and parse_success_rate >= 0.20):
        # Thực hiện ép kiểu toàn bộ cột (chuyển ^/ thành -, đổi , thành ., và bóc đơn vị)
        full_str = (
            series.astype(str)
            .str.strip()
            .str.replace(r"^/\s*", "-", regex=True)
            .str.replace(",", ".")
        )
        full_stripped = full_str.str.replace(UNIT_CLEAN_REGEX, "", regex=True)
        # Thay các sentinel string thành NaN trước khi ép
        full_numeric = pd.to_numeric(full_stripped, errors="coerce")
        return full_numeric, "converted_to_numeric"
    else:
        # Cột chuỗi phân loại thực sự: Cắt khoảng trắng đầu/cuối, chuẩn hóa Title Case và missing ẩn
        cleaned_text = series.astype(str).str.strip()
        is_missing = (
            cleaned_text.str.lower().isin(HIDDEN_MISSING_STRINGS)
            | cleaned_text.str.lower().isin(SENTINEL_TEXT)
            | (cleaned_text == "")
            | (series.isna())
        )
        cleaned_text = cleaned_text.mask(is_missing, np.nan)

        # Chuẩn hóa Title Case cho text hợp lệ
        valid_mask = cleaned_text.notna()
        cleaned_text.loc[valid_mask] = cleaned_text.loc[valid_mask].str.title()

        # Ánh xạ các lỗi chính tả / viết tắt thường gặp về chuẩn quốc tế của Home Credit
        cleaned_text = cleaned_text.replace(TEXT_MAPPING_RULES)
        return cleaned_text, "cleaned_text"


if __name__ == "__main__":
    import sys
    if sys.platform.startswith("win"):
        try:
            if hasattr(sys.stdout, "reconfigure"):
                sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    print("\n[DEMO KIỂM THỬ: column_sanitizer.py]")
    print("-" * 50)
    
    # 1. Thử nghiệm cột số bị nhiễm đơn vị tiền tệ/đo lường
    s_numeric_raw = pd.Series(["100,50 đ", "200.75 VND", "/50 USD", "n/a", "1,200.0"])
    cleaned_num, action_num = sanitize_column_series(s_numeric_raw, "AMT_INCOME_TOTAL")
    print(f"• Cột số AMT_INCOME_TOTAL: Hành động = {action_num}")
    for raw, clean in zip(s_numeric_raw, cleaned_num):
        print(f"   {raw!r:15} -> {clean}")

    # 2. Thử nghiệm cột text phân loại bị nhiễm khoảng trắng, lỗi chính tả
    s_text_raw = pd.Series(["  Revolving Loa - ", "Chưa Rõ", "Cash Loans -", "nan", "Secondary / Secondary Specia"])
    cleaned_txt, action_txt = sanitize_column_series(s_text_raw, "NAME_CONTRACT_TYPE")
    print(f"\n• Cột text NAME_CONTRACT_TYPE: Hành động = {action_txt}")
    for raw, clean in zip(s_text_raw, cleaned_txt):
        print(f"   {raw!r:35} -> {clean!r}")
