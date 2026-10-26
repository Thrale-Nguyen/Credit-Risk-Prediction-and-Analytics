"""
Module: outlier_cleaner.py
Package: source.pipeline.table_sanitizer.application
Mô tả: Chuyên trách khử ngoại lai phi lý và áp dụng quy tắc miền cho bảng application_train:
       - Chuẩn hóa điểm tín dụng EXT_SOURCE_* trong đoạn bắt buộc [0.0, 1.0].
       - Khử số âm bất hợp lý trên các chỉ số tài chính (AMT_*, CNT_*).
       - Khử nhiễu nhân 1,000x trên AMT_CREDIT, AMT_INCOME_TOTAL, AMT_GOODS_PRICE.
       - Chuẩn hóa khoản trả định kỳ AMT_ANNUITY (loại bỏ <= 0, giảm phóng đại).
       - Lọc lính canh CNT_CHILDREN >= 15.
       - Chuẩn hóa giới tính CODE_GENDER về F và M, đưa nhãn rác về NaN.
"""

import numpy as np
import pandas as pd


def clean_application_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Áp dụng các quy tắc khử ngoại lai và làm sạch miền giá trị cho bảng application_train.
    
    Parameters:
        df: DataFrame application_train đã khử nhiễm kiểu dữ liệu.
        
    Returns:
        pd.DataFrame: DataFrame sau khi khử ngoại lai.
    """
    # 1. Khử ngoại lai phi lý trên EXT_SOURCE_* (chuẩn hóa bắt buộc trong [0.0, 1.0])
    for ext_col in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]:
        if ext_col in df.columns:
            invalid_ext = (df[ext_col] < 0.0) | (df[ext_col] > 1.0)
            n_inv = int(invalid_ext.sum())
            if n_inv > 0:
                df.loc[invalid_ext, ext_col] = np.nan
                print(f"  + [EXT_SOURCE] Đã lọc {n_inv:,} giá trị ngoại lai (<0 hoặc >1) trên cột [{ext_col}] về NaN")

    # 2. Khử số âm trên các thước đo tài chính và số lượng (AMT_*, CNT_*)
    for c in df.columns:
        if c.startswith("AMT_") or c.startswith("CNT_"):
            if pd.api.types.is_numeric_dtype(df[c]) and (df[c] < 0).any():
                df[c] = df[c].abs()

    # 3. Khử lính canh âm/0 và ngoại lai phóng đại trên khoản trả định kỳ (AMT_ANNUITY)
    if "AMT_ANNUITY" in df.columns:
        df.loc[df["AMT_ANNUITY"] <= 0, "AMT_ANNUITY"] = np.nan
        df.loc[df["AMT_ANNUITY"] > 5e5, "AMT_ANNUITY"] /= 1000.0

    # 4. Khử nhiễu nhân 1,000x trên các khoản tài chính (> 10,000,000)
    for c in ["AMT_CREDIT", "AMT_INCOME_TOTAL", "AMT_GOODS_PRICE"]:
        if c in df.columns:
            df.loc[df[c] > 1e7, c] /= 1000.0

    # 5. Khử ngoại lai lính canh số con cái: CNT_CHILDREN >= 15 hoặc 9999
    if "CNT_CHILDREN" in df.columns:
        df.loc[df["CNT_CHILDREN"] >= 15, "CNT_CHILDREN"] = np.nan

    # 6. Chuẩn hóa CODE_GENDER về F và M
    if "CODE_GENDER" in df.columns:
        df["CODE_GENDER"] = df["CODE_GENDER"].astype(str).str.strip().str.upper()
        df.loc[df["CODE_GENDER"].isin(["XNA", "NAN"]), "CODE_GENDER"] = np.nan

    return df


if __name__ == "__main__":
    import sys
    if sys.platform.startswith("win"):
        try:
            if hasattr(sys.stdout, "reconfigure"):
                sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    print("\n[DEMO KIỂM THỬ: application/outlier_cleaner.py]")
    print("-" * 50)
    df_sample = pd.DataFrame({
        "EXT_SOURCE_1": [-0.2, 0.75, 1.8],
        "AMT_CREDIT": [25000000.0, 450000.0, 800000.0],
        "AMT_ANNUITY": [0.0, 750000.0, 32000.0],
        "CNT_CHILDREN": [2, 18, 0],
        "CODE_GENDER": ["f", "XNA", "M"]
    })
    print("Dữ liệu thô trước khi khử ngoại lai:")
    print(df_sample)
    df_clean = clean_application_outliers(df_sample)
    print("\nDữ liệu sau khi khử ngoại lai & chuẩn hóa:")
    print(df_clean)
