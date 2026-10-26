"""
Module: domain_rules.py
Package: source.pipeline.table_sanitizer.secondary
Mô tả: Áp dụng các quy tắc miền đặc thù và khử ngoại lai cho 5 bảng phụ:
       - Chuẩn hóa mục đích vay NAME_CASH_LOAN_PURPOSE trên previous_application.
       - Khử số âm bất hợp lý trên AMT_*, CNT_*.
       - Khử nhiễu nhân 1,000x trên các chỉ số tài chính AMT_*.
       - Chuẩn hóa khoản trả định kỳ AMT_ANNUITY.
"""

import numpy as np
import pandas as pd


def apply_secondary_domain_rules(df: pd.DataFrame, table_name: str) -> pd.DataFrame:
    """
    Áp dụng các quy tắc xử lý đặc thù theo từng bảng phụ.
    
    Parameters:
        df: DataFrame bảng phụ cần chuẩn hóa.
        table_name: Tên bảng phụ (application_train, bureau, previous_application...).
        
    Returns:
        pd.DataFrame: DataFrame sau khi áp dụng quy tắc miền.
    """
    # 1. Chuẩn hóa đặc thù cột mục đích vay nếu là bảng previous_application
    if table_name == "previous_application" and "NAME_CASH_LOAN_PURPOSE" in df.columns:
        df["NAME_CASH_LOAN_PURPOSE"] = df["NAME_CASH_LOAN_PURPOSE"].astype(str).str.strip().str.title()
        invalid_purposes = {"Xap", "Xna", "Chưa Rõ", "Chua Ro", "Nan", ""}
        df.loc[df["NAME_CASH_LOAN_PURPOSE"].isin(invalid_purposes), "NAME_CASH_LOAN_PURPOSE"] = np.nan

    # 2. Khử số âm và ngoại lai phóng đại 1,000x trên AMT_* và CNT_* của các bảng phụ
    for c in df.columns:
        if c.startswith("AMT_") or c.startswith("CNT_"):
            if pd.api.types.is_numeric_dtype(df[c]):
                if (df[c] < 0).any():
                    df[c] = df[c].abs()
                if c.startswith("AMT_") and (df[c] > 1e7).any():
                    df.loc[df[c] > 1e7, c] /= 1000.0

    # 3. Chuẩn hóa khoản trả định kỳ AMT_ANNUITY
    if "AMT_ANNUITY" in df.columns:
        df.loc[df["AMT_ANNUITY"] <= 0, "AMT_ANNUITY"] = np.nan
        df.loc[df["AMT_ANNUITY"] > 5e5, "AMT_ANNUITY"] /= 1000.0

    return df


if __name__ == "__main__":
    import sys
    if sys.platform.startswith("win"):
        try:
            if hasattr(sys.stdout, "reconfigure"):
                sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    print("\n[DEMO KIỂM THỬ: secondary/domain_rules.py]")
    print("-" * 50)
    df_sample = pd.DataFrame({
        "NAME_CASH_LOAN_PURPOSE": ["xap", "Mua Xe", "Chưa Rõ", "Du Lịch", "XNA"],
        "AMT_PAYMENT": [-500.0, 15000000.0, 2000.0, 3000.0, 4000.0],
        "AMT_ANNUITY": [0.0, 600000.0, 15000.0, 20000.0, 25000.0]
    })
    print("Dữ liệu bảng previous_application trước khi áp quy tắc miền:")
    print(df_sample)
    df_clean = apply_secondary_domain_rules(df_sample, "previous_application")
    print("\nDữ liệu sau khi áp quy tắc miền:")
    print(df_clean)
