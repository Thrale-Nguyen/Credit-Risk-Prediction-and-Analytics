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


try:
    from ..config import CORE_FINANCIAL_COLS, TIME_SENTINEL_VALUE
except (ImportError, ValueError):
    CORE_FINANCIAL_COLS = ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE"]
    TIME_SENTINEL_VALUE = 365243


def clean_application_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Áp dụng các quy tắc khử ngoại lai và làm sạch miền giá trị cho bảng application_train.
    Tuân thủ cơ chế chọn lọc rõ ràng (Whitelist), không áp dụng quy tắc phủ đầu (blanket rules).
    
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

    # 2. Chuẩn hóa nhóm thước đo tài sản & hạn mức cốt lõi (Cơ chế Whitelist: bắt buộc > 0)
    # Tuyệt đối không dùng quy tắc phủ đầu (blanket rule) gán NaN hay abs() trên toàn bộ cột AMT_*
    for c in CORE_FINANCIAL_COLS:
        if c in df.columns:
            # Bắt triệt để các lính canh phi lý: <= 0, 1.0 (do abs(-1)), và 9999.0
            invalid_mask = (df[c] <= 0) | df[c].isin([1.0, 9999.0])
            n_inv = int(invalid_mask.sum())
            if n_inv > 0:
                df.loc[invalid_mask, c] = np.nan
                print(f"  + [{c}] Đã lọc {n_inv:,} giá trị lính canh/phi lý (<=0, 1.0, 9999) về NaN")

    # Khử nhiễu nhân 1,000x trên các khoản tài chính cốt lõi
    if "AMT_ANNUITY" in df.columns:
        df.loc[df["AMT_ANNUITY"] > 5e5, "AMT_ANNUITY"] /= 1000.0

    for c in ["AMT_CREDIT", "AMT_INCOME_TOTAL", "AMT_GOODS_PRICE"]:
        if c in df.columns:
            df.loc[df[c] > 1e7, c] /= 1000.0

    # 3. Khử ngoại lai lính canh số con cái & số thành viên gia đình
    if "CNT_CHILDREN" in df.columns:
        invalid_child = (df["CNT_CHILDREN"] >= 15) | (df["CNT_CHILDREN"] == 9999) | (df["CNT_CHILDREN"] < 0)
        n_child = int(invalid_child.sum())
        if n_child > 0:
            df.loc[invalid_child, "CNT_CHILDREN"] = np.nan
            print(f"  + [CNT_CHILDREN] Đã lọc {n_child:,} giá trị lính canh (>=15, 9999, <0) về NaN")

    if "CNT_FAM_MEMBERS" in df.columns:
        invalid_fam = (df["CNT_FAM_MEMBERS"] == 9999) | (df["CNT_FAM_MEMBERS"] > 20) | (df["CNT_FAM_MEMBERS"] < 1)
        n_fam = int(invalid_fam.sum())
        if n_fam > 0:
            df.loc[invalid_fam, "CNT_FAM_MEMBERS"] = np.nan
            print(f"  + [CNT_FAM_MEMBERS] Đã lọc {n_fam:,} giá trị lính canh (9999, >20, <1) về NaN")

    # 4. Xử lý lính canh thời gian kinh điển: DAYS_EMPLOYED == 365243 (hưu trí / không có việc)
    if "DAYS_EMPLOYED" in df.columns:
        anom_mask = (df["DAYS_EMPLOYED"] == TIME_SENTINEL_VALUE)
        n_anom = int(anom_mask.sum())
        if n_anom > 0:
            df["DAYS_EMPLOYED_ANOM"] = anom_mask.astype(np.int8)
            df.loc[anom_mask, "DAYS_EMPLOYED"] = np.nan
            print(f"  + [DAYS_EMPLOYED] Bắt dị thường hưu trí: Đã tạo cờ [DAYS_EMPLOYED_ANOM] ({n_anom:,} ca = 1) và gán NaN")

    # 5. Chuẩn hóa CODE_GENDER về F và M
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
