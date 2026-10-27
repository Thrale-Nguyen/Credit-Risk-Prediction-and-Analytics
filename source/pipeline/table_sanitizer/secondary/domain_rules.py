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

    # 2. Xử lý đặc thù bảng previous_application: lính canh thời gian 365243 & tách biệt AMT_APPLICATION vs AMT_CREDIT
    if table_name == "previous_application":
        # Khử lính canh thời gian kinh điển 365243 (1000 năm / chưa từng xảy ra)
        time_cols = [
            "DAYS_FIRST_DRAWING",
            "DAYS_FIRST_DUE",
            "DAYS_LAST_DUE_1ST_VERSION",
            "DAYS_LAST_DUE",
            "DAYS_TERMINATION",
        ]
        for tc in time_cols:
            if tc in df.columns:
                df.loc[df[tc] == 365243, tc] = np.nan

        # Khử lính canh trên các thước đo tài chính
        for c in ["AMT_APPLICATION", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE", "AMT_DOWN_PAYMENT"]:
            if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
                # Lọc lính canh nhân tạo
                df.loc[df[c].isin([-1, -1.0, 1.0, 9999, 9999.0]), c] = np.nan
                # Khử nhiễu nhân 1,000x
                df.loc[df[c] > 1e7, c] /= 1000.0

        # Với AMT_ANNUITY: khoản trả góp bắt buộc > 0
        if "AMT_ANNUITY" in df.columns:
            df.loc[df["AMT_ANNUITY"] <= 0, "AMT_ANNUITY"] = np.nan
            df.loc[df["AMT_ANNUITY"] > 5e5, "AMT_ANNUITY"] /= 1000.0

        # LƯU Ý NGHIỆP VỤ: Với AMT_CREDIT, GIỮ NGUYÊN giá trị 0.0 cho các hồ sơ Canceled / Refused
        # vì phản ánh đúng thực tế giải ngân (không phát sinh dư nợ), tuyệt đối không gán NaN.

    # 3. Xử lý đặc thù bảng credit_card_balance
    elif table_name == "credit_card_balance":
        # BẢO TOÀN SỐ DƯ ÂM: AMT_BALANCE < 0 thể hiện khách hàng thanh toán dư hạn mức (Overpayment).
        # Tuyệt đối không dùng abs() làm sai lệch hành vi thanh toán tốt của khách hàng.
        for c in df.columns:
            if c.startswith("AMT_") and pd.api.types.is_numeric_dtype(df[c]):
                df.loc[df[c].isin([-1, -1.0, 1.0, 9999, 9999.0]), c] = np.nan
                if (df[c] > 1e7).any():
                    df.loc[df[c] > 1e7, c] /= 1000.0
            elif c.startswith("CNT_") and pd.api.types.is_numeric_dtype(df[c]):
                # Lọc số âm bất hợp lý trên số lượng giao dịch
                df.loc[df[c] < 0, c] = np.nan
                # Lọc sạch các giá trị lính canh nhân tạo: 9999, 100, 1000, 10000, 100000...
                sentinel_cnt = [
                    9999, 9999.0,
                    100, 100.0, 1000, 1000.0, 10000, 10000.0, 100000, 100000.0,
                    200, 200.0, 300, 300.0, 400, 400.0, 500, 500.0, 600, 600.0,
                    2000, 2000.0, 3000, 3000.0, 4000, 4000.0, 5000, 5000.0, 6000, 6000.0
                ]
                df.loc[df[c].isin(sentinel_cnt), c] = np.nan
                # Khử lính canh/ngoại lai vượt ngưỡng vật lý nghiệp vụ
                if c == "CNT_INSTALMENT_MATURE_CUM":
                    df.loc[df[c] > 120.0, c] = np.nan
                else:
                    # Các cột số lần rút ATM, quẹt thẻ... trong 1 tháng không vượt quá 50 lần
                    df.loc[df[c] > 50.0, c] = np.nan

    # 4. Xử lý đặc thù bảng installments_payments
    elif table_name == "installments_payments":
        for c in ["AMT_INSTALMENT", "AMT_PAYMENT"]:
            if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
                df.loc[df[c].isin([-1, -1.0, 1.0, 9999, 9999.0]), c] = np.nan
                if (df[c] > 1e7).any():
                    df.loc[df[c] > 1e7, c] /= 1000.0

    # 5. Xử lý đặc thù bảng bureau & POS_CASH_balance
    else:
        # BẢO TOÀN GIÁ TRỊ 0: AMT_CREDIT_SUM_OVERDUE = 0 thể hiện không có nợ quá hạn (lịch sử tốt)
        # Giữ nguyên giá trị 0, chỉ lọc lính canh nhân tạo 9999 và khử phóng đại
        for c in df.columns:
            if (c.startswith("AMT_") or c.startswith("CNT_")) and pd.api.types.is_numeric_dtype(df[c]):
                df.loc[df[c].isin([9999, 9999.0]), c] = np.nan
                if c.startswith("AMT_") and (df[c] > 1e7).any():
                    df.loc[df[c] > 1e7, c] /= 1000.0

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
