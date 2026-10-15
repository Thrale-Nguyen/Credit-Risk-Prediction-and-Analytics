"""
Script: audit_raw_data.py
Mục đích: Hệ thống Kiểm toán Chuyên sâu Toàn diện (Forensic Data Audit Suite)
         So sánh tập dữ liệu mới (data/raw) với tập dữ liệu cũ ban đầu (data/old_data).
         Bóc tách 100% các bẫy làm dơ dữ liệu: Missing ẩn, Trùng lặp, Vi phạm Grain,
         Dị thường nhãn TARGET, Đảo dấu DAYS_*, Khóa ngoại mồ côi, và Ngoại lai cực đoan.
"""

import sys
import io
import gc
import time
from pathlib import Path
from typing import Dict, List, Set, Optional
import numpy as np
import pandas as pd

# Thiết lập encoding UTF-8 cho Windows console
if sys.platform.startswith("win"):
    try:
        if sys.stdout and hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if sys.stderr and hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

OLD_DIR = Path("data/old_data")
RAW_DIR = Path("data/raw")

FILES = [
    "application_train.csv",
    "bureau.csv",
    "previous_application.csv",
    "POS_CASH_balance.csv",
    "installments_payments.csv",
    "credit_card_balance.csv",
]

# Danh mục mở rộng toàn diện các ký hiệu missing ẩn / sentinel values
HIDDEN_MISSING_PATTERNS = {
    # Dạng chữ
    "?", "n/a", "na", "nan", "null", "none", "missing", "unknown", 
    "#n/a", "#na", "n.a.", "n/k", "nil", "void", "empty",
    # Dấu ký tự
    "-", "--", "---", ".", "", " ", "\t", "\n", "\xa0",
    # Số lính canh (Sentinel values)
    "-1", "-1.0", "-99", "-999", "-9999", "999", "9999", "99999", "999999",
    # Mã kỹ thuật Home Credit (theo dõi riêng)
    "xna", "xap"
}

# Biến toàn cục lưu danh sách ID khách hàng bảng chính để kiểm tra khóa ngoại mồ côi
MAIN_TRAIN_CURR_IDS: Optional[Set[int]] = None


def audit_single_file(filename: str):
    global MAIN_TRAIN_CURR_IDS
    print("\n" + "=" * 95)
    print(f"🔍 BẮT ĐẦU KIỂM TOÁN TẬP TIN: {filename}")
    print("=" * 95)

    old_path = OLD_DIR / filename
    raw_path = RAW_DIR / filename

    if not raw_path.exists():
        print(f"❌ Không tìm thấy file trong data/raw: {raw_path}")
        return

    # 1. KÍCH THƯỚC FILE TRÊN ĐĨA
    old_size_mb = old_path.stat().st_size / (1024 * 1024) if old_path.exists() else 0
    raw_size_mb = raw_path.stat().st_size / (1024 * 1024)
    print(f"• Dung lượng đĩa: Cũ (Old) = {old_size_mb:.2f} MB | Mới (Raw) = {raw_size_mb:.2f} MB (Chênh lệch: {raw_size_mb - old_size_mb:+.2f} MB)")

    # 2. ĐỌC DỮ LIỆU
    t0 = time.time()
    print("⏳ Đang nạp dữ liệu mới (data/raw)...")
    df_raw = pd.read_csv(raw_path, low_memory=False)
    print(f"✓ Nạp xong {filename} mới trong {time.time()-t0:.2f}s | Shape mới: {df_raw.shape}")

    df_old = None
    if old_path.exists():
        t0 = time.time()
        df_old = pd.read_csv(old_path, low_memory=False)
        print(f"✓ Nạp xong {filename} cũ trong {time.time()-t0:.2f}s | Shape cũ: {df_old.shape}")

    # 3. SO SÁNH SHAPE & SCHEMA
    rows_raw, cols_raw = df_raw.shape
    if df_old is not None:
        rows_old, cols_old = df_old.shape
        diff_rows = rows_raw - rows_old
        diff_cols = cols_raw - cols_old
        print(f"\n[1. CẤU TRÚC & QUY MÔ]")
        print(f"  - Số dòng: Cũ {rows_old:,} -> Mới {rows_raw:,} ({diff_rows:+,} dòng)")
        print(f"  - Số cột : Cũ {cols_old} -> Mới {cols_raw} ({diff_cols:+} cột)")

        cols_added = set(df_raw.columns) - set(df_old.columns)
        cols_removed = set(df_old.columns) - set(df_raw.columns)
        if cols_added:
            print(f"  🚨 CỘT MỚI XUẤT HIỆN ({len(cols_added)}): {list(cols_added)[:10]}")
        if cols_removed:
            print(f"  🚨 CỘT BỊ MẤT ({len(cols_removed)}): {list(cols_removed)[:10]}")

        # Kiểm tra biến đổi dtypes
        dtype_changes = []
        for c in set(df_old.columns) & set(df_raw.columns):
            dt_old = str(df_old[c].dtype)
            dt_raw = str(df_raw[c].dtype)
            if dt_old != dt_raw:
                dtype_changes.append((c, dt_old, dt_raw))
        if dtype_changes:
            print(f"\n  🚨 CÁC CỘT BỊ THAY ĐỔI DTYPE ({len(dtype_changes)} cột):")
            for c, dt_o, dt_d in dtype_changes[:15]:
                print(f"     * Cột '{c}': Cũ=[{dt_o}] -> Mới=[{dt_d}] (Dấu hiệu bị tiêm chuỗi bẩn!)")

    # 4. KIỂM TRA TRÙNG LẶP HOÀN TOÀN (DUPLICATE ROWS)
    dup_rows = int(df_raw.duplicated().sum())
    print(f"\n[2. TÍNH TOÀN VẸN & TRÙNG LẶP]")
    if dup_rows > 0:
        print(f"  🚨 PHÁT HIỆN: {dup_rows:,d} dòng TRÙNG LẶP HOÀN TOÀN (Duplicated Rows)!")
    else:
        print(f"  ✓ Không có dòng trùng lặp 100%.")

    # 5. KIỂM TRA KHÓA CHÍNH & KHÓA NGOẠI MỒ CÔI (KEY INTEGRITY)
    if "SK_ID_CURR" in df_raw.columns:
        null_curr = int(df_raw["SK_ID_CURR"].isna().sum())
        if null_curr > 0:
            print(f"  🚨 PHÁT HIỆN: {null_curr:,d} dòng có khóa SK_ID_CURR bị NULL!")
        else:
            print(f"  ✓ Khóa SK_ID_CURR không có giá trị Null.")

        if filename == "application_train.csv":
            dup_keys = int(df_raw["SK_ID_CURR"].duplicated().sum())
            if dup_keys > 0:
                print(f"  🚨 VI PHẠM GRAIN NGHIÊM TRỌNG: {dup_keys:,d} mã SK_ID_CURR BỊ TRÙNG LẶP TRONG BẢNG CHÍNH!")
            else:
                print(f"  ✓ Khóa chính SK_ID_CURR là DUY NHẤT (Bảo toàn 1 khách hàng = 1 dòng).")
            # Lưu lại tập ID của bảng chính
            MAIN_TRAIN_CURR_IDS = set(df_raw["SK_ID_CURR"].dropna().astype(int))
        else:
            # Kiểm tra Khóa ngoại mồ côi (Orphaned Foreign Keys) trên bảng phụ
            if MAIN_TRAIN_CURR_IDS is not None:
                sub_curr_ids = set(df_raw["SK_ID_CURR"].dropna().astype(int))
                orphaned_ids = sub_curr_ids - MAIN_TRAIN_CURR_IDS
                if orphaned_ids:
                    print(f"  🚨 BẪY KHÓA NGOẠI MỒ CÔI: Phát hiện {len(orphaned_ids):,d} mã SK_ID_CURR trong bảng phụ KHÔNG TỒN TẠI ở application_train.csv!")
                    print(f"     Ví dụ ID mồ côi: {list(orphaned_ids)[:5]}")
                else:
                    print(f"  ✓ Tính toàn vẹn liên kết: 100% mã SK_ID_CURR khớp với bảng chính.")

    if "SK_ID_PREV" in df_raw.columns:
        null_prev = int(df_raw["SK_ID_PREV"].isna().sum())
        if null_prev > 0:
            print(f"  🚨 PHÁT HIỆN: {null_prev:,d} dòng có SK_ID_PREV bị NULL!")

    # 6. KIỂM TRA CHUYÊN BIỆT NHÃN MỤC TIÊU TARGET (CHỈ CHO APPLICATION_TRAIN)
    if filename == "application_train.csv" and "TARGET" in df_raw.columns:
        print(f"\n[3. KIỂM TOÁN NHÃN MỤC TIÊU TARGET]")
        target_counts = df_raw["TARGET"].value_counts(dropna=False).to_dict()
        print(f"  • Phân bố giá trị TARGET (Mới): {target_counts}")
        
        # Kiểm tra nhãn dị thường ngoài 0 và 1
        valid_targets = {0, 1, 0.0, 1.0}
        invalid_targets = [v for v in target_counts.keys() if v not in valid_targets]
        if invalid_targets:
            print(f"  🚨 PHÁT HIỆN NHÃN DỊ BIỆT BẤT THƯỜNG: {invalid_targets} (Ví dụ nhãn 2, -1, hoặc NaN!)")
        
        # So sánh tỷ lệ vỡ nợ Default Rate
        default_rate_raw = (df_raw["TARGET"] == 1).mean() * 100
        if df_old is not None and "TARGET" in df_old.columns:
            default_rate_old = (df_old["TARGET"] == 1).mean() * 100
            print(f"  • Tỷ lệ vỡ nợ: Cũ = {default_rate_old:.2f}% | Mới = {default_rate_raw:.2f}% (Chênh lệch: {default_rate_raw - default_rate_old:+.2f}%)")

    # 7. QUÉT TÌM MISSING ẨN (HIDDEN SENTINEL VALUES)
    print(f"\n[4. QUÉT MISSING ẨN & KÝ TỰ RÁC DẠNG CHUỖI]")
    found_hidden_missing = {}
    str_cols = df_raw.select_dtypes(include=["object"]).columns.tolist()

    for col in str_cols:
        series_str = df_raw[col].dropna().astype(str).str.strip().str.lower()
        val_counts = series_str.value_counts()
        hit_patterns = {}
        for pattern in HIDDEN_MISSING_PATTERNS:
            if pattern in val_counts.index:
                cnt = int(val_counts[pattern])
                if cnt > 0:
                    hit_patterns[pattern] = cnt
        if hit_patterns:
            found_hidden_missing[col] = hit_patterns

    if found_hidden_missing:
        print(f"  🚨 PHÁT HIỆN MISSING ẨN TRÊN {len(found_hidden_missing)} CỘT:")
        for col, hits in list(found_hidden_missing.items())[:15]:
            print(f"     * Cột '{col}': {hits}")
    else:
        print(f"  ✓ Không phát hiện các ký tự missing kinh điển (?, N/A, -, null...).")

    # 8. QUÉT KHOẢNG TRẮNG DƯ THỪA (WHITESPACE POLLUTION)
    whitespace_cols = []
    for col in str_cols:
        series_orig = df_raw[col].dropna().astype(str)
        has_spaces = (series_orig.str.startswith(" ") | series_orig.str.endswith(" ")).sum()
        if has_spaces > 0:
            whitespace_cols.append((col, int(has_spaces)))

    if whitespace_cols:
        print(f"\n[5. NHIỄM BẨN KHOẢNG TRẮNG ĐẦU/CUỐI CHUỖI]")
        print(f"  🚨 PHÁT HIỆN {len(whitespace_cols)} CỘT CÓ KHOẢNG TRẮNG THỪA (LEADING/TRAILING SPACES):")
        for col, cnt in whitespace_cols[:15]:
            print(f"     * Cột '{col}': {cnt:,d} giá trị bị nhiễm khoảng trắng!")

    # 9. QUÉT BẪY ĐẢO DẤU BIẾN THỜI GIAN DAYS_* (> 0 BẤT THƯỜNG)
    days_cols = [c for c in df_raw.columns if c.startswith("DAYS_") and pd.api.types.is_numeric_dtype(df_raw[c])]
    days_positive_anomalies = []
    for col in days_cols:
        # DAYS_EMPLOYED có ngoại lệ hợp lệ là 365243 (hưu trí)
        if col == "DAYS_EMPLOYED":
            pos_mask = (df_raw[col] > 0) & (df_raw[col] != 365243)
        else:
            pos_mask = df_raw[col] > 0
        pos_cnt = int(pos_mask.sum())
        if pos_cnt > 0:
            max_val = float(df_raw.loc[pos_mask, col].max())
            days_positive_anomalies.append((col, pos_cnt, max_val))

    if days_positive_anomalies:
        print(f"\n[6. BẪY ĐẢO DẤU BIẾN THỜI GIAN (DAYS_* BỊ DƯƠNG HÓA VÔ LÝ)]")
        print(f"  🚨 PHÁT HIỆN {len(days_positive_anomalies)} CỘT THỜI GIAN CÓ GIÁ TRỊ DƯƠNG BẤT THƯỜNG:")
        for col, cnt, mval in days_positive_anomalies:
            print(f"     * Cột '{col}': {cnt:,d} dòng mang giá trị > 0 (Max: {mval}) -> Trái quy ước quá khứ âm!")

    # 10. QUÉT GIÁ TRỊ ÂM BẤT THƯỜNG TRÊN CÁC CỘT SỐ ĐO LƯỜNG
    print(f"\n[7. QUÉT GIÁ TRỊ ÂM VÔ LÝ & DỊ THƯỜNG SỐ HỌC]")
    num_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist()
    negative_anomalies = []
    for col in num_cols:
        if any(prefix in col for prefix in ["DAYS_", "MONTHS_"]):
            continue
        if "SK_ID" in col:
            continue
        neg_mask = df_raw[col] < 0
        neg_count = int(neg_mask.sum())
        if neg_count > 0:
            min_val = float(df_raw[col].min())
            negative_anomalies.append((col, neg_count, min_val))

    if negative_anomalies:
        print(f"  🚨 PHÁT HIỆN {len(negative_anomalies)} CỘT SỐ CÓ GIÁ TRỊ ÂM VÔ LÝ:")
        for col, cnt, mval in negative_anomalies[:15]:
            print(f"     * Cột '{col}': {cnt:,d} dòng mang giá trị âm (Min: {mval})")
    else:
        print(f"  ✓ Không phát hiện giá trị âm vô lý trên các biến số đo lường.")

    # 11. ĐỐI CHIẾU NGOẠI LAI CỰC ĐOAN (EXTREME OUTLIER SPIKES) TRÊN CÁC CỘT TRỌNG TÂM
    key_metrics = [
        "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE",
        "CNT_CHILDREN", "CNT_FAM_MEMBERS", "AMT_PAYMENT", "AMT_INSTALMENT", "AMT_BALANCE"
    ]
    outlier_spikes = []
    for col in key_metrics:
        if col in df_raw.columns and pd.api.types.is_numeric_dtype(df_raw[col]):
            raw_min = float(df_raw[col].min())
            raw_max = float(df_raw[col].max())
            raw_med = float(df_raw[col].median())
            if df_old is not None and col in df_old.columns and pd.api.types.is_numeric_dtype(df_old[col]):
                old_max = float(df_old[col].max())
                old_min = float(df_old[col].min())
                # Phát hiện nếu Max tăng vọt gấp 10 lần hoặc Min tụt về 0/âm
                if raw_max > old_max * 5 or raw_min < old_min:
                    outlier_spikes.append((col, old_min, old_max, raw_min, raw_max))

    if outlier_spikes:
        print(f"\n[8. BIẾN ĐỘNG NGOẠI LAI CỰC ĐOAN (OUTLIER SPIKES)]")
        print(f"  🚨 PHÁT HIỆN {len(outlier_spikes)} ĐẠI LƯỢNG BỊ BƠM GIÁ TRỊ ĐỘT BIẾN:")
        for col, o_min, o_max, r_min, r_max in outlier_spikes:
            print(f"     * Cột '{col}': Cũ=[{o_min:,.1f} ~ {o_max:,.1f}] -> Mới=[{r_min:,.1f} ~ {r_max:,.1f}]")

    # 12. SO SÁNH GIÁ TRỊ NULL GIỮA 2 TẬP
    if df_old is not None:
        common_cols = list(set(df_old.columns) & set(df_raw.columns))
        old_nulls = df_old[common_cols].isna().sum()
        raw_nulls = df_raw[common_cols].isna().sum()
        null_diff = (raw_nulls - old_nulls)
        significant_null_diff = null_diff[null_diff != 0]
        if len(significant_null_diff) > 0:
            print(f"\n[9. BIẾN ĐỘNG SỐ LƯỢNG NULL TRỰC TIẾP ({len(significant_null_diff)} cột)]")
            for c in significant_null_diff.abs().sort_values(ascending=False).index[:15]:
                print(f"     * Cột '{c}': Cũ={old_nulls[c]:,d} Null -> Mới={raw_nulls[c]:,d} Null (Chênh: {null_diff[c]:+,d})")

    # Dọn rác giải phóng RAM
    del df_raw
    if df_old is not None:
        del df_old
    gc.collect()


def main():
    print("=" * 95)
    print("HỆ THỐNG KIỂM TOÁN TẬP DỮ LIỆU MỚI TẠI FOLDER: data/raw")
    print("ĐỐI CHIẾU TOÀN DIỆN VỚI TẬP DỮ LIỆU CŨ: data/old_data")
    print("=" * 95)

    for fname in FILES:
        audit_single_file(fname)

    print("\n" + "=" * 95)
    print("✓ HOÀN TẤT KIỂM TOÁN TOÀN BỘ 6 TẬP TIN!")
    print("=" * 95)


if __name__ == "__main__":
    main()
