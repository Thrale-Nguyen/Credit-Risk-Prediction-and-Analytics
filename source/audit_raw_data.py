"""
Script: audit_raw_data.py
Mục đích: Kiểm toán chuyên sâu (Forensic Data Audit) so sánh tập dữ liệu mới (data/raw)
         với tập dữ liệu chuẩn ban đầu (Dataset - Home Credit Default Risk).
         Bóc tách toàn bộ các bẫy làm dơ dữ liệu do giảng viên cài cắm.
"""

import sys
import io
import gc
import time
from pathlib import Path
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

HIDDEN_MISSING_PATTERNS = {
    "?", "n/a", "na", "nan", "null", "none", "missing", "unknown", 
    "-999", "-9999", "999999", "", " ", "xna", "xap"
}

def audit_single_file(filename: str):
    print("\n" + "=" * 90)
    print(f"🔍 BẮT ĐẦU KIỂM TOÁN TẬP TIN: {filename}")
    print("=" * 90)

    old_path = OLD_DIR / filename
    raw_path = RAW_DIR / filename

    if not raw_path.exists():
        print(f"❌ Không tìm thấy file trong data/raw: {raw_path}")
        return

    # 1. KÍCH THƯỚC FILE TRÊN ĐĨA
    old_size_mb = old_path.stat().st_size / (1024 * 1024) if old_path.exists() else 0
    raw_size_mb = raw_path.stat().st_size / (1024 * 1024)
    print(f"• Dung lượng đĩa: Gốc = {old_size_mb:.2f} MB | Mới = {raw_size_mb:.2f} MB (Chênh lệch: {raw_size_mb - old_size_mb:+.2f} MB)")

    # 2. ĐỌC DỮ LIỆU
    t0 = time.time()
    print("⏳ Đang nạp dữ liệu (data/raw)...")
    # Đọc raw dạng string / object cho các cột nhạy cảm để không bị pandas tự ép kiểu sai
    df_raw = pd.read_csv(raw_path, low_memory=False)
    print(f"✓ Nạp xong {filename} mới trong {time.time()-t0:.2f}s | Shape mới: {df_raw.shape}")

    df_old = None
    if old_path.exists():
        t0 = time.time()
        # Đọc old
        df_old = pd.read_csv(old_path, low_memory=False)
        print(f"✓ Nạp xong {filename} gốc trong {time.time()-t0:.2f}s | Shape gốc: {df_old.shape}")

    # 3. SO SÁNH SHAPE & SCHEMA
    rows_raw, cols_raw = df_raw.shape
    if df_old is not None:
        rows_old, cols_old = df_old.shape
        diff_rows = rows_raw - rows_old
        diff_cols = cols_raw - cols_old
        print(f"\n[1. CẤU TRÚC & QUY MÔ]")
        print(f"  - Số dòng: Gốc {rows_old:,} -> Mới {rows_raw:,} ({diff_rows:+,} dòng)")
        print(f"  - Số cột : Gốc {cols_old} -> Mới {cols_raw} ({diff_cols:+} cột)")

        cols_added = set(df_raw.columns) - set(df_old.columns)
        cols_removed = set(df_old.columns) - set(df_raw.columns)
        if cols_added:
            print(f"  ⚠️ Cột MỚI xuất hiện ({len(cols_added)}): {list(cols_added)[:10]}")
        if cols_removed:
            print(f"  ⚠️ Cột BỊ MẤT ({len(cols_removed)}): {list(cols_removed)[:10]}")

        # Kiểm tra biến đổi dtypes
        dtype_changes = []
        for c in set(df_old.columns) & set(df_raw.columns):
            dt_old = str(df_old[c].dtype)
            dt_raw = str(df_raw[c].dtype)
            if dt_old != dt_raw:
                dtype_changes.append((c, dt_old, dt_raw))
        if dtype_changes:
            print(f"\n  ⚠️ CÁC CỘT BỊ THAY ĐỔI DTYPE ({len(dtype_changes)} cột):")
            for c, dt_c, dt_d in dtype_changes[:15]:
                print(f"     * Cột '{c}': Gốc=[{dt_c}] -> Mới=[{dt_d}] (Dấu hiệu bị tiêm ký tự lạ!)")

    # 4. KIỂM TRA TRÙNG LẶP HOÀN TOÀN (DUPLICATE ROWS)
    dup_rows = int(df_raw.duplicated().sum())
    print(f"\n[2. TÍNH TOÀN VẸN & TRÙNG LẶP]")
    if dup_rows > 0:
        print(f"  🚨 PHÁT HIỆN: {dup_rows:,d} dòng TRÙNG LẶP HOÀN TOÀN (Duplicated Rows)!")
    else:
        print(f"  ✓ Không có dòng trùng lặp 100%.")

    # 5. KIỂM TRA KHÓA CHÍNH (KEY INTEGRITY)
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

    if "SK_ID_PREV" in df_raw.columns:
        null_prev = int(df_raw["SK_ID_PREV"].isna().sum())
        if null_prev > 0:
            print(f"  🚨 PHÁT HIỆN: {null_prev:,d} dòng có SK_ID_PREV bị NULL!")

    # 6. QUÉT TÌM MISSING ẨN (HIDDEN SENTINEL VALUES)
    print(f"\n[3. QUÉT MISSING ẨN & KÝ TỰ RÁC DẠNG CHUỖI]")
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
        print(f"  ✓ Không phát hiện các ký tự missing kinh điển (?, N/A, null...).")

    # 7. QUÉT KHOẢNG TRẮNG DƯ THỪA (WHITESPACE POLLUTION)
    whitespace_cols = []
    for col in str_cols:
        series_orig = df_raw[col].dropna().astype(str)
        has_spaces = (series_orig.str.startswith(" ") | series_orig.str.endswith(" ")).sum()
        if has_spaces > 0:
            whitespace_cols.append((col, int(has_spaces)))

    if whitespace_cols:
        print(f"\n[4. NHIỄM BẨN KHOẢNG TRẮNG ĐẦU/CUỐI CHUỖI]")
        print(f"  🚨 PHÁT HIỆN {len(whitespace_cols)} CỘT CÓ KHOẢNG TRẮNG THỪA (LEADING/TRAILING SPACES):")
        for col, cnt in whitespace_cols[:15]:
            print(f"     * Cột '{col}': {cnt:,d} giá trị bị nhiễm khoảng trắng!")

    # 8. QUÉT GIÁ TRỊ ÂM BẤT THƯỜNG TRÊN CÁC CỘT SỐ
    print(f"\n[5. QUÉT GIÁ TRỊ ÂM VÔ LÝ & DỊ THƯỜNG SỐ HỌC]")
    num_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist()
    negative_anomalies = []
    for col in num_cols:
        if any(prefix in col for prefix in ["DAYS_", "MONTHS_"]):
            continue  # Bỏ qua các cột ngày tháng âm hợp lệ
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

    # 9. SO SÁNH GIÁ TRỊ NULL GIỮA 2 TẬP
    if df_old is not None:
        common_cols = list(set(df_old.columns) & set(df_raw.columns))
        old_nulls = df_old[common_cols].isna().sum()
        raw_nulls = df_raw[common_cols].isna().sum()
        null_diff = (raw_nulls - old_nulls)
        significant_null_diff = null_diff[null_diff != 0]
        if len(significant_null_diff) > 0:
            print(f"\n[6. BIẾN ĐỘNG SỐ LƯỢNG NULL TRỰC TIẾP ({len(significant_null_diff)} cột)]")
            for c in significant_null_diff.abs().sort_values(ascending=False).index[:15]:
                print(f"     * Cột '{c}': Gốc={old_nulls[c]:,d} Null -> Mới={raw_nulls[c]:,d} Null (Chênh: {null_diff[c]:+,d})")

    # Dọn rác giải phóng RAM
    del df_raw
    if df_old is not None:
        del df_old
    gc.collect()

def main():
    print("=" * 90)
    print("HỆ THỐNG KIỂM TOÁN TẬP DỮ LIỆU MỚI TẠI FOLDER: data/raw")
    print("ĐỐI CHIẾU VỚI TẬP DỮ LIỆU GỐC: Dataset - Home Credit Default Risk")
    print("=" * 90)

    for fname in FILES:
        audit_single_file(fname)

    print("\n" + "=" * 90)
    print("✓ HOÀN TẤT KIỂM TOÁN TOÀN BỘ 6 TẬP TIN!")
    print("=" * 90)

if __name__ == "__main__":
    main()


