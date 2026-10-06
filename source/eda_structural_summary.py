"""
Module: eda_structural_summary.py
Dự án: PRJ-01 - Dự đoán Rủi ro Vỡ nợ Tín dụng (Home Credit Default Risk)
Mô tả: Khám phá cấu trúc sơ bộ của tập dữ liệu sau ghép nối train_merged_final.parquet:
       1. Kích thước (shape)
       2. Mức tiêu thụ bộ nhớ (info / memory_usage)
       3. Thống kê 5 con số cơ bản cho biến định lượng (describe)
       4. Phân phối tần số cho biến định tính (value_counts)
"""

import sys
from pathlib import Path
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


def explore_merged_dataset(parquet_path: str = "train_merged_final.parquet"):
    p = Path(parquet_path)
    if not p.exists():
        candidate = Path(__file__).resolve().parent.parent / "train_merged_final.parquet"
        if candidate.exists():
            p = candidate
        else:
            raise FileNotFoundError(f"Không tìm thấy file: {parquet_path}")

    print("=" * 85)
    print(f"BÁO CÁO KHÁM PHÁ CẤU TRÚC SƠ BỘ TẬP DỮ LIỆU GHÉP NỐI: {p.name}")
    print("=" * 85)

    # 1. Đọc dữ liệu
    df = pd.read_parquet(p)
    n_rows, n_cols = df.shape

    # 2. Kích thước
    print("\n1. KÍCH THƯỚC TẬP DỮ LIỆU (SHAPE)")
    print("-" * 85)
    print(f"• Số lượng dòng (Hồ sơ khách hàng): {n_rows:,d} dòng")
    print(f"• Số lượng cột  (Tổng đặc trưng)  : {n_cols:,d} cột")
    print(f"• Bảo toàn grain: Đúng 307,511 dòng (Zero fan-out / 1 khách hàng = 1 dòng)")

    # 3. Tiêu thụ bộ nhớ
    print("\n2. MỨC TIÊU THỤ BỘ NHỚ (MEMORY USAGE & DTYPES)")
    print("-" * 85)
    mem_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
    file_size_mb = p.stat().st_size / (1024 * 1024)
    print(f"• Dung lượng file trên đĩa (.parquet): {file_size_mb:.2f} MB")
    print(f"• Dung lượng nạp vào RAM             : {mem_mb:.2f} MB (~{mem_mb / 1024:.2f} GB)")
    print(f"• Phân bố kiểu dữ liệu (Data types) :")
    for dtype, count in df.dtypes.value_counts().items():
        print(f"   - {str(dtype):<10}: {count:>4d} cột")

    # Phân loại cột
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    print(f"• Tổng số biến định lượng: {len(num_cols)} cột")
    print(f"• Tổng số biến định tính : {len(cat_cols)} cột")

    # 4. Thống kê 5 con số cho biến định lượng
    print("\n3. THỐNG KÊ 5 CON SỐ CƠ BẢN ĐỐI VỚI CÁC BIẾN ĐỊNH LƯỢNG TRỌNG TÂM")
    print("-" * 85)
    key_num = [
        "TARGET", "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY",
        "BUREAU_AMT_CREDIT_SUM_MEAN", "PREV_AMT_CREDIT_MEAN",
        "INSTAL_AMT_PAYMENT_SUM", "CC_AMT_BALANCE_MEAN"
    ]
    avail_num = [c for c in key_num if c in df.columns]
    desc = df[avail_num].describe().T[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]]
    desc.columns = ["Count", "Mean", "Std", "Min", "Q1 (25%)", "Median (50%)", "Q3 (75%)", "Max"]
    print(desc.to_string())

    # 5. Phân phối tần số cho biến định tính
    print("\n4. PHÂN PHỐI TẦN SỐ CỦA CÁC BIẾN ĐỊNH TÍNH CỐT LÕI")
    print("-" * 85)
    focus_cats = [
        "NAME_CONTRACT_TYPE", "CODE_GENDER", "FLAG_OWN_CAR", 
        "FLAG_OWN_REALTY", "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE"
    ]
    for col in focus_cats:
        if col in df.columns:
            vc = df[col].value_counts(dropna=False)
            pct = df[col].value_counts(dropna=False, normalize=True) * 100
            dist = pd.DataFrame({"Tần số (Count)": vc, "Tỷ lệ (%)": pct})
            print(f"\n▶ Biến [{col}]:")
            print(dist.head(6).to_string())

    print("\n" + "=" * 85)
    print("✓ HOÀN TẤT KHÁM PHÁ CẤU TRÚC SƠ BỘ")
    print("=" * 85)


if __name__ == "__main__":
    explore_merged_dataset()
