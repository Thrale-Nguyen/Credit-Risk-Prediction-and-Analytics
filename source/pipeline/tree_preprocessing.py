"""
Module: tree_preprocessing.py
Package: source.pipeline
Dự án: PRJ-01 - Dự đoán Rủi ro Vỡ nợ Tín dụng (Home Credit Default Risk)
Mô tả: [Bước 3 trong Pipeline Tuyến tính]
       Tiền xử lý dữ liệu chuyên biệt tối ưu cho LightGBM:
       1. Xử lý dị thường nghiệp vụ:
          - Dị thường hưu trí DAYS_EMPLOYED == 365243 -> Tạo cờ DAYS_EMPLOYED_ANOM = 1 và gán np.nan.
          - Cờ vắng mặt cấu trúc thẻ tín dụng: NO_CC = 1 cho khách hàng chưa từng dùng thẻ.
          - Dương hóa các biến thời gian âm DAYS_* và tính tuổi thực tế AGE_YEARS.
          - Bổ sung 4 tỷ lệ đòn bẩy tài chính cốt lõi (CREDIT_TO_INCOME, ANNUITY_TO_INCOME/DTI,
            CREDIT_TERM, INCOME_PER_PERSON).
       2. Tối ưu bộ nhớ RAM (Safe Dynamic Downcasting sang float32/int32/int8).
       3. Chuẩn hóa 16 biến phân loại sang pd.Categorical phục vụ thuật toán phân nhánh tối ưu của LightGBM.
       4. Bảo toàn giá trị thiếu tự nhiên (Native NaN) theo đúng thiết kế của LightGBM.
       5. Phân chia Stratified Train/Val Split (80/20) giữ nguyên tỷ lệ nợ xấu 7.89% trên cả 2 tập.
       6. Xuất bản:
          - data/processed/train_lgb.parquet
          - data/processed/val_lgb.parquet
          - data/processed/pipeline_metadata.json
          - Hàm cấp cao: get_lightgbm_data()
"""

import os
import gc
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Thiết lập encoding UTF-8 cho Windows console
if sys.platform.startswith("win"):
    try:
        if sys.stdout and hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if sys.stderr and hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


# =============================================================================
# CẤU HÌNH BIẾN PHÂN LOẠI & DỊ THƯỜNG
# =============================================================================

DEFAULT_GROUP_KEY = "SK_ID_CURR"
DEFAULT_TARGET = "TARGET"

# 16 biến phân loại gốc trong bảng application_train
CATEGORICAL_FEATURES = [
    "NAME_CONTRACT_TYPE",
    "CODE_GENDER",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
    "NAME_TYPE_SUITE",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "OCCUPATION_TYPE",
    "WEEKDAY_APPR_PROCESS_START",
    "ORGANIZATION_TYPE",
    "FONDKAPREMONT_MODE",
    "HOUSETYPE_MODE",
    "WALLSMATERIAL_MODE",
    "EMERGENCYSTATE_MODE",
]


def resolve_paths(
    merged_parquet: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None
) -> Tuple[Path, Path]:
    """Xác định đường dẫn file train_merged.parquet đầu vào và thư mục lưu kết quả."""
    if merged_parquet is None:
        candidates = [
            Path("data/processed/train_merged.parquet"),
            Path("../data/processed/train_merged.parquet"),
            Path("../../data/processed/train_merged.parquet"),
        ]
        found_file = None
        for c in candidates:
            if c.exists() and c.is_file():
                found_file = c.resolve()
                break
        if found_file is None:
            raise FileNotFoundError(
                "Không tìm thấy file 'train_merged.parquet'. "
                "Vui lòng chạy Bước 2 (02_merge_pipeline.py) trước!"
            )
        in_file = found_file
    else:
        in_file = Path(merged_parquet).resolve()

    if output_dir is None:
        out_dir = in_file.parent
    else:
        out_dir = Path(output_dir).resolve()

    out_dir.mkdir(parents=True, exist_ok=True)
    return in_file, out_dir


# =============================================================================
# HÀM XỬ LÝ ĐẶC TRƯNG NGHIỆP VỤ (BUSINESS FEATURE ENGINEERING)
# =============================================================================

def apply_business_features(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Xử lý dị thường nghiệp vụ và tạo các chỉ số đòn bẩy tài chính quan trọng:
    1. Cờ dị thường hưu trí DAYS_EMPLOYED_ANOM và gán np.nan cho 365243.
    2. Cờ vắng mặt cấu trúc NO_CC.
    3. Tính tuổi thực tế AGE_YEARS và dương hóa các cột ngày âm DAYS_*.
    4. 4 tỷ lệ đòn bẩy tài chính cốt lõi (CREDIT_TO_INCOME, ANNUITY_TO_INCOME, CREDIT_TERM, INCOME_PER_PERSON).
    """
    t0 = time.time()
    if verbose:
        print("  • Đang thực hiện Feature Engineering & Xử lý Dị thường Nghiệp vụ...")

    # 0. Khử số âm vô lý trên các thước đo dương (AMT_*, CNT_*) do bẫy đảo dấu
    amt_cols = [c for c in df.columns if c.startswith("AMT_")]
    for c in amt_cols:
        if (df[c] < 0).any():
            df[c] = np.abs(df[c])

    cnt_cols = [c for c in df.columns if c.startswith("CNT_")]
    for c in cnt_cols:
        if (df[c] < 0).any():
            df[c] = np.abs(df[c])

    # Xử lý ngoại lai lính canh: CNT_CHILDREN >= 15 hoặc 9999 gán np.nan
    if "CNT_CHILDREN" in df.columns:
        df.loc[df["CNT_CHILDREN"] >= 15, "CNT_CHILDREN"] = np.nan

    # 1. Dị thường hưu trí DAYS_EMPLOYED == 365243
    if "DAYS_EMPLOYED" in df.columns:
        anom_mask = (df["DAYS_EMPLOYED"] == 365243)
        anom_count = int(anom_mask.sum())
        df["DAYS_EMPLOYED_ANOM"] = anom_mask.astype(np.int8)
        df.loc[anom_mask, "DAYS_EMPLOYED"] = np.nan
        if verbose:
            print(f"    - Bắt dị thường hưu trí: Đã tạo cờ [DAYS_EMPLOYED_ANOM] ({anom_count:,} bản ghi = 1) và gán DAYS_EMPLOYED -> NaN")

    # 2. Cờ vắng mặt cấu trúc thẻ tín dụng (NO_CC)
    # Khách hàng không có thông tin chi tiêu thẻ tín dụng
    if "CC_AMT_BALANCE_MEAN" in df.columns:
        no_cc_mask = df["CC_AMT_BALANCE_MEAN"].isna()
        no_cc_count = int(no_cc_mask.sum())
        df["NO_CC"] = no_cc_mask.astype(np.int8)
        if verbose:
            print(f"    - Tạo cờ vắng mặt thẻ [NO_CC]: {no_cc_count:,} khách hàng ({no_cc_count/len(df)*100:.1f}%) chưa từng dùng thẻ tín dụng")
    else:
        df["NO_CC"] = np.int8(0)

    # 3. Tính tuổi thực tế AGE_YEARS và dương hóa các cột ngày tháng âm DAYS_*
    if "DAYS_BIRTH" in df.columns:
        df["AGE_YEARS"] = (np.abs(df["DAYS_BIRTH"]) / 365.25).astype("float32")
        if verbose:
            print("    - Đã tạo đặc trưng độ tuổi thực tế: [AGE_YEARS]")

    # Dương hóa các cột ngày âm (DAYS_*)
    days_cols = [c for c in df.columns if c.startswith("DAYS_") and c != "DAYS_EMPLOYED_ANOM"]
    for c in days_cols:
        # Nếu cột có giá trị âm thì lấy giá trị tuyệt đối
        if (df[c] < 0).any():
            df[c] = np.abs(df[c])
    if verbose and days_cols:
        print(f"    - Đã dương hóa {len(days_cols)} cột biến thời gian [DAYS_*]")

    # 4. Bổ sung 4 Tỷ Lệ Đòn Bẩy Tài Chính Cốt Lõi (Domain Financial Ratios)
    if "AMT_CREDIT" in df.columns and "AMT_INCOME_TOTAL" in df.columns:
        income_safe = df["AMT_INCOME_TOTAL"].clip(lower=1.0)
        df["CREDIT_TO_INCOME"] = (df["AMT_CREDIT"] / income_safe).astype("float32")
        if verbose:
            print("    - Đã tạo tỷ lệ đòn bẩy: [CREDIT_TO_INCOME] (Khoản vay / Thu nhập)")

    if "AMT_ANNUITY" in df.columns and "AMT_INCOME_TOTAL" in df.columns:
        income_safe = df["AMT_INCOME_TOTAL"].clip(lower=1.0)
        df["ANNUITY_TO_INCOME"] = (df["AMT_ANNUITY"] / income_safe).astype("float32")
        if verbose:
            print("    - Đã tạo tỷ lệ gánh nặng nợ (DTI): [ANNUITY_TO_INCOME] (Khoản trả định kỳ / Thu nhập)")

    if "AMT_CREDIT" in df.columns and "AMT_ANNUITY" in df.columns:
        annuity_safe = df["AMT_ANNUITY"].clip(lower=1.0)
        df["CREDIT_TERM"] = (df["AMT_CREDIT"] / annuity_safe).astype("float32")
        if verbose:
            print("    - Đã tạo kỳ hạn vay ước tính: [CREDIT_TERM] (Tổng khoản vay / Khoản trả mỗi kỳ)")

    if "AMT_INCOME_TOTAL" in df.columns and "CNT_FAM_MEMBERS" in df.columns:
        fam_safe = df["CNT_FAM_MEMBERS"].fillna(1.0).clip(lower=1.0)
        df["INCOME_PER_PERSON"] = (df["AMT_INCOME_TOTAL"] / fam_safe).astype("float32")
        if verbose:
            print("    - Đã tạo thu nhập bình quân: [INCOME_PER_PERSON] (Thu nhập / Số thành viên)")

    elapsed = time.time() - t0
    if verbose:
        print(f"  ✓ Hoàn tất Feature Engineering trong {elapsed:.1f}s")

    return df


# =============================================================================
# HÀM CHUẨN HÓA CATEGORICAL & TỐI ƯU BỘ NHỚ RAM CHO LIGHTGBM
# =============================================================================

def prepare_lightgbm_features(df: pd.DataFrame, verbose: bool = True) -> Tuple[pd.DataFrame, List[str]]:
    """
    Chuẩn hóa đặc trưng chuyên biệt cho LightGBM:
    - 16 biến phân loại gốc được ép sang kiểu pd.Categorical.
    - Bảo toàn giá trị thiếu tự nhiên (Native NaN).
    - Downcast toàn bộ float64 -> float32 và int64 -> int32/int8.
    """
    t0 = time.time()
    actual_cat_cols = []

    # 1. Ép 16 biến phân loại sang pd.Categorical
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            # Chuyển sang category của pandas, giữ nguyên NaN tự nhiên
            df[col] = df[col].astype("category")
            actual_cat_cols.append(col)

    # 2. Dynamic Memory Downcasting
    # float64 -> float32
    float_cols = df.select_dtypes(include=["float64"]).columns
    for c in float_cols:
        df[c] = df[c].astype("float32")

    # int64 -> int32 (trừ khóa chính SK_ID_CURR giữ nguyên định danh)
    int_cols = df.select_dtypes(include=["int64"]).columns
    for c in int_cols:
        if c != DEFAULT_GROUP_KEY:
            # Nếu min/max nhỏ có thể ép int8/int16
            c_min, c_max = df[c].min(), df[c].max()
            if c_min >= -128 and c_max <= 127:
                df[c] = df[c].astype(np.int8)
            elif c_min >= -32768 and c_max <= 32767:
                df[c] = df[c].astype(np.int16)
            else:
                df[c] = df[c].astype(np.int32)

    elapsed = time.time() - t0
    mem_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)

    if verbose:
        print(f"  ✓ Đã chuẩn hóa {len(actual_cat_cols)} biến sang pd.Categorical cho LightGBM")
        print(f"  ✓ Safe Dynamic Downcasting hoàn tất trong {elapsed:.1f}s | RAM tiêu thụ: {mem_mb:.2f} MB (~{mem_mb/1024:.2f} GB)")

    return df, actual_cat_cols


# =============================================================================
# HÀM PHÂN CHIA STRATIFIED TRAIN / VAL SPLIT (80 / 20)
# =============================================================================

def split_stratified_train_val(
    df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42,
    verbose: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Phân chia tập dữ liệu thành Train và Validation theo tỷ lệ 80/20 có phân tầng (Stratified):
    - Bảo toàn chính xác tỷ lệ vỡ nợ (Default Rate = 7.89%) trên cả 2 tập.
    - Loại bỏ nguy cơ lệch phân phối hay rò rỉ dữ liệu.
    """
    assert DEFAULT_TARGET in df.columns, f"Không tìm thấy cột nhãn mục tiêu '{DEFAULT_TARGET}'!"

    y = df[DEFAULT_TARGET].astype(int)
    total_len = len(df)
    orig_rate = (y == 1).mean() * 100

    # Phân chia phân tầng theo y
    train_df, val_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    train_rate = (train_df[DEFAULT_TARGET] == 1).mean() * 100
    val_rate = (val_df[DEFAULT_TARGET] == 1).mean() * 100

    split_stats = {
        "total_rows": total_len,
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "train_pct": len(train_df) / total_len * 100,
        "val_pct": len(val_df) / total_len * 100,
        "overall_default_rate": float(orig_rate),
        "train_default_rate": float(train_rate),
        "val_default_rate": float(val_rate),
        "random_state": random_state,
    }

    if verbose:
        print("=" * 70)
        print("KẾT QUẢ PHÂN CHIA STRATIFIED TRAIN / VAL SPLIT (80/20):")
        print("=" * 70)
        print(f"• Toàn bộ tập dữ liệu : {total_len:,} dòng | Nợ xấu: {orig_rate:.2f}%")
        print(f"• Tập Train (80%)     : {len(train_df):,} dòng | Nợ xấu: {train_rate:.2f}% (Bảo toàn chuẩn 100%)")
        print(f"• Tập Validation (20%): {len(val_df):,} dòng | Nợ xấu: {val_rate:.2f}% (Bảo toàn chuẩn 100%)")
        print("=" * 70)

    return train_df, val_df, split_stats


# =============================================================================
# HÀM ĐIỀU PHỐI CHÍNH BƯỚC 3 (MAIN RUNNER)
# =============================================================================

def run_tree_preprocessing(
    merged_parquet: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    test_size: float = 0.20,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Thực thi trọn gói kịch bản Bước 3: Tiền xử lý dữ liệu cho LightGBM:
    1. Đọc train_merged.parquet (234,072 dòng x 872 cột).
    2. Bổ sung các đặc trưng nghiệp vụ: cờ dị thường hưu trí, cờ NO_CC, tuổi thực tế, 4 tỷ lệ đòn bẩy.
    3. Ép 16 biến phân loại sang pd.Categorical và downcast RAM.
    4. Stratified Train/Val Split (80/20) bảo toàn 7.89% nợ xấu.
    5. Xuất train_lgb.parquet, val_lgb.parquet và pipeline_metadata.json.
    """
    t_start = time.time()
    in_file, out_dir = resolve_paths(merged_parquet, output_dir)

    print("=" * 85)
    print("KHỞI CHẠY BƯỚC 3: TIỀN XỬ LÝ CHUYÊN BIỆT CHO MÔ HÌNH CÂY LIGHTGBM")
    print("=" * 85)
    print(f"• Nguồn ma trận đặc trưng (Input): {in_file}")
    print(f"• Thư mục xuất bản dữ liệu (Output): {out_dir}")
    print(f"• Chiến lược phân chia           : Stratified Train/Val Split ({int((1-test_size)*100)}/{int(test_size*100)})")
    print("-" * 85)

    # 1. Đọc train_merged.parquet
    t0 = time.time()
    print(f"⏳ [1/4] Đang nạp ma trận đặc trưng: {in_file.name}...")
    df = pd.read_parquet(in_file, engine="pyarrow")
    print(f"✓ Nạp xong ma trận trong {time.time()-t0:.1f}s | Shape: {df.shape}")

    # 2. Xử lý đặc trưng nghiệp vụ & Dị thường
    print("\n⏳ [2/4] Xử lý dị thường nghiệp vụ & Tạo tỷ lệ tài chính...")
    df = apply_business_features(df, verbose=True)

    # 3. Chuẩn hóa pd.Categorical & Downcast RAM cho LightGBM
    print("\n⏳ [3/4] Chuẩn hóa biến Categorical và Downcast RAM cho LightGBM...")
    df, actual_cat_cols = prepare_lightgbm_features(df, verbose=True)

    # 4. Phân chia Stratified Train / Val (80/20)
    print("\n⏳ [4/4] Thực hiện phân chia phân tầng Stratified 80/20...")
    train_df, val_df, split_stats = split_stratified_train_val(
        df,
        test_size=test_size,
        random_state=random_state,
        verbose=True
    )

    # 5. Xuất các file Parquet sẵn sàng huấn luyện
    print("\n" + "=" * 85)
    print("⏳ ĐANG XUẤT CÁC TẬP DỮ LIỆU LIGHTGBM RA ĐĨA...")
    train_out = out_dir / "train_lgb.parquet"
    val_out = out_dir / "val_lgb.parquet"
    meta_out = out_dir / "pipeline_metadata.json"

    train_df.to_parquet(train_out, index=False, engine="pyarrow")
    val_df.to_parquet(val_out, index=False, engine="pyarrow")

    train_size_mb = train_out.stat().st_size / (1024 * 1024)
    val_size_mb = val_out.stat().st_size / (1024 * 1024)

    # 6. Ghi file Metadata chẩn đoán cấu trúc
    metadata = {
        "status": "success",
        "model_target": "LightGBM",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_file": str(in_file),
        "total_rows": len(df),
        "total_features": df.shape[1] - 2,  # Trừ SK_ID_CURR và TARGET
        "total_columns": df.shape[1],
        "split_stats": split_stats,
        "categorical_features": actual_cat_cols,
        "engineered_features": [
            "DAYS_EMPLOYED_ANOM",
            "NO_CC",
            "AGE_YEARS",
            "CREDIT_TO_INCOME",
            "ANNUITY_TO_INCOME",
            "CREDIT_TERM",
            "INCOME_PER_PERSON",
        ],
        "files": {
            "train_parquet": {
                "path": str(train_out),
                "rows": len(train_df),
                "size_mb": round(train_size_mb, 2),
            },
            "val_parquet": {
                "path": str(val_out),
                "rows": len(val_df),
                "size_mb": round(val_size_mb, 2),
            },
        },
        "elapsed_seconds": round(time.time() - t_start, 2),
    }

    with open(meta_out, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    total_elapsed = time.time() - t_start
    print("=" * 85)
    print("✓ HOÀN TẤT THÀNH CÔNG BƯỚC 3: DỮ LIỆU SẴN SÀNG HUẤN LUYỆN LIGHTGBM")
    print("=" * 85)
    print(f"• Tập Train Parquet   : {train_out} ({train_size_mb:.2f} MB | {len(train_df):,} dòng)")
    print(f"• Tập Val Parquet     : {val_out} ({val_size_mb:.2f} MB | {len(val_df):,} dòng)")
    print(f"• File Metadata JSON  : {meta_out}")
    print(f"• Tổng số cột         : {df.shape[1]} cột (Gồm 16 cột pd.Categorical và {df.shape[1]-18} cột số float32)")
    print(f"• Tổng thời gian      : {total_elapsed:.1f} giây ({total_elapsed/60:.2f} phút)")
    print("=" * 85)

    del df, train_df, val_df
    gc.collect()

    return metadata


# =============================================================================
# HÀM CẤP CAO ĐỌC DỮ LIỆU NẠP THẲNG VÀO LIGHTGBM (API CHO GIAI ĐOẠN 5)
# =============================================================================

def get_lightgbm_data(
    data_dir: Optional[Union[str, Path]] = None,
    return_keys: bool = False
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, List[str]]:
    """
    Hàm giao diện tiện ích phục vụ trực tiếp cho Giai đoạn huấn luyện LightGBM:
    Đọc trực tiếp từ train_lgb.parquet và val_lgb.parquet đã tạo ở Bước 3.
    
    Returns:
    --------
    X_train : pd.DataFrame (Đặc trưng huấn luyện, đã ép pd.Categorical)
    y_train : pd.Series (Nhãn mục tiêu 0/1)
    X_val   : pd.DataFrame (Đặc trưng kiểm định)
    y_val   : pd.Series (Nhãn mục tiêu kiểm định)
    cat_cols: List[str] (Danh sách 16 tên cột phân loại để truyền vào lgb.train)
    """
    if data_dir is None:
        data_dir = Path("data/processed")
    else:
        data_dir = Path(data_dir)

    train_file = data_dir / "train_lgb.parquet"
    val_file = data_dir / "val_lgb.parquet"

    if not train_file.exists() or not val_file.exists():
        # Nếu chưa có file phân chia, tự động chạy Bước 3
        print("Chưa tìm thấy train_lgb.parquet / val_lgb.parquet, đang tự động chạy Bước 3...")
        run_tree_preprocessing(output_dir=data_dir)

    train_df = pd.read_parquet(train_file, engine="pyarrow")
    val_df = pd.read_parquet(val_file, engine="pyarrow")

    # Đảm bảo 16 biến phân loại giữ đúng kiểu category sau khi đọc từ parquet
    actual_cats = []
    for c in CATEGORICAL_FEATURES:
        if c in train_df.columns:
            train_df[c] = train_df[c].astype("category")
            val_df[c] = val_df[c].astype("category")
            actual_cats.append(c)

    # Tách X và y
    drop_cols = [DEFAULT_TARGET]
    if not return_keys and DEFAULT_GROUP_KEY in train_df.columns:
        drop_cols.append(DEFAULT_GROUP_KEY)

    X_train = train_df.drop(columns=drop_cols)
    y_train = train_df[DEFAULT_TARGET].astype(int)

    X_val = val_df.drop(columns=drop_cols)
    y_val = val_df[DEFAULT_TARGET].astype(int)

    return X_train, y_train, X_val, y_val, actual_cats


if __name__ == "__main__":
    run_tree_preprocessing()
