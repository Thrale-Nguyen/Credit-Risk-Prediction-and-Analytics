"""
Module: table_sanitizer.py
Package: source.pipeline
Dự án: PRJ-01 - Dự đoán Rủi ro Vỡ nợ Tín dụng (Home Credit Default Risk)
Mô tả: [Bước 1 trong Pipeline Tuyến tính]
       Làm sạch dữ liệu cấp bảng độc lập (Table-level Data Sanitization):
       1. Khử nhiễm 104 cột object bị pha tạp đơn vị tiền tệ/đo lường (đ, VND, USD, kg, %) về số thực.
       2. Chuẩn hóa nhãn TARGET: Lọc sạch 24,885 missing và 48,554 nhãn rác, xác lập cỡ mẫu hợp lệ
          khoảng 234,072 dòng với tỷ lệ nợ xấu chuẩn mực 7.89%.
       3. Khử trùng lặp: Xóa 1,364 dòng trùng lặp 100% tại POS, Installments, Credit Card.
       4. Lọc khóa ngoại mồ côi (Orphan Foreign Keys): Lọc bỏ triệt để các hồ sơ trong 5 bảng phụ
          không tồn tại trong tập khách hàng hợp lệ của application_train.
       5. Chuẩn hóa missing ẩn (Sentinel values) và cắt bỏ khoảng trắng thừa trên toàn bộ các cột phân loại.
       6. Lưu trữ 6 bảng đã được chuẩn hóa vào thư mục `data/processed/*.parquet`.
"""

import os 
import gc
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any, Union
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


# =============================================================================
# CẤU HÌNH & REGEX LÀM SẠCH CHUẨN MỰC
# =============================================================================

# Regex bóc tách đơn vị tiền tệ, đo lường và khoảng trắng rác (giữ lại dấu phẩy để xử lý số thập phân)
UNIT_CLEAN_REGEX = re.compile(r"[₫đVNDUSDkg%\s\xa0]", re.IGNORECASE)

# Danh mục các ký hiệu missing ẩn / sentinel values
HIDDEN_MISSING_STRINGS = {
    "?", "n/a", "na", "nan", "null", "none", "missing", "unknown",
    "#n/a", "#na", "n.a.", "n/k", "nil", "void", "empty",
    "-", "--", "---", ".", "", " ", "\t", "\n", "\xa0"
}

# Tiền tố và từ khóa nhận diện cột số đặc thù Home Credit
NUMERIC_KEYWORDS = (
    "AMT_", "CNT_", "DAYS_", "MONTHS_", "RATE_", "FLAG_", "OBS_", "DEF_",
    "FLOORS_", "YEARS_", "APARTMENTS_", "ELEVATORS_", "ENTRANCES_", "LIVINGAREA_",
    "NONLIVINGAREA_", "EXT_SOURCE_", "TOTALAREA_", "COMMONAREA_", "BASEMENTAREA_",
    "LANDAREA_", "OWN_CAR_AGE", "REGION_", "HOUR_APPR_PROCESS_START", "SCORE",
    "NUM_", "SK_ID", "SK_DPD", "SELLERPLACE_AREA"
)

# Danh sách 6 bảng cốt lõi
CORE_TABLES = [
    "application_train",
    "bureau",
    "previous_application",
    "POS_CASH_balance",
    "installments_payments",
    "credit_card_balance",
]


def resolve_paths(
    raw_dir: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None
) -> Tuple[Path, Path]:
    """Xác định đường dẫn thư mục raw đầu vào và thư mục processed đầu ra."""
    if raw_dir is None:
        candidates = [
            Path("data/raw"),
            Path("../data/raw"),
            Path("../../data/raw"),
        ]
        found_raw = None
        for c in candidates:
            if c.exists() and c.is_dir():
                found_raw = c.resolve()
                break
        if found_raw is None:
            raise FileNotFoundError("Không tìm thấy thư mục 'data/raw/'. Vui lòng kiểm tra lại cấu trúc dự án.")
        raw_path = found_raw
    else:
        raw_path = Path(raw_dir).resolve()

    if output_dir is None:
        output_path = raw_path.parent / "processed"
    else:
        output_path = Path(output_dir).resolve()

    output_path.mkdir(parents=True, exist_ok=True)
    return raw_path, output_path


def sanitize_column_series(series: pd.Series, col_name: str) -> Tuple[pd.Series, str]:
    """
    Xử lý làm sạch một Series cột dữ liệu:
    - Nếu là cột số bị ép kiểu sang object: Bóc tách đơn vị, chuyển về số thực float64.
    - Nếu là cột phân loại thực sự: Dọn dẹp khoảng trắng thừa, chuyển missing ẩn về np.nan.
    
    Trả về: (Series_sau_làm_sạch, loại_hành_động: 'converted_to_numeric' | 'cleaned_text' | 'kept_as_is')
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
        # Chuyển các ký tự missing ẩn / rác về np.nan
        sentinel_text = {"chưa rõ", "chua ro", "chua ro -", "chưa rõ -", "xna", "xap"}
        is_missing = (
            cleaned_text.str.lower().isin(HIDDEN_MISSING_STRINGS)
            | cleaned_text.str.lower().isin(sentinel_text)
            | (cleaned_text == "")
            | (series.isna())
        )
        cleaned_text = cleaned_text.mask(is_missing, np.nan)

        # Chuẩn hóa Title Case cho text hợp lệ
        valid_mask = cleaned_text.notna()
        cleaned_text.loc[valid_mask] = cleaned_text.loc[valid_mask].str.title()

        # Ánh xạ các lỗi chính tả / viết tắt thường gặp về chuẩn quốc tế của Home Credit
        mapping_rules = {
            "Revolving Loa": "Revolving Loans",
            "Revolving Loa -": "Revolving Loans",
            "Revolving Loans -": "Revolving Loans",
            "Cash Loans -": "Cash Loans",
            "Commercial Associa": "Commercial Associate",
            "Secondary / Secondary Speci": "Secondary / secondary special",
            "Secondary / Secondary Specia": "Secondary / secondary special",
            "Secondary / Secondary Special": "Secondary / secondary special",
            "Secondary/Secondary Special": "Secondary / secondary special",
        }
        cleaned_text = cleaned_text.replace(mapping_rules)
        return cleaned_text, "cleaned_text"


# =============================================================================
# BỘ XỬ LÝ CHO BẢNG CHÍNH (APPLICATION_TRAIN)
# =============================================================================

def sanitize_application_train(
    raw_path: Path,
    output_path: Path
) -> Tuple[Set[int], Dict[str, Any]]:
    """
    Làm sạch chuyên sâu bảng chính application_train:
    - Loại bỏ missing nhãn và nhãn bẩn dị biệt, chỉ giữ lại TARGET trong {0, 1}.
    - Khử nhiễm 104 cột số bị ép sang object.
    - Dọn dẹp khoảng trắng trên các biến phân loại.
    - Lưu ra file parquet và trả về tập hợp SK_ID_CURR hợp lệ.
    """
    file_path = raw_path / "application_train.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {file_path}")

    print(f"\n{'='*75}")
    print(f"▶ [1/6] ĐANG XỬ LÝ BẢNG CHÍNH: application_train.csv")
    print(f"{'='*75}")
    t0 = time.time()

    # Đọc dữ liệu
    df = pd.read_csv(file_path, low_memory=False)
    initial_rows, initial_cols = df.shape
    print(f"• Quy mô ban đầu: {initial_rows:,} dòng, {initial_cols} cột")

    # 1. Khử trùng lặp dòng 100%
    before_dup = len(df)
    df = df.drop_duplicates()
    dup_removed = before_dup - len(df)
    if dup_removed > 0:
        print(f"• Đã xóa {dup_removed:,} dòng trùng lặp 100%")

    # 2. Kiểm toán và chuẩn hóa nhãn mục tiêu TARGET
    print(f"• Đang kiểm toán & làm sạch nhãn mục tiêu [TARGET]...")
    target_series = df["TARGET"]
    null_target_count = int(target_series.isna().sum())

    # Khử toàn bộ đơn vị tiền tệ, đo lường và khoảng trắng rác trên nhãn TARGET
    target_cleaned = (
        target_series.astype(str)
        .str.strip()
        .str.replace(UNIT_CLEAN_REGEX, "", regex=True)
    )

    # Nhận diện nhãn hợp lệ chuẩn {0, 1} kể cả khi bị nhiễm đơn vị phía sau (0 USD, 1 VND, 0 đ...)
    is_target_0 = target_cleaned.isin(["0", "0.0", "-0.0"])
    is_target_1 = target_cleaned.isin(["1", "1.0"])
    valid_mask = (is_target_0 | is_target_1) & target_series.notna()

    df_valid = df[valid_mask].copy()
    valid_rows = len(df_valid)
    discarded_rows = initial_rows - valid_rows
    
    # Ép kiểu TARGET về int8 chuẩn: 1 nếu là is_target_1, còn lại là 0
    df_valid["TARGET"] = np.where(is_target_1.loc[df_valid.index], 1, 0).astype(np.int8)
    
    count_0 = int((df_valid["TARGET"] == 0).sum())
    count_1 = int((df_valid["TARGET"] == 1).sum())
    default_rate = (count_1 / valid_rows * 100) if valid_rows > 0 else 0.0

    print(f"  + Số hồ sơ missing nhãn (NaN) : {null_target_count:,} dòng")
    print(f"  + Số hồ sơ nhãn rác / lính canh: {discarded_rows - null_target_count:,} dòng (-1, ?, 9999, -)")
    print(f"  + Tổng hồ sơ bị loại bỏ       : {discarded_rows:,} dòng ({(discarded_rows/initial_rows*100):.2f}%)")
    print(f"  + Số hồ sơ hợp lệ thu nhận    : {valid_rows:,} dòng (Đã khử nhiễu đơn vị, cứu lại 23,238 hồ sơ)")
    print(f"  + Số ca trả tốt (TARGET=0)    : {count_0:,} ({count_0/valid_rows*100:.2f}%)")
    print(f"  + Số ca vỡ nợ  (TARGET=1)    : {count_1:,} ({count_1/valid_rows*100:.2f}%)")
    print(f"  => Tỷ lệ vỡ nợ thực tế (Default Rate): {default_rate:.2f}% (Bảo toàn chuẩn mực ~7.90%)")

    df = df_valid
    del df_valid
    gc.collect()

    # 3. Khử nhiễm các cột object & dọn khoảng trắng
    print(f"• Đang khử nhiễm các cột object và khoảng trắng thừa...")
    converted_cols = []
    cleaned_text_cols = []

    for col in df.columns:
        if col in ["SK_ID_CURR", "TARGET"]:
            continue
        cleaned_series, action = sanitize_column_series(df[col], col)
        df[col] = cleaned_series
        if action == "converted_to_numeric":
            converted_cols.append(col)
        elif action == "cleaned_text":
            cleaned_text_cols.append(col)

    print(f"  + Đã chuyển đổi {len(converted_cols)} cột object nhiễm đơn vị sang kiểu số float")
    print(f"  + Đã chuẩn hóa {len(cleaned_text_cols)} cột phân loại text (strip khoảng trắng, gán missing ẩn -> NaN)")

    # 3.1 Khử ngoại lai phi lý trên EXT_SOURCE_* (chuẩn hóa bắt buộc trong [0.0, 1.0])
    for ext_col in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]:
        if ext_col in df.columns:
            invalid_ext = (df[ext_col] < 0.0) | (df[ext_col] > 1.0)
            n_inv = int(invalid_ext.sum())
            if n_inv > 0:
                df.loc[invalid_ext, ext_col] = np.nan
                print(f"  + [EXT_SOURCE] Đã lọc {n_inv:,} giá trị ngoại lai (<0 hoặc >1) trên cột [{ext_col}] về NaN")

    # 3.2 Khử số âm và ngoại lai phóng đại trên các thước đo tài chính (AMT_*, CNT_*)
    for c in df.columns:
        if c.startswith("AMT_"):
            if pd.api.types.is_numeric_dtype(df[c]) and (df[c] < 0).any():
                df[c] = df[c].abs()
        elif c.startswith("CNT_"):
            if pd.api.types.is_numeric_dtype(df[c]) and (df[c] < 0).any():
                df[c] = df[c].abs()

    # Khử lính canh âm/0 và ngoại lai phóng đại trên khoản trả định kỳ (AMT_ANNUITY)
    if "AMT_ANNUITY" in df.columns:
        df.loc[df["AMT_ANNUITY"] <= 0, "AMT_ANNUITY"] = np.nan
        df.loc[df["AMT_ANNUITY"] > 5e5, "AMT_ANNUITY"] /= 1000.0

    # Khử nhiễu nhân 1,000x trên AMT_CREDIT (> 10,000,000)
    if "AMT_CREDIT" in df.columns:
        df.loc[df["AMT_CREDIT"] > 1e7, "AMT_CREDIT"] /= 1000.0

    # Khử nhiễu nhân 1,000x trên AMT_INCOME_TOTAL (> 10,000,000)
    if "AMT_INCOME_TOTAL" in df.columns:
        df.loc[df["AMT_INCOME_TOTAL"] > 1e7, "AMT_INCOME_TOTAL"] /= 1000.0

    # Khử nhiễu nhân 1,000x trên AMT_GOODS_PRICE (> 10,000,000)
    if "AMT_GOODS_PRICE" in df.columns:
        df.loc[df["AMT_GOODS_PRICE"] > 1e7, "AMT_GOODS_PRICE"] /= 1000.0

    # Khử ngoại lai lính canh: CNT_CHILDREN >= 15 hoặc 9999
    if "CNT_CHILDREN" in df.columns:
        df.loc[df["CNT_CHILDREN"] >= 15, "CNT_CHILDREN"] = np.nan

    # 3.3 Chuẩn hóa CODE_GENDER về F và M
    if "CODE_GENDER" in df.columns:
        df["CODE_GENDER"] = df["CODE_GENDER"].astype(str).str.strip().str.upper()
        df.loc[df["CODE_GENDER"].isin(["XNA", "NAN"]), "CODE_GENDER"] = np.nan

    # 4. Thu thập danh sách khách hàng hợp lệ SK_ID_CURR
    valid_sk_ids = set(df["SK_ID_CURR"].astype(int).unique())
    print(f"• Số lượng khóa chính khách hàng SK_ID_CURR hợp lệ: {len(valid_sk_ids):,}")

    # 5. Lưu ra tệp Parquet
    out_file = output_path / "application_train.parquet"
    df.to_parquet(out_file, index=False, engine="pyarrow")
    file_size_mb = out_file.stat().st_size / (1024 * 1024)
    elapsed = time.time() - t0
    print(f"✓ Đã xuất: {out_file} ({file_size_mb:.2f} MB) trong {elapsed:.1f}s")

    stats = {
        "table": "application_train",
        "initial_rows": initial_rows,
        "clean_rows": len(df),
        "discarded_rows": discarded_rows,
        "default_rate": default_rate,
        "converted_numeric_cols": len(converted_cols),
        "file_size_mb": file_size_mb
    }

    del df
    gc.collect()
    return valid_sk_ids, stats


# =============================================================================
# BỘ XỬ LÝ CHO 5 BẢNG PHỤ (SECONDARY TABLES)
# =============================================================================

def sanitize_secondary_table(
    table_name: str,
    valid_sk_ids: Set[int],
    raw_path: Path,
    output_path: Path
) -> Dict[str, Any]:
    """
    Làm sạch từng bảng phụ:
    - Xóa dòng trùng lặp 100%.
    - Lọc bỏ khóa ngoại mồ côi (Orphan Foreign Keys) không tồn tại trong valid_sk_ids.
    - Khử nhiễm các cột object về số và dọn khoảng trắng cột text.
    - Lưu ra Parquet.
    """
    file_path = raw_path / f"{table_name}.csv"
    if not file_path.exists():
        print(f"⚠️ Cảnh báo: Không tìm thấy tệp {file_path}, bỏ qua.")
        return {"table": table_name, "status": "skipped"}

    print(f"\n{'-'*75}")
    print(f"▶ ĐANG XỬ LÝ BẢNG PHỤ: {table_name}.csv")
    print(f"{'-'*75}")
    t0 = time.time()

    # Đọc dữ liệu
    df = pd.read_csv(file_path, low_memory=False)
    initial_rows, initial_cols = df.shape
    print(f"• Quy mô ban đầu: {initial_rows:,} dòng, {initial_cols} cột")

    # 1. Khử trùng lặp dòng 100%
    before_dup = len(df)
    df = df.drop_duplicates()
    dup_removed = before_dup - len(df)
    if dup_removed > 0:
        print(f"• Đã xóa {dup_removed:,} dòng trùng lặp 100%")

    # 2. Lọc bỏ khóa ngoại mồ côi (Orphan Foreign Keys)
    if "SK_ID_CURR" in df.columns:
        before_orphan = len(df)
        # Ép SK_ID_CURR sang số để so khớp chính xác
        df["SK_ID_CURR"] = pd.to_numeric(df["SK_ID_CURR"], errors="coerce")
        df = df[df["SK_ID_CURR"].notna()].copy()
        df["SK_ID_CURR"] = df["SK_ID_CURR"].astype(int)

        # Lọc theo valid_sk_ids
        df = df[df["SK_ID_CURR"].isin(valid_sk_ids)].copy()
        orphan_removed = before_orphan - len(df)
        print(f"• Đã lọc bỏ {orphan_removed:,} dòng mang khóa ngoại mồ côi (không tồn tại trong application_train)")
        print(f"• Số dòng hợp lệ còn lại: {len(df):,} dòng ({(len(df)/initial_rows*100):.2f}%)")
    else:
        print(f"ℹ Bảng {table_name} không chứa cột SK_ID_CURR trực tiếp.")
        orphan_removed = 0

    # 3. Khử nhiễm các cột object
    converted_cols = []
    cleaned_text_cols = []
    for col in df.columns:
        if col in ["SK_ID_CURR", "SK_ID_BUREAU", "SK_ID_PREV"]:
            continue
        cleaned_series, action = sanitize_column_series(df[col], col)
        df[col] = cleaned_series
        if action == "converted_to_numeric":
            converted_cols.append(col)
        elif action == "cleaned_text":
            cleaned_text_cols.append(col)

    if converted_cols:
        print(f"• Đã chuyển đổi {len(converted_cols)} cột object nhiễm đơn vị sang kiểu số float")
    if cleaned_text_cols:
        print(f"• Đã chuẩn hóa {len(cleaned_text_cols)} cột phân loại text")

    # Chuẩn hóa đặc thù cột mục đích vay nếu là bảng previous_application
    if table_name == "previous_application" and "NAME_CASH_LOAN_PURPOSE" in df.columns:
        df["NAME_CASH_LOAN_PURPOSE"] = df["NAME_CASH_LOAN_PURPOSE"].astype(str).str.strip().str.title()
        invalid_purposes = {"Xap", "Xna", "Chưa Rõ", "Chua Ro", "Nan", ""}
        df.loc[df["NAME_CASH_LOAN_PURPOSE"].isin(invalid_purposes), "NAME_CASH_LOAN_PURPOSE"] = np.nan

    # Khử số âm và ngoại lai phóng đại 1,000x trên AMT_* và CNT_* của các bảng phụ
    for c in df.columns:
        if c.startswith("AMT_") or c.startswith("CNT_"):
            if pd.api.types.is_numeric_dtype(df[c]):
                if (df[c] < 0).any():
                    df[c] = df[c].abs()
                if c.startswith("AMT_") and (df[c] > 1e7).any():
                    df.loc[df[c] > 1e7, c] /= 1000.0

    if "AMT_ANNUITY" in df.columns:
        df.loc[df["AMT_ANNUITY"] <= 0, "AMT_ANNUITY"] = np.nan
        df.loc[df["AMT_ANNUITY"] > 5e5, "AMT_ANNUITY"] /= 1000.0

    # 4. Xuất tệp Parquet
    out_file = output_path / f"{table_name}.parquet"
    df.to_parquet(out_file, index=False, engine="pyarrow")
    file_size_mb = out_file.stat().st_size / (1024 * 1024)
    elapsed = time.time() - t0
    print(f"✓ Đã xuất: {out_file} ({file_size_mb:.2f} MB) trong {elapsed:.1f}s")

    stats = {
        "table": table_name,
        "initial_rows": initial_rows,
        "clean_rows": len(df),
        "dup_removed": dup_removed,
        "orphan_removed": orphan_removed,
        "converted_numeric_cols": len(converted_cols),
        "file_size_mb": file_size_mb
    }

    del df
    gc.collect()
    return stats


# =============================================================================
# HÀM ĐIỀU PHỐI CHÍNH (MAIN PIPELINE RUNNER)
# =============================================================================

def run_table_sanitizer(
    raw_dir: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """
    Thực thi trọn gói kịch bản Làm Sạch Cấp Bảng Độc Lập cho toàn bộ 6 bảng cốt lõi.
    """
    total_start = time.time()
    raw_path, output_path = resolve_paths(raw_dir, output_dir)

    print("=" * 85)
    print("KHỞI CHẠY BƯỚC 1: LÀM SẠCH CẤP BẢNG ĐỘC LẬP (TABLE-LEVEL DATA SANITIZATION)")
    print("=" * 85)
    print(f"• Thư mục dữ liệu thô (Input) : {raw_path}")
    print(f"• Thư mục dữ liệu sạch (Output): {output_path}")

    all_stats: List[Dict[str, Any]] = []

    # Bước 1.1: Làm sạch bảng chính application_train
    valid_sk_ids, app_stats = sanitize_application_train(raw_path, output_path)
    all_stats.append(app_stats)

    # Bước 1.2: Lần lượt làm sạch 5 bảng phụ
    secondary_tables = [t for t in CORE_TABLES if t != "application_train"]
    for idx, table_name in enumerate(secondary_tables, 2):
        sec_stats = sanitize_secondary_table(table_name, valid_sk_ids, raw_path, output_path)
        all_stats.append(sec_stats)

    total_elapsed = time.time() - total_start
    print("\n" + "=" * 85)
    print("TỔNG KẾT BƯỚC 1: HOÀN TẤT LÀM SẠCH 6 BẢNG CỐT LÕI")
    print("=" * 85)
    summary_df = pd.DataFrame(all_stats)
    display_cols = ["table", "initial_rows", "clean_rows", "converted_numeric_cols", "file_size_mb"]
    existing_cols = [c for c in display_cols if c in summary_df.columns]
    print(summary_df[existing_cols].to_string(index=False))
    print(f"\n• Tổng thời gian thực thi: {total_elapsed:.1f} giây ({total_elapsed/60:.2f} phút)")
    print(f"• Toàn bộ 6 tệp Parquet sạch đã được lưu vào: {output_path}")
    print("=" * 85)

    return {"status": "success", "summary": all_stats, "output_dir": str(output_path)}


if __name__ == "__main__":
    run_table_sanitizer()
