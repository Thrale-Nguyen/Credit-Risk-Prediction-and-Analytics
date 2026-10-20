"""
Module: data_diagnostics.py
Package: source
Dự án: PRJ-01 - Dự đoán Rủi ro Vỡ nợ Tín dụng (Home Credit Default Risk)
Mô tả: Hệ thống Chẩn đoán Dữ liệu & Schema Tổng quát (Comprehensive Data & Schema Diagnostics Suite).
       Hợp nhất toàn diện khả năng khảo sát cấu trúc schema của `inspect_schema_for_merge.py` 
       và năng lực kiểm toán bóc tách dữ liệu bẩn của `audit_raw_data.py`:
       - ĐỘC LẬP TOÀN DIỆN: Hoạt động tự động trên bất kỳ tập dữ liệu nào (thư mục tệp CSV 
         hoặc Dict các DataFrame) mà KHÔNG cần tham chiếu đến `old_data`.
       - KIỂM TOÁN CẤU TRÚC & SCHEMA: Kích thước dòng/cột, dung lượng RAM & đĩa, tỷ lệ khuyết thiếu.
       - TÍNH TOÀN VẸN KHÓA & GRAIN: Phát hiện dòng trùng lặp 100%, vi phạm Grain khóa chính,
         và bóc tách khóa ngoại mồ côi (Orphaned Foreign Keys) trên các bảng con.
       - KHẢO SÁT GHÉP NỐI ĐA BẢNG (MERGING FEASIBILITY):
         + Bản số quan hệ (Cardinality 1:1, 1:N) và cảnh báo rủi ro bùng nổ dòng (Fan-out Risk).
         + Tự động phát hiện xung đột tên cột ngoài khóa chính giữa các bảng.
       - FORENSIC DIRTY DATA AUDIT:
         + Bóc tách các cột số bị tiêm chuỗi đơn vị (đ, VND, USD, kg, %) ép thành kiểu `object`.
         + Kiểm toán nhãn mục tiêu `TARGET` (bắt nhãn dị thường, đo tỷ lệ vỡ nợ Default Rate).
         + Quét missing ẩn (Sentinel values: ?, n/a, null, -1, 9999, xna...).
         + Quét khoảng trắng thừa đầu/cuối chuỗi (Whitespace pollution).
         + Quét bẫy đảo dấu biến thời gian (DAYS_* > 0).
         + Quét số âm vô lý trên các thước đo dương (AMT_*, CNT_* < 0).
         + Quét ngoại lai cực đoan & vi phạm logic nghiệp vụ ngân hàng.
       - XUẤT BÁO CÁO: Tạo báo cáo văn bản chuyên nghiệp, trực quan và đưa ra khuyến nghị
         kỹ thuật hành động cụ thể cho Pipeline Tiền xử lý & Ghép nối.
"""

import os
import gc
import sys
import re
import time
import argparse
from pathlib import Path
from typing import Dict, Union, Optional, Any, List, Set, Tuple
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
# CẤU HÌNH & TỪ ĐIỂN MẪU DỊ THƯỜNG DỮ LIỆU CHUẨN MỰC
# =============================================================================

DEFAULT_MAIN_TABLE = "application_train"
DEFAULT_JOIN_KEY = "SK_ID_CURR"

EXPECTED_TABLES = [
    "application_train",
    "bureau",
    "previous_application",
    "POS_CASH_balance",
    "installments_payments",
    "credit_card_balance",
    "bureau_balance",
]

# Danh mục mở rộng toàn diện các ký hiệu missing ẩn / sentinel values
HIDDEN_MISSING_PATTERNS = {
    # Dạng chữ
    "?", "n/a", "na", "nan", "null", "none", "missing", "unknown",
    "#n/a", "#na", "n.a.", "n/k", "nil", "void", "empty",
    # Dấu ký tự
    "-", "--", "---", ".", "", " ", "\t", "\n", "\xa0",
    # Số lính canh (dạng chuỗi)
    "-1", "-1.0", "-99", "-999", "-9999", "999", "9999", "99999", "999999",
    # Mã kỹ thuật đặc thù Home Credit
    "xna", "xap",
}

# Tập hợp số lính canh dạng số thực / nguyên
NUMERIC_SENTINELS = {-1, -1.0, -99, -999, -9999, 999, 9999, 99999, 999999}

# Regex khử đơn vị tiền tệ, đo lường và khoảng trắng rác khi kiểm tra ép kiểu
UNIT_CLEAN_REGEX = re.compile(r"[₫đVNDUSDkg%\s,\xa0]", re.IGNORECASE)

# Tiền tố các cột số thông thường trong Home Credit
NUMERIC_NAME_KEYWORDS = [
    "AMT_", "CNT_", "DAYS_", "MONTHS_", "RATE_", "FLAG_", "OBS_", "DEF_",
    "FLOORS_", "YEARS_", "APARTMENTS_", "ELEVATORS_", "ENTRANCES_", "LIVINGAREA_",
    "NONLIVINGAREA_", "EXT_SOURCE_", "TOTALAREA_", "COMMONAREA_", "BASEMENTAREA_",
    "LANDAREA_", "OWN_CAR_AGE", "REGION_", "HOUR_APPR_PROCESS_START", "SCORE",
    "NUM_", "SK_ID",
]


# =============================================================================
# CÁC HÀM TIỆN ÍCH DÒ TÌM & NẠP DỮ LIỆU
# =============================================================================

def find_dataset_directory(data_path: Optional[Union[str, Path]] = None) -> Path:
    """
    Tự động tìm kiếm thư mục dữ liệu nếu không được truyền vào rõ ràng.
    Ưu tiên tìm `data/raw` trước, sau đó tới `Dataset - Home Credit Default Risk`.
    """
    if data_path is not None:
        p = Path(data_path).resolve()
        if p.exists() and p.is_dir():
            return p
        raise FileNotFoundError(f"Đường dẫn thư mục dữ liệu không tồn tại: {p}")

    candidate_paths = [
        Path("data/raw"),
        Path("../data/raw"),
        Path("Dataset - Home Credit Default Risk"),
        Path("../Dataset - Home Credit Default Risk"),
        Path("PRJ-01/data/raw"),
        Path("PRJ-01/Dataset - Home Credit Default Risk"),
    ]

    if "__file__" in globals():
        base_dir = Path(__file__).resolve().parent
        candidate_paths.extend([
            base_dir.parent / "data" / "raw",
            base_dir.parent / "Dataset - Home Credit Default Risk",
            base_dir.parent.parent / "data" / "raw",
            base_dir.parent.parent / "Dataset - Home Credit Default Risk",
        ])

    for cand in candidate_paths:
        if cand.exists() and cand.is_dir():
            return cand.resolve()

    raise FileNotFoundError(
        "Không thể tự động định vị thư mục dữ liệu ('data/raw' hoặc 'Dataset - Home Credit Default Risk'). "
        "Vui lòng chỉ định rõ đường dẫn qua tham số `data`."
    )


# =============================================================================
# CÁC HÀM KIỂM TOÁN CHUYÊN SÂU TỪNG KHÍA CẠNH DỮ LIỆU (STANDALONE HEURISTICS)
# =============================================================================

def detect_suspected_numeric_contamination(
    series: pd.Series,
    col_name: str,
    val_counts: pd.Series
) -> Optional[Dict[str, Any]]:
    """
    Phát hiện các cột vốn dĩ là số nhưng bị ép sang kiểu `object` do bị chèn
    các chuỗi đơn vị tiền tệ/đo lường ('0 đ', '0 VND', '0 USD', '0 kg', '0 %')
    hoặc ký tự rác ('?', '-', 'nan') mà không cần tham chiếu đến `old_data`.
    """
    if not val_counts.empty and pd.api.types.is_object_dtype(series):
        # Trích xuất các giá trị duy nhất khác null
        vc_clean = val_counts[val_counts.index.notna()]
        if vc_clean.empty:
            return None
        unique_vals = vc_clean.index.astype(str)

        # Khử đơn vị và khoảng trắng
        cleaned_vals = [UNIT_CLEAN_REGEX.sub("", v) for v in unique_vals]

        # Kiểm tra tỷ lệ chuyển đổi thành công sang số
        parsed_nums = pd.to_numeric(pd.Series(cleaned_vals), errors="coerce")
        valid_mask = parsed_nums.notna().values

        # Tính tổng số dòng chuyển đổi thành công có trọng số theo tần số
        total_rows = vc_clean.sum()
        convertible_rows = int(vc_clean.iloc[valid_mask].sum())
        convertible_pct = (convertible_rows / total_rows * 100) if total_rows > 0 else 0

        # Kiểm tra tên cột có chứa từ khóa số đặc trưng
        has_numeric_name = any(kw in col_name.upper() for kw in NUMERIC_NAME_KEYWORDS) or col_name.endswith(("_AVG", "_MODE", "_MEDI"))

        # Tiêu chí nhận diện: Tỷ lệ parse thành công >= 50% hoặc (>= 20% và tên cột là cột số)
        if convertible_pct >= 50.0 or (has_numeric_name and convertible_pct >= 20.0):
            # Thu thập các giá trị gây bẩn (làm fail việc ép số)
            dirty_indices = np.where(~valid_mask)[0]
            dirty_samples = [str(unique_vals[i]) for i in dirty_indices[:8]]

            # Đề xuất kiểu dữ liệu đích
            sample_parsed = parsed_nums[valid_mask].dropna()
            is_likely_int = False
            if len(sample_parsed) > 0:
                is_likely_int = bool(np.all(sample_parsed.values == np.floor(sample_parsed.values)))

            return {
                "col_name": col_name,
                "convertible_pct": convertible_pct,
                "convertible_rows": convertible_rows,
                "total_rows": total_rows,
                "suggested_dtype": "int64" if is_likely_int else "float64",
                "dirty_samples": dirty_samples,
                "has_numeric_name": has_numeric_name
            }

    return None


def audit_target_column(series: pd.Series) -> Dict[str, Any]:
    """
    Kiểm toán chuyên sâu nhãn mục tiêu TARGET trong bảng chính application_train:
    - Bắt các nhãn dị thường ngoài {0, 1} (nhãn tiền tệ '0 đ', '0 VND', số lính canh '-1', '9999'...).
    - Thống kê tỷ lệ khuyết thiếu của nhãn TARGET.
    - Tính toán tỷ lệ nợ xấu (Default Rate) trên tập nhãn hợp lệ chuẩn {0, 1}.
    """
    total_len = len(series)
    null_count = int(series.isna().sum())
    val_counts = series.value_counts(dropna=False).to_dict()

    # Nhận diện các giá trị hợp lệ nhị phân chuẩn
    valid_binary_0 = {"0", "0.0", 0, 0.0}
    valid_binary_1 = {"1", "1.0", 1, 1.0}

    count_0 = 0
    count_1 = 0
    invalid_labels: Dict[str, int] = {}

    for val, count in val_counts.items():
        if pd.isna(val):
            continue
        val_str = str(val).strip()
        if val in valid_binary_0 or val_str in valid_binary_0:
            count_0 += int(count)
        elif val in valid_binary_1 or val_str in valid_binary_1:
            count_1 += int(count)
        else:
            invalid_labels[str(val)] = int(count)

    total_valid = count_0 + count_1
    default_rate = (count_1 / total_valid * 100) if total_valid > 0 else 0.0
    total_invalid_rows = sum(invalid_labels.values())
    contamination_pct = ((total_invalid_rows + null_count) / total_len * 100) if total_len > 0 else 0.0

    return {
        "total_rows": total_len,
        "null_count": null_count,
        "count_0": count_0,
        "count_1": count_1,
        "total_valid": total_valid,
        "default_rate": default_rate,
        "invalid_labels": invalid_labels,
        "total_invalid_rows": total_invalid_rows,
        "contamination_pct": contamination_pct,
        "is_contaminated": bool(invalid_labels or null_count > 0)
    }


def scan_hidden_missing(
    df: pd.DataFrame,
    str_cols: List[str]
) -> Dict[str, Dict[str, int]]:
    """
    Quét tìm các ký tự missing ẩn và số lính canh (Sentinel Values) trên các cột chuỗi.
    Tối ưu hóa tốc độ cực đại bằng value_counts() trên từng cột thay vì duyệt từng dòng.
    """
    hidden_missing_found: Dict[str, Dict[str, int]] = {}

    for col in str_cols:
        vc = df[col].dropna().value_counts()
        if vc.empty:
            continue

        # Chuẩn hóa giá trị unique để đối chiếu nhanh
        normalized_idx = vc.index.astype(str).str.strip().str.lower()
        patterns_hit: Dict[str, int] = {}

        for pattern in HIDDEN_MISSING_PATTERNS:
            mask = (normalized_idx == pattern)
            if mask.any():
                hit_cnt = int(vc.iloc[mask].sum())
                if hit_cnt > 0:
                    patterns_hit[pattern] = hit_cnt

        if patterns_hit:
            hidden_missing_found[col] = patterns_hit

    return hidden_missing_found


def scan_whitespace_pollution(
    df: pd.DataFrame,
    str_cols: List[str]
) -> Dict[str, int]:
    """
    Quét khoảng trắng thừa đầu/cuối chuỗi (Whitespace Pollution) trên các cột object.
    Tối ưu hóa qua index của value_counts().
    """
    whitespace_cols: Dict[str, int] = {}

    for col in str_cols:
        vc = df[col].dropna().value_counts()
        if vc.empty:
            continue

        idx_str = vc.index.astype(str)
        ws_mask = idx_str.str.startswith(" ") | idx_str.str.endswith(" ") | idx_str.str.contains("[\t\n\xa0]", regex=True)
        if ws_mask.any():
            polluted_cnt = int(vc.iloc[ws_mask].sum())
            if polluted_cnt > 0:
                whitespace_cols[col] = polluted_cnt

    return whitespace_cols


def scan_temporal_anomalies(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Quét bẫy đảo dấu biến thời gian DAYS_* (> 0 bất thường).
    Trong nghiệp vụ Home Credit, DAYS_* đo khoảng thời gian từ quá khứ đến hiện tại nên phải mang dấu âm.
    Ngoại lệ hợp lệ duy nhất: DAYS_EMPLOYED == 365243 (mã kỹ thuật cho người hưu trí / thất nghiệp).
    """
    anomalies: List[Dict[str, Any]] = []
    days_cols = [c for c in df.columns if c.startswith("DAYS_")]

    for col in days_cols:
        s = df[col]
        # Xử lý nếu cột đang bị ép kiểu chuỗi
        if pd.api.types.is_object_dtype(s):
            cleaned = s.dropna().astype(str).str.replace(UNIT_CLEAN_REGEX, "", regex=True)
            num_s = pd.to_numeric(cleaned, errors="coerce")
        elif pd.api.types.is_numeric_dtype(s):
            num_s = s.dropna()
        else:
            continue

        if col == "DAYS_EMPLOYED":
            pos_mask = (num_s > 0) & (num_s != 365243)
        else:
            pos_mask = (num_s > 0)

        pos_cnt = int(pos_mask.sum())
        if pos_cnt > 0:
            max_val = float(num_s[pos_mask].max())
            anomalies.append({
                "column": col,
                "positive_count": pos_cnt,
                "max_value": max_val
            })

    return anomalies


def scan_illogical_negatives(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Quét giá trị âm vô lý trên các thước đo dương:
    - Khoản tiền (AMT_*): Thu nhập, Hạn mức vay, Khoản trả định kỳ, Giá trị hàng hóa, Dư nợ, Thanh toán...
    - Số lượng (CNT_*): Số con, Số thành viên gia đình, Số lần tra cứu CIC...
    - Tự động bỏ qua các cột hợp lệ mang dấu âm (DAYS_*, MONTHS_*, SK_ID).
    """
    neg_anomalies: List[Dict[str, Any]] = []

    for col in df.columns:
        if any(prefix in col for prefix in ["DAYS_", "MONTHS_", "SK_ID"]):
            continue
        # Tập trung vào AMT_* và CNT_*
        if not (col.startswith("AMT_") or col.startswith("CNT_") or "PRICE" in col or "INCOME" in col):
            continue

        s = df[col]
        if pd.api.types.is_object_dtype(s):
            cleaned = s.dropna().astype(str).str.replace(UNIT_CLEAN_REGEX, "", regex=True)
            num_s = pd.to_numeric(cleaned, errors="coerce")
        elif pd.api.types.is_numeric_dtype(s):
            num_s = s.dropna()
        else:
            continue

        neg_mask = (num_s < 0)
        neg_cnt = int(neg_mask.sum())
        if neg_cnt > 0:
            min_val = float(num_s[neg_mask].min())
            neg_anomalies.append({
                "column": col,
                "negative_count": neg_cnt,
                "min_value": min_val
            })

    return neg_anomalies


def scan_extreme_domain_outliers(tbl_name: str, df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Quét các ngoại lai cực đoan phá vỡ logic thực tế nghiệp vụ tài chính - nhân khẩu học:
    - Thu nhập cá nhân > 10,000,000 (10 triệu VND/tháng chuẩn hoặc điểm lỗi 117M).
    - Số con cái > 10 hoặc số người trong gia đình > 15.
    - Khoản trả định kỳ ANNUITY > Hạn mức vay CREDIT.
    - Độ tuổi DAYS_BIRTH quy đổi ra năm < 18 hoặc > 100 tuổi.
    """
    outliers: List[Dict[str, Any]] = []

    def get_numeric_series(col: str) -> Optional[pd.Series]:
        if col not in df.columns:
            return None
        s = df[col]
        if pd.api.types.is_object_dtype(s):
            cleaned = s.astype(str).str.replace(UNIT_CLEAN_REGEX, "", regex=True)
            return pd.to_numeric(cleaned, errors="coerce")
        elif pd.api.types.is_numeric_dtype(s):
            return s
        return None

    # 1. Thu nhập cực đoan
    s_inc = get_numeric_series("AMT_INCOME_TOTAL")
    if s_inc is not None:
        s_inc_clean = s_inc.dropna()
        high_inc = int((s_inc_clean > 10_000_000).sum())
        zero_or_neg = int((s_inc_clean <= 0).sum())
        if high_inc > 0 or zero_or_neg > 0:
            max_inc = float(s_inc_clean.max()) if len(s_inc_clean) > 0 else 0.0
            outliers.append({
                "metric": "AMT_INCOME_TOTAL",
                "description": f"Thu nhập > 10M ({high_inc:,d} dòng, Max: {max_inc:,.0f}) | Thu nhập <= 0 ({zero_or_neg:,d} dòng)"
            })

    # 2. Số con cái bất thường
    s_child = get_numeric_series("CNT_CHILDREN")
    if s_child is not None:
        s_child_clean = s_child.dropna()
        extreme_child = int((s_child_clean > 10).sum())
        if extreme_child > 0:
            max_child = float(s_child_clean.max()) if len(s_child_clean) > 0 else 0.0
            outliers.append({
                "metric": "CNT_CHILDREN",
                "description": f"Số con cái > 10 ({extreme_child:,d} dòng, Max: {max_child:.0f})"
            })

    # 3. Khoản trả định kỳ vượt hạn mức vay
    s_ann = get_numeric_series("AMT_ANNUITY")
    s_cre = get_numeric_series("AMT_CREDIT")
    if s_ann is not None and s_cre is not None:
        # So sánh dựa trên mảng numpy để tránh lỗi lệch index giữa 2 cột có null khác nhau
        val_ann = s_ann.to_numpy(dtype=float, na_value=np.nan)
        val_cre = s_cre.to_numpy(dtype=float, na_value=np.nan)
        mask_valid = np.isfinite(val_ann) & np.isfinite(val_cre)
        ann_exceed = int(((val_ann > val_cre) & (val_cre > 0) & mask_valid).sum())
        if ann_exceed > 0:
            outliers.append({
                "metric": "AMT_ANNUITY > AMT_CREDIT",
                "description": f"Khoản trả định kỳ vượt tổng hạn mức vay ({ann_exceed:,d} hồ sơ)"
            })

    # 4. Độ tuổi bất thường
    s_birth = get_numeric_series("DAYS_BIRTH")
    if s_birth is not None:
        ages = (s_birth.dropna().abs() / 365.25)
        under_18 = int((ages < 18).sum())
        over_100 = int((ages > 100).sum())
        if under_18 > 0 or over_100 > 0:
            outliers.append({
                "metric": "DAYS_BIRTH (Độ tuổi)",
                "description": f"Dưới 18 tuổi ({under_18:,d} hồ sơ) | Trên 100 tuổi ({over_100:,d} hồ sơ)"
            })

    return outliers


# =============================================================================
# HÀM KIỂM TOÁN TỔNG THỂ CHO MỘT BẢNG DỮ LIỆU ĐỘC LẬP (SINGLE TABLE AUDIT)
# =============================================================================

def audit_single_table_health(
    tbl_name: str,
    df: pd.DataFrame,
    main_keys: Optional[Set[int]] = None,
    is_main_table: bool = False,
    join_key: str = DEFAULT_JOIN_KEY,
    file_size_mb: Optional[float] = None
) -> Dict[str, Any]:
    """
    Thực hiện kiểm toán toàn diện tất cả các tiêu chí chất lượng cho một DataFrame độc lập.
    """
    n_rows, n_cols = df.shape
    mem_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)

    # 1. Thống kê kiểu dữ liệu tổng thể
    dtype_summary: Dict[str, int] = {}
    for dt, count in df.dtypes.value_counts().items():
        dtype_summary[str(dt)] = int(count)

    # 2. Kiểm tra trùng lặp 100% dòng (Exact Duplicates)
    dup_rows = int(df.duplicated().sum())

    # 3. Kiểm tra tính toàn vẹn khóa chính / khóa ghép nối
    has_join_key = join_key in df.columns
    key_nulls = int(df[join_key].isna().sum()) if has_join_key else -1
    key_unique = int(df[join_key].nunique()) if has_join_key else 0
    key_dtype = str(df[join_key].dtype) if has_join_key else "N/A"
    dup_main_keys = int(df[join_key].duplicated().sum()) if (has_join_key and is_main_table) else 0

    # 4. Kiểm tra khóa ngoại mồ côi (Orphaned Foreign Keys) đối chiếu với bảng chính
    orphaned_keys_count = 0
    orphaned_rows_count = 0
    orphaned_sample_ids: List[int] = []

    if has_join_key and not is_main_table and main_keys is not None:
        try:
            sub_keys = set(df[join_key].dropna().astype(int).unique())
            orphaned_set = sub_keys - main_keys
            orphaned_keys_count = len(orphaned_set)
            if orphaned_keys_count > 0:
                orphaned_sample_ids = sorted(list(orphaned_set))[:5]
                orphaned_rows_count = int(df[join_key].isin(orphaned_set).sum())
        except Exception:
            pass

    # 5. Phân tích khuyết thiếu cơ bản
    null_counts = df.isna().sum()
    cols_with_null = int((null_counts > 0).sum())
    max_null_pct = float(null_counts.max() / n_rows * 100) if n_rows > 0 else 0.0
    max_null_col = str(null_counts.idxmax()) if cols_with_null > 0 else "None"
    high_null_cols = [c for c, cnt in null_counts.items() if (cnt / n_rows) >= 0.80]
    all_null_cols = [c for c, cnt in null_counts.items() if cnt == n_rows]
    constant_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]

    # 6. Kiểm toán nhãn TARGET nếu có
    target_audit = None
    if "TARGET" in df.columns:
        target_audit = audit_target_column(df["TARGET"])

    # 7. Quét các cột Object: Tìm cột số bị nhiễm chuỗi, missing ẩn, whitespace thừa
    str_cols = df.select_dtypes(include=["object"]).columns.tolist()
    suspected_contaminated_cols: List[Dict[str, Any]] = []

    for col in str_cols:
        vc = df[col].value_counts(dropna=False)
        contam_info = detect_suspected_numeric_contamination(df[col], col, vc)
        if contam_info:
            suspected_contaminated_cols.append(contam_info)

    hidden_missing = scan_hidden_missing(df, str_cols)
    whitespace_pollution = scan_whitespace_pollution(df, str_cols)

    # 8. Quét dị thường logic số & thời gian
    temporal_anomalies = scan_temporal_anomalies(df)
    illogical_negatives = scan_illogical_negatives(df)
    extreme_outliers = scan_extreme_domain_outliers(tbl_name, df)

    return {
        "table_name": tbl_name,
        "is_main_table": is_main_table,
        "rows": n_rows,
        "cols": n_cols,
        "memory_mb": mem_mb,
        "file_size_mb": file_size_mb,
        "columns": list(df.columns),
        "dtypes": dtype_summary,
        "dup_rows": dup_rows,
        "has_join_key": has_join_key,
        "join_key": join_key,
        "key_dtype": key_dtype,
        "key_nulls": key_nulls,
        "key_unique": key_unique,
        "dup_main_keys": dup_main_keys,
        "orphaned_keys_count": orphaned_keys_count,
        "orphaned_rows_count": orphaned_rows_count,
        "orphaned_sample_ids": orphaned_sample_ids,
        "cols_with_null": cols_with_null,
        "max_null_pct": max_null_pct,
        "max_null_col": max_null_col,
        "high_null_cols": high_null_cols,
        "all_null_cols": all_null_cols,
        "constant_cols": constant_cols,
        "target_audit": target_audit,
        "suspected_contaminated_cols": suspected_contaminated_cols,
        "hidden_missing": hidden_missing,
        "whitespace_pollution": whitespace_pollution,
        "temporal_anomalies": temporal_anomalies,
        "illogical_negatives": illogical_negatives,
        "extreme_outliers": extreme_outliers,
    }


# =============================================================================
# HÀM PHÂN TÍCH QUAN HỆ & BẢN SỐ GHÉP NỐI ĐA BẢNG (RELATIONAL / MERGE AUDIT)
# =============================================================================

def analyze_merging_conditions(
    table_audits: Dict[str, Dict[str, Any]],
    main_table_name: str = DEFAULT_MAIN_TABLE,
    join_key: str = DEFAULT_JOIN_KEY
) -> Dict[str, Any]:
    """
    Khảo sát cấu trúc khóa, cardinality, rủi ro bùng nổ dòng và xung đột tên cột
    giữa các bảng phụ và bảng chính (Kế thừa trọn vẹn từ inspect_schema_for_merge).
    """
    if main_table_name not in table_audits:
        # Tự động tìm bảng chính có tên chứa application_train
        for t in table_audits:
            if "application_train" in t.lower():
                main_table_name = t
                break

    main_info = table_audits.get(main_table_name, {})
    main_cols = set(main_info.get("columns", []))
    main_rows = main_info.get("rows", 0)
    main_unique = main_info.get("key_unique", 0)
    main_key_dtype = main_info.get("key_dtype", "N/A")

    join_relations: List[Dict[str, Any]] = []

    for tbl_name, info in table_audits.items():
        if tbl_name == main_table_name:
            continue

        sec_cols = set(info["columns"])
        common_cols = sorted(list(main_cols.intersection(sec_cols)))
        conflicting_cols = [c for c in common_cols if c != join_key]

        sec_rows = info["rows"]
        sec_unique = info["key_unique"]
        has_key = info["has_join_key"]

        if not has_key or sec_unique == 0:
            cardinality = "KHÔNG CÓ KHÓA"
            fan_out_ratio = 0.0
            merge_risk = "KHÔNG THỂ GHÉP: Thiếu khóa chính chung."
        elif sec_rows == sec_unique:
            cardinality = "1 : 1 (Độc nhất)"
            fan_out_ratio = 1.0
            merge_risk = "AN TOÀN: Có thể Left Join trực tiếp không làm tăng dòng."
        else:
            fan_out_ratio = sec_rows / sec_unique if sec_unique > 0 else 0.0
            cardinality = f"1 : N (Trung bình {fan_out_ratio:.1f} dòng/khách hàng)"
            merge_risk = f"BÙNG NỔ DÒNG (Fan-out)! Bắt buộc GroupBy Aggregate trước khi Merge."

        key_match_type = (info["key_dtype"] == main_key_dtype) if has_key else False

        join_relations.append({
            "table_name": tbl_name,
            "join_key": join_key if has_key else "THIẾU",
            "key_match_type": key_match_type,
            "sec_key_dtype": info["key_dtype"],
            "sec_rows": sec_rows,
            "sec_unique": sec_unique,
            "cardinality": cardinality,
            "fan_out_ratio": fan_out_ratio,
            "conflicting_cols": conflicting_cols,
            "merge_risk": merge_risk
        })

    return {
        "main_table_name": main_table_name,
        "main_rows": main_rows,
        "main_unique": main_unique,
        "main_key_dtype": main_key_dtype,
        "join_key": join_key,
        "relations": join_relations
    }


# =============================================================================
# HÀM TẠO BÁO CÁO VĂN BẢN ĐỊNH DẠNG CHUYÊN NGHIỆP (REPORT FORMATTER)
# =============================================================================

def format_diagnostic_report(
    table_audits: Dict[str, Dict[str, Any]],
    merge_analysis: Dict[str, Any],
    data_source_desc: str = ""
) -> str:
    """
    Tạo báo cáo chẩn đoán dữ liệu toàn diện dạng text chuẩn UTF-8, bố cục trực quan,
    chia thành 8 mục rõ ràng, phục vụ tra cứu kỹ thuật và báo cáo hội đồng.
    """
    sep_double = "=" * 95
    sep_single = "-" * 95

    lines: List[str] = []
    lines.append(sep_double)
    lines.append("BÁO CÁO CHẨN ĐOÁN DỮ LIỆU & SCHEMA TỔNG QUÁT (DATA DIAGNOSTICS & QUALITY AUDIT)")
    lines.append("Dự án: PRJ-01 - Dự đoán Rủi ro Vỡ nợ Tín dụng (Home Credit Default Risk)")
    lines.append(sep_double)
    if data_source_desc:
        lines.append(f"• Nguồn dữ liệu chẩn đoán : {data_source_desc}")
    lines.append(f"• Bảng chính mục tiêu     : {merge_analysis['main_table_name']}")
    lines.append(f"• Khóa ghép nối chủ lực   : {merge_analysis['join_key']}")
    lines.append(f"• Tổng số bảng khảo sát   : {len(table_audits)} bảng")
    lines.append(f"• Thời gian chẩn đoán     : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(sep_single)

    # ---------------------------------------------------------
    # MỤC 1: BẢNG TỔNG QUAN HỆ THỐNG CƠ SỞ DỮ LIỆU
    # ---------------------------------------------------------
    lines.append("\n[MỤC 1] TỔNG QUAN QUY MÔ, BỘ NHỚ, TRÙNG LẶP & TỶ LỆ KHUYẾT THIẾU")
    lines.append(sep_single)
    header1 = f"{'Tên Bảng':<24} | {'Số Dòng':>11} | {'Số Cột':>6} | {'RAM (MB)':>9} | {'Trùng Lặp':>9} | {'Cột Null':>8} | {'Max Null %':>10}"
    lines.append(header1)
    lines.append(sep_single)

    for tbl_name, info in table_audits.items():
        lines.append(
            f"{tbl_name:<24} | {info['rows']:>11,d} | {info['cols']:>6d} | {info['memory_mb']:>9.2f} | "
            f"{info['dup_rows']:>9,d} | {info['cols_with_null']:>8d} | {info['max_null_pct']:>9.1f}%"
        )
    lines.append(sep_single)

    # ---------------------------------------------------------
    # MỤC 2: KHẢO SÁT QUAN HỆ GHÉP NỐI & RỦI RO BÙNG NỔ DÒNG
    # ---------------------------------------------------------
    lines.append("\n[MỤC 2] KHẢO SÁT QUAN HỆ GHÉP NỐI, BẢN SỐ (CARDINALITY) & XUNG ĐỘT TÊN CỘT")
    lines.append(sep_single)
    lines.append(f"• Khóa trên bảng chính [{merge_analysis['main_table_name']}]: "
                 f"{merge_analysis['main_unique']:,d} unique / {merge_analysis['main_rows']:,d} dòng -> Chuẩn Grain 1 khách hàng = 1 dòng")
    lines.append(sep_single)

    for item in merge_analysis["relations"]:
        lines.append(f"▶ Cặp ghép: [{merge_analysis['main_table_name']}] <---> [{item['table_name']}]")
        lines.append(f"   - Khóa ghép nối       : '{item['join_key']}' (Dtype: {item['sec_key_dtype']} | Khớp bảng chính: {'CÓ' if item['key_match_type'] else 'KHÔNG'})")
        lines.append(f"   - Bản số quan hệ      : {item['cardinality']}")
        lines.append(f"   - Hồ sơ khách hàng    : {item['sec_unique']:,d} unique keys trên {item['sec_rows']:,d} dòng")
        lines.append(f"   - Đánh giá ghép nối   : {item['merge_risk']}")
        if item["conflicting_cols"]:
            lines.append(f"   ⚠️ Xung đột tên cột    : {len(item['conflicting_cols'])} cột bị trùng ngoài khóa chính -> {item['conflicting_cols'][:8]}")
            lines.append(f"      Khuyến nghị: Cần gắn tiền tố (prefix) trước khi ghép để tránh sinh hậu tố `_x`, `_y`.")
        else:
            lines.append("   ✓ Xung đột tên cột    : Không có cột trùng ngoài khóa chính.")
        lines.append("")
    lines.append(sep_single)

    # ---------------------------------------------------------
    # MỤC 3: TÍNH TOÀN VẸN KHÓA & KHÓA NGOẠI MỒ CÔI
    # ---------------------------------------------------------
    lines.append("[MỤC 3] TÍNH TOÀN VẸN KHÓA: GRAIN VIOLATIONS & KHÓA NGOẠI MỒ CÔI (ORPHAN KEYS)")
    lines.append(sep_single)

    main_info = table_audits.get(merge_analysis["main_table_name"])
    if main_info:
        if main_info["dup_main_keys"] > 0:
            lines.append(f"🚨 VI PHẠM GRAIN NGHIÊM TRỌNG TRÊN BẢNG CHÍNH [{main_info['table_name']}]:")
            lines.append(f"   -> Phát hiện {main_info['dup_main_keys']:,d} mã {merge_analysis['join_key']} BỊ TRÙNG LẶP!")
        else:
            lines.append(f"✓ Bảng chính [{main_info['table_name']}]: Khóa chính {merge_analysis['join_key']} là DUY NHẤT 100% (Bảo toàn 1 khách hàng = 1 dòng).")

    has_orphans = False
    for tbl_name, info in table_audits.items():
        if info["is_main_table"]:
            continue
        if info["orphaned_keys_count"] > 0:
            has_orphans = True
            lines.append(f"🚨 BẢNG PHỤ [{tbl_name}]: Phát hiện {info['orphaned_keys_count']:,d} mã khóa ({info['orphaned_rows_count']:,d} dòng) "
                         f"KHÔNG TỒN TẠI ở bảng chính [{merge_analysis['main_table_name']}]!")
            lines.append(f"   -> Ví dụ ID mồ côi: {info['orphaned_sample_ids']}")
            lines.append(f"   -> Khuyến nghị: Khi Left Join, các dòng này sẽ tự động bị loại bỏ, hoặc cần lọc trước để tăng tốc.\n")

    if not has_orphans:
        lines.append("✓ 100% mã khóa trên tất cả các bảng phụ đều khớp hoàn hảo với bảng chính (Zero Orphan Keys).\n")
    lines.append(sep_single)

    # ---------------------------------------------------------
    # MỤC 4: KIỂM TOÁN NHÃN MỤC TIÊU TARGET
    # ---------------------------------------------------------
    lines.append("[MỤC 4] KIỂM TOÁN CHUYÊN SÂU NHÃN MỤC TIÊU TARGET")
    lines.append(sep_single)

    target_found = False
    for tbl_name, info in table_audits.items():
        if info["target_audit"]:
            target_found = True
            t_aud = info["target_audit"]
            lines.append(f"• Bảng chứa TARGET: [{tbl_name}] ({t_aud['total_rows']:,d} dòng)")
            lines.append(f"• Phân bố nhãn chuẩn {0, 1}:")
            lines.append(f"   - Nhãn 0 (Trả nợ tốt) : {t_aud['count_0']:,d} hồ sơ")
            lines.append(f"   - Nhãn 1 (Vỡ nợ)      : {t_aud['count_1']:,d} hồ sơ")
            lines.append(f"   - Tỷ lệ vỡ nợ (Default Rate chuẩn): {t_aud['default_rate']:.2f}% (Tỷ lệ lệch nhãn: ~1 : {t_aud['count_0']/max(1, t_aud['count_1']):.1f})")

            if t_aud["null_count"] > 0:
                lines.append(f"🚨 Phát hiện {t_aud['null_count']:,d} hồ sơ có nhãn TARGET bị NULL (NaN)!")

            if t_aud["invalid_labels"]:
                lines.append(f"🚨 PHÁT HIỆN NHÃN DỊ BIỆT BẤT THƯỜNG / ĐẦU ĐỘC CHUỖI:")
                lines.append(f"   - Tổng số dòng mang nhãn bẩn : {t_aud['total_invalid_rows']:,d} dòng ({t_aud['contamination_pct']:.2f}% toàn bảng)")
                lines.append(f"   - Chi tiết các nhãn dị biệt  :")
                for inv_val, inv_cnt in list(t_aud["invalid_labels"].items())[:15]:
                    lines.append(f"        * '{inv_val}': {inv_cnt:,d} hồ sơ")
                lines.append(f"   - Khuyến nghị: Bắt buộc phải có bước Table-level Sanitizer chuẩn hóa về {{0, 1}}.")
            else:
                lines.append("✓ Nhãn TARGET hoàn toàn sạch sẽ, không chứa ký tự bẩn hay giá trị dị thường.")
            lines.append("")

    if not target_found:
        lines.append("• Không tìm thấy cột 'TARGET' trong các bảng được kiểm toán.\n")
    lines.append(sep_single)

    # ---------------------------------------------------------
    # MỤC 5: PHÁT HIỆN ÉP KIỂU NHIỄM BẨN (NUMERIC COERCED TO OBJECT)
    # ---------------------------------------------------------
    lines.append("[MỤC 5] PHÁT HIỆN CỘT SỐ BỊ ÉP SANG OBJECT (SUSPECTED NUMERIC COERCED TO OBJECT)")
    lines.append(sep_single)

    total_contam_cols = 0
    for tbl_name, info in table_audits.items():
        contams = info["suspected_contaminated_cols"]
        if contams:
            total_contam_cols += len(contams)
            lines.append(f"🚨 BẢNG [{tbl_name}]: Phát hiện {len(contams)} cột có bản chất là số nhưng bị ép thành `object` do chuỗi bẩn:")
            for c_inf in contams[:12]:
                lines.append(f"   * Cột '{c_inf['col_name']}': {c_inf['convertible_pct']:.1f}% số hợp lệ (Đề xuất ép về: {c_inf['suggested_dtype']})")
                lines.append(f"     -> Chuỗi gây bẩn mẫu: {c_inf['dirty_samples']}")
            if len(contams) > 12:
                lines.append(f"     ... và {len(contams) - 12} cột khác tương tự.")
            lines.append("")

    if total_contam_cols == 0:
        lines.append("✓ Không phát hiện hiện tượng ép kiểu nhầm lẫn từ số sang object trên các bảng.\n")
    lines.append(sep_single)

    # ---------------------------------------------------------
    # MỤC 6: BẪY MISSING ẨN & NHIỄM BẨN KHOẢNG TRẮNG
    # ---------------------------------------------------------
    lines.append("[MỤC 6] BẪY MISSING ẨN (SENTINEL VALUES) & NHIỄM BẨN KHOẢNG TRẮNG (WHITESPACE POLLUTION)")
    lines.append(sep_single)

    for tbl_name, info in table_audits.items():
        hm = info["hidden_missing"]
        ws = info["whitespace_pollution"]

        if hm or ws:
            lines.append(f"▶ Bảng [{tbl_name}]:")
            if hm:
                lines.append(f"   🚨 Missing ẩn trên {len(hm)} cột:")
                for c, hits in list(hm.items())[:8]:
                    lines.append(f"      * Cột '{c}': {hits}")
                if len(hm) > 8:
                    lines.append(f"      ... và {len(hm) - 8} cột khác.")
            if ws:
                lines.append(f"   ⚠️ Khoảng trắng đầu/cuối trên {len(ws)} cột:")
                for c, cnt in list(ws.items())[:8]:
                    lines.append(f"      * Cột '{c}': {cnt:,d} dòng bị thừa khoảng trắng")
                if len(ws) > 8:
                    lines.append(f"      ... và {len(ws) - 8} cột khác.")
            lines.append("")
        else:
            lines.append(f"✓ Bảng [{tbl_name}]: Sạch hoàn toàn missing ẩn kinh điển và khoảng trắng thừa.\n")
    lines.append(sep_single)

    # ---------------------------------------------------------
    # MỤC 7: DỊ THƯỜNG LOGIC THỜI GIAN, SỐ ÂM & NGOẠI LAI
    # ---------------------------------------------------------
    lines.append("[MỤC 7] DỊ THƯỜNG NGHIỆP VỤ: ĐẢO DẤU THỜI GIAN, SỐ ÂM VÔ LÝ & NGOẠI LAI CỰC ĐOAN")
    lines.append(sep_single)

    for tbl_name, info in table_audits.items():
        anoms_time = info["temporal_anomalies"]
        anoms_neg = info["illogical_negatives"]
        outliers = info["extreme_outliers"]

        if anoms_time or anoms_neg or outliers:
            lines.append(f"▶ Bảng [{tbl_name}]:")
            if anoms_time:
                lines.append(f"   🚨 Bẫy đảo dấu thời gian ({len(anoms_time)} cột DAYS_* bị dương hóa bất thường):")
                for item in anoms_time:
                    lines.append(f"      * Cột '{item['column']}': {item['positive_count']:,d} dòng > 0 (Max: {item['max_value']:,.1f})")
            if anoms_neg:
                lines.append(f"   🚨 Số âm vô lý trên thước đo dương ({len(anoms_neg)} cột):")
                for item in anoms_neg[:6]:
                    lines.append(f"      * Cột '{item['column']}': {item['negative_count']:,d} dòng mang dấu âm (Min: {item['min_value']:,.2f})")
            if outliers:
                lines.append(f"   ⚠️ Ngoại lai cực đoan & Vi phạm quy tắc nghiệp vụ ({len(outliers)} chỉ số):")
                for item in outliers:
                    lines.append(f"      * {item['metric']}: {item['description']}")
            lines.append("")
        else:
            lines.append(f"✓ Bảng [{tbl_name}]: Không phát hiện dị thường đảo dấu thời gian, số âm hay ngoại lai cực đoan.\n")

    # ---------------------------------------------------------
    # MỤC 8: TỔNG KẾT KHUYẾN NGHỊ HÀNH ĐỘNG
    # ---------------------------------------------------------
    lines.append(sep_double)
    lines.append("TỔNG HỢP KHUYẾN NGHỊ KỸ THUẬT CHO PIPELINE LÀM SẠCH VÀ GHÉP NỐI:")
    lines.append("1. Giai đoạn Tiền Làm Sạch Cấp Bảng (Table-Level Sanitizer):")
    lines.append("   - Khử toàn bộ đơn vị tiền tệ/đo lường ('đ', 'VND', 'USD', 'kg', '%') trước khi ép kiểu số.")
    lines.append("   - Chuẩn hóa nhãn TARGET về đúng nhị phân {0, 1}; xử lý hoặc loại bỏ các dòng TARGET bị NaN/rác.")
    lines.append("   - Chuyển đổi các ký hiệu missing ẩn (?, n/a, -, 9999...) thành `np.nan` chuẩn.")
    lines.append("   - Áp dụng .str.strip() trên toàn bộ các cột phân loại để loại bỏ khoảng trắng thừa.")
    lines.append("2. Xử Lý Bẫy Thời Gian & Dị Thường Nghiệp Vụ:")
    lines.append("   - Đảo dấu hoặc lấy abs() đối với các cột DAYS_* bị dương hóa bất thường.")
    lines.append("   - Riêng DAYS_EMPLOYED == 365243: tạo cờ DAYS_EMPLOYED_ANOM = 1 và gán lại np.nan.")
    lines.append("   - Xử lý số âm vô lý trên AMT_* bằng cách gán np.nan hoặc clip về 0.0.")
    lines.append("3. Giai đoạn Ghép Nối Dữ Liệu Đa Bảng (Data Merging Pipeline):")
    lines.append("   - TUYỆT ĐỐI KHÔNG Left Join trực tiếp các bảng quan hệ 1:N vào application_train.")
    lines.append("   - Bắt buộc phải GroupBy theo SK_ID_CURR và tính Aggregation (min, max, mean, sum cho số; OHE cho phân loại).")
    lines.append("   - Gắn tiền tố (prefix) rõ ràng cho tất cả các cột phái sinh để tránh xung đột tên cột (_x, _y).")
    lines.append("   - Lọc bỏ hoặc bỏ qua các mã SK_ID_CURR mồ côi (không tồn tại trong bảng chính).")
    lines.append(sep_double)

    return "\n".join(lines)


# =============================================================================
# HÀM ĐIỀU PHỐI CHÍNH TOÀN BỘ SUITE CHẨN ĐOÁN (MAIN SUITE COORDINATOR)
# =============================================================================

def diagnose_dataset(
    data: Optional[Union[Dict[str, pd.DataFrame], str, Path]] = None,
    output_file: Optional[Union[str, Path]] = None,
    main_table_name: str = DEFAULT_MAIN_TABLE,
    join_key: str = DEFAULT_JOIN_KEY,
    tables_to_inspect: Optional[List[str]] = None,
    verbose: bool = True
) -> Tuple[Dict[str, Any], str]:
    """
    Hàm điều phối chẩn đoán dữ liệu tổng quát cho toàn bộ các bảng trong hệ thống.

    Parameters:
    -----------
    data : Dict[str, pd.DataFrame] | str | Path, optional
        - Nếu là Dict: Chứa {tên_bảng: DataFrame}.
        - Nếu là str / Path: Đường dẫn tới thư mục chứa các tệp CSV.
        - Nếu None: Tự động dò tìm thư mục dữ liệu 'data/raw' hoặc 'Dataset - Home Credit Default Risk'.
    output_file : str | Path, optional
        Đường dẫn file text để lưu báo cáo. Mặc định lưu tại 'data_diagnostic_report.txt'.
    main_table_name : str, default 'application_train'
        Tên bảng chính cần kiểm tra Grain và dùng làm mốc so khớp khóa ngoại.
    join_key : str, default 'SK_ID_CURR'
        Tên khóa chính chung.
    tables_to_inspect : List[str], optional
        Danh sách các bảng cụ thể cần khảo sát. Mặc định khảo sát 6 bảng cốt lõi.
    verbose : bool, default True
        Nếu True, in kết quả ra màn hình console.

    Returns:
    --------
    Tuple[Dict[str, Any], str]:
        - Dict chứa dữ liệu chi tiết kết quả chẩn đoán của từng bảng.
        - String toàn bộ nội dung báo cáo đã định dạng đẹp mắt.
    """
    t_start = time.time()
    is_dict_input = isinstance(data, dict)
    data_dir: Optional[Path] = None
    csv_files: Dict[str, Path] = {}
    data_source_desc = ""

    if is_dict_input:
        table_names = list(data.keys())
        data_source_desc = f"Bộ nhớ RAM (Dict gồm {len(table_names)} DataFrames)"
    else:
        data_dir = find_dataset_directory(data)
        data_source_desc = f"Thư mục đĩa: '{data_dir}'"

        target_tables = tables_to_inspect or EXPECTED_TABLES
        for tbl in target_tables:
            p_csv = data_dir / f"{tbl}.csv"
            if p_csv.exists():
                csv_files[tbl] = p_csv
            else:
                for actual_f in data_dir.glob("*.csv"):
                    if actual_f.stem.lower() == tbl.lower():
                        csv_files[tbl] = actual_f
                        break

        # Nếu không khớp bảng nào trong danh sách chuẩn, nạp tất cả các file CSV trong thư mục
        if not csv_files:
            for actual_f in data_dir.glob("*.csv"):
                csv_files[actual_f.stem] = actual_f

        if not csv_files:
            raise FileNotFoundError(f"Không tìm thấy tệp CSV nào trong thư mục: {data_dir}")

        table_names = list(csv_files.keys())

    # Đảm bảo bảng chính được xử lý đầu tiên để thu thập tập khóa SK_ID_CURR
    actual_main_table = main_table_name
    if actual_main_table not in table_names:
        for t in table_names:
            if "application_train" in t.lower():
                actual_main_table = t
                break

    if actual_main_table in table_names:
        ordered_tables = [actual_main_table] + [t for t in table_names if t != actual_main_table]
    else:
        ordered_tables = table_names

    if verbose:
        print("=" * 85)
        print("🚀 BẮT ĐẦU QUY TRÌNH CHẨN ĐOÁN DỮ LIỆU & SCHEMA TOÀN DIỆN")
        print(f"• Nguồn dữ liệu: {data_source_desc}")
        print(f"• Số bảng khảo sát: {len(ordered_tables)} bảng -> {ordered_tables}")
        print("=" * 85)

    table_audits: Dict[str, Dict[str, Any]] = {}
    main_train_curr_ids: Optional[Set[int]] = None

    # Duyệt và kiểm toán từng bảng một (Streaming để tối ưu RAM)
    for tbl_name in ordered_tables:
        if verbose:
            print(f"\n⏳ Đang kiểm toán bảng: [{tbl_name}] ...")

        file_size_mb = None
        if is_dict_input:
            df = data[tbl_name]
            should_cleanup = False
        else:
            p_csv = csv_files[tbl_name]
            file_size_mb = p_csv.stat().st_size / (1024 * 1024)
            t_load = time.time()
            df = pd.read_csv(p_csv, low_memory=False)
            if verbose:
                print(f"   ✓ Nạp xong từ đĩa ({file_size_mb:.2f} MB) trong {time.time()-t_load:.2f}s | Shape: {df.shape}")
            should_cleanup = True

        is_main = (tbl_name == actual_main_table)

        # Chẩn đoán chi tiết bảng
        audit_res = audit_single_table_health(
            tbl_name=tbl_name,
            df=df,
            main_keys=main_train_curr_ids,
            is_main_table=is_main,
            join_key=join_key,
            file_size_mb=file_size_mb
        )
        table_audits[tbl_name] = audit_res

        # Nếu là bảng chính, lưu lại tập khóa duy nhất để đối chiếu khóa ngoại cho các bảng phụ
        if is_main and audit_res["has_join_key"]:
            try:
                main_train_curr_ids = set(df[join_key].dropna().astype(int).unique())
                if verbose:
                    print(f"   ✓ Đã thu thập {len(main_train_curr_ids):,d} mã {join_key} duy nhất từ bảng chính [{tbl_name}]")
            except Exception as e:
                if verbose:
                    print(f"   ⚠️ Không thể thu thập mã {join_key} dạng int: {e}")

        if verbose:
            print(f"   ✓ Hoàn tất chẩn đoán [{tbl_name}] | RAM: {audit_res['memory_mb']:.2f} MB | Trùng lặp: {audit_res['dup_rows']} dòng")
            if audit_res["suspected_contaminated_cols"]:
                print(f"   🚨 Phát hiện {len(audit_res['suspected_contaminated_cols'])} cột số bị ép sang `object` do chuỗi bẩn!")
            if audit_res["orphaned_keys_count"] > 0:
                print(f"   🚨 Phát hiện {audit_res['orphaned_keys_count']:,d} mã khóa ngoại mồ côi ({audit_res['orphaned_rows_count']:,d} dòng)!")

        # Giải phóng bộ nhớ RAM tức thì
        if should_cleanup:
            del df
            gc.collect()

    # Phân tích quan hệ ghép nối & Cardinality
    if verbose:
        print("\n⏳ Đang phân tích quan hệ ghép nối đa bảng & Cardinality...")
    merge_analysis = analyze_merging_conditions(
        table_audits=table_audits,
        main_table_name=actual_main_table,
        join_key=join_key
    )

    # Tạo nội dung báo cáo hoàn chỉnh
    report_text = format_diagnostic_report(
        table_audits=table_audits,
        merge_analysis=merge_analysis,
        data_source_desc=data_source_desc
    )

    # Lưu ra tệp báo cáo
    if output_file is None:
        output_file = Path("data_diagnostic_report.txt")

    out_p = Path(output_file).resolve()
    try:
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            f.write(report_text)
        if verbose:
            print(f"\n✓ ĐÃ LƯU BÁO CÁO CHẨN ĐOÁN TOÀN DIỆN VÀO: {out_p}")
    except Exception as e:
        if verbose:
            print(f"\n⚠️ Không thể ghi báo cáo vào {out_p}: {e}")

    if verbose:
        print(f"\n⏱️ Tổng thời gian chẩn đoán: {time.time()-t_start:.2f} giây.")
        print("=" * 85)
        try:
            print(report_text)
        except UnicodeEncodeError:
            enc = sys.stdout.encoding or "utf-8"
            print(report_text.encode(enc, errors="replace").decode(enc))

    diagnostic_summary = {
        "table_audits": table_audits,
        "merge_analysis": merge_analysis,
        "total_tables": len(table_audits),
        "execution_time_seconds": time.time() - t_start,
        "output_file": str(out_p)
    }

    return diagnostic_summary, report_text


# =============================================================================
# GIAO DIỆN DÒNG LỆNH (CLI INTERFACE)
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="PRJ-01 - Hệ Thống Chẩn Đoán Dữ Liệu & Schema Tổng Quát (Không Cần old_data)"
    )
    parser.add_argument(
        "--data", "-d",
        type=str,
        default=None,
        help="Đường dẫn đến thư mục chứa các tệp CSV cần chẩn đoán (Mặc định tự tìm 'data/raw')."
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="data_diagnostic_report.txt",
        help="Đường dẫn tệp text lưu trữ báo cáo chẩn đoán (Mặc định: 'data_diagnostic_report.txt')."
    )
    parser.add_argument(
        "--main-table", "-m",
        type=str,
        default=DEFAULT_MAIN_TABLE,
        help="Tên bảng chính cần khảo sát grain (Mặc định: 'application_train')."
    )
    parser.add_argument(
        "--join-key", "-k",
        type=str,
        default=DEFAULT_JOIN_KEY,
        help="Tên khóa chính chung (Mặc định: 'SK_ID_CURR')."
    )

    args = parser.parse_args()

    diagnose_dataset(
        data=args.data,
        output_file=args.output,
        main_table_name=args.main_table,
        join_key=args.join_key,
        verbose=True
    )


if __name__ == "__main__":
    main()
