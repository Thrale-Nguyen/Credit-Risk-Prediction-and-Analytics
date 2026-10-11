"""
Module: inspect_schema_for_merge.py
Package: source.data_merging
Dự án: PRJ-01 - Dự đoán Rủi ro Vỡ nợ Tín dụng (Home Credit Default Risk)
Mô tả: Khảo sát cấu trúc schema của 6 DataFrames trước khi thực hiện ghép nối (Merge):
       - Kích thước, kiểu dữ liệu, tỷ lệ null của từng bảng.
       - Xác định cặp khóa ghép nối (Join Keys) và kiểm tra cardinality (1:1, 1:N).
       - Tự động phát hiện xung đột tên cột ngoài khóa chính.
       - Xuất kết quả báo cáo dạng text ngắn gọn, dễ dàng copy/tra cứu.
"""

import os
import gc
import sys
from pathlib import Path
from typing import Dict, Union, Optional, Any, List
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


def inspect_schema_for_merge(
    data: Optional[Union[Dict[str, pd.DataFrame], str, Path]] = None,
    output_file: Optional[Union[str, Path]] = None,
    main_table_name: str = "application_train",
    join_key: str = "SK_ID_CURR",
    verbose: bool = True
) -> str:
    """
    Duyệt qua các DataFrame (nhận từ Dict hoặc đọc từ thư mục CSV) và lập báo cáo schema.

    Parameters:
    -----------
    data : Dict[str, pd.DataFrame] | str | Path, optional
        - Nếu là Dict: Chứa {tên_bảng: DataFrame}.
        - Nếu là str / Path: Đường dẫn tới thư mục chứa các tệp CSV.
        - Nếu None: Tự động dò tìm thư mục 'Dataset - Home Credit Default Risk'.
    output_file : str | Path, optional
        Đường dẫn file text để lưu báo cáo. Mặc định lưu tại 'source/data_merging/schema_report.txt'.
    main_table_name : str, default 'application_train'
        Tên bảng chính cần ghép nối các bảng phụ vào.
    join_key : str, default 'SK_ID_CURR'
        Tên khóa chính chung dùng để liên kết bảng.
    verbose : bool, default True
        Nếu True, in kết quả báo cáo ra console.

    Returns:
    --------
    str : Toàn bộ nội dung báo cáo schema dạng text đã được định dạng rõ ràng, ngắn gọn.
    """
    # ---------------------------------------------------------
    # 1. Xác định nguồn dữ liệu đầu vào
    # ---------------------------------------------------------
    is_dict_input = isinstance(data, dict)
    data_dir: Optional[Path] = None
    csv_files: Dict[str, Path] = {}

    if not is_dict_input:
        if data is None:
            # Tự động tìm thư mục dữ liệu
            candidate_paths = [
                Path("Dataset - Home Credit Default Risk"),
                Path("../Dataset - Home Credit Default Risk"),
                Path("PRJ-01/Dataset - Home Credit Default Risk"),
                Path(__file__).resolve().parent.parent.parent / "Dataset - Home Credit Default Risk" if "__file__" in globals() else None,
                Path(__file__).resolve().parent.parent / "Dataset - Home Credit Default Risk" if "__file__" in globals() else None,
            ]
            for p in candidate_paths:
                if p and p.exists() and p.is_dir():
                    data_dir = p.resolve()
                    break
            if data_dir is None:
                raise FileNotFoundError(
                    "Không tìm thấy thư mục 'Dataset - Home Credit Default Risk'. "
                    "Vui lòng truyền đường dẫn thư mục vào tham số `data` hoặc truyền dict các DataFrame."
                )
        else:
            data_dir = Path(data).resolve()
            if not data_dir.exists():
                raise FileNotFoundError(f"Đường dẫn không tồn tại: {data_dir}")

        expected_tables = [
            "application_train",
            "bureau",
            "previous_application",
            "POS_CASH_balance",
            "installments_payments",
            "credit_card_balance"
        ]
        for tbl in expected_tables:
            p_csv = data_dir / f"{tbl}.csv"
            if p_csv.exists():
                csv_files[tbl] = p_csv
            else:
                for actual_f in data_dir.glob("*.csv"):
                    if actual_f.stem.lower() == tbl.lower():
                        csv_files[tbl] = actual_f
                        break

        if not csv_files:
            raise FileNotFoundError(f"Không tìm thấy các tệp CSV cần thiết trong thư mục: {data_dir}")

    # ---------------------------------------------------------
    # 2. Khởi tạo cấu trúc thu thập thông tin
    # ---------------------------------------------------------
    schema_info: Dict[str, Dict[str, Any]] = {}
    table_names = list(data.keys()) if is_dict_input else list(csv_files.keys())

    if main_table_name not in table_names:
        for t in table_names:
            if "application_train" in t.lower():
                main_table_name = t
                break

    # ---------------------------------------------------------
    # 3. Duyệt và trích xuất siêu dữ liệu từng bảng
    # ---------------------------------------------------------
    for tbl_name in table_names:
        if verbose:
            print(f"-> Đang phân tích schema bảng: {tbl_name} ...")

        if is_dict_input:
            df = data[tbl_name]
            should_cleanup = False
        else:
            df = pd.read_csv(csv_files[tbl_name])
            should_cleanup = True

        n_rows, n_cols = df.shape
        mem_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)

        has_join_key = join_key in df.columns
        key_dtype = str(df[join_key].dtype) if has_join_key else "N/A"
        key_unique = int(df[join_key].nunique()) if has_join_key else 0
        key_nulls = int(df[join_key].isnull().sum()) if has_join_key else 0

        null_counts = df.isnull().sum()
        cols_with_null = (null_counts > 0).sum()
        max_null_pct = (null_counts.max() / n_rows * 100) if n_rows > 0 else 0
        max_null_col = null_counts.idxmax() if n_rows > 0 and cols_with_null > 0 else "None"

        dtype_summary: Dict[str, int] = {}
        for dt, count in df.dtypes.value_counts().items():
            dtype_summary[str(dt)] = int(count)

        schema_info[tbl_name] = {
            "rows": n_rows,
            "cols": n_cols,
            "memory_mb": mem_mb,
            "columns": list(df.columns),
            "dtypes": dtype_summary,
            "has_key": has_join_key,
            "key_dtype": key_dtype,
            "key_unique": key_unique,
            "key_nulls": key_nulls,
            "cols_with_null": int(cols_with_null),
            "max_null_col": max_null_col,
            "max_null_pct": max_null_pct
        }

        if should_cleanup:
            del df
            gc.collect()

    # ---------------------------------------------------------
    # 4. Phân tích xung đột và Khóa liên kết với bảng chính
    # ---------------------------------------------------------
    main_cols = set(schema_info[main_table_name]["columns"]) if main_table_name in schema_info else set()
    main_rows = schema_info[main_table_name]["rows"] if main_table_name in schema_info else 0
    main_key_uniq = schema_info[main_table_name]["key_unique"] if main_table_name in schema_info else 0

    join_analysis: List[Dict[str, Any]] = []

    for tbl_name in table_names:
        if tbl_name == main_table_name:
            continue

        sec_info = schema_info[tbl_name]
        sec_cols = set(sec_info["columns"])

        common_cols = sorted(list(main_cols.intersection(sec_cols)))
        conflicting_cols = [c for c in common_cols if c != join_key]

        sec_rows = sec_info["rows"]
        sec_uniq = sec_info["key_unique"]

        if sec_uniq == 0:
            cardinality = "KHÔNG CÓ KHÓA"
            merge_risk = "Không thể ghép trực tiếp (Thiếu khóa ghép nối)"
        elif sec_rows == sec_uniq:
            cardinality = "1 : 1 (Độc nhất)"
            merge_risk = "An toàn - Ghép trực tiếp không làm tăng số dòng"
        else:
            cardinality = f"1 : N (Nhiều dòng / 1 khách hàng: trung bình {sec_rows/sec_uniq:.1f} dòng)"
            merge_risk = "NGUY HIỂM: Gây bùng nổ số dòng (Fan-out)! Bắt buộc phải Aggregate trước khi Merge."

        join_analysis.append({
            "table": tbl_name,
            "join_key": join_key if sec_info["has_key"] else "THIẾU",
            "key_match_type": (sec_info["key_dtype"] == schema_info[main_table_name]["key_dtype"]) if sec_info["has_key"] else False,
            "cardinality": cardinality,
            "sec_rows": sec_rows,
            "sec_uniq": sec_uniq,
            "conflicting_cols": conflicting_cols,
            "merge_risk": merge_risk
        })

    # ---------------------------------------------------------
    # 5. Định dạng báo cáo văn bản có cấu trúc chuẩn
    # ---------------------------------------------------------
    sep_double = "=" * 85
    sep_single = "-" * 85

    lines: List[str] = []
    lines.append(sep_double)
    lines.append("BÁO CÁO PHÂN TÍCH SCHEMA & ĐIỀU KIỆN GHÉP NỐI (DATA MERGING SCHEMA AUDIT)")
    lines.append(sep_double)
    lines.append(f"• Bảng chính (Main DataFrame)       : {main_table_name}")
    lines.append(f"• Khóa ghép nối mục tiêu (Join Key) : {join_key}")
    lines.append(f"• Tổng số bảng khảo sát             : {len(table_names)} bảng")
    lines.append(sep_single)

    lines.append("\n[MỤC 1] TỔNG QUAN KÍCH THƯỚC, BỘ NHỚ & TỶ LỆ KHUYẾT THIẾU")
    lines.append(sep_single)
    header1 = f"{'Tên Bảng':<26} | {'Số Dòng':>12} | {'Số Cột':>6} | {'RAM (MB)':>9} | {'Cột Null':>8} | {'Max Null %':>10}"
    lines.append(header1)
    lines.append(sep_single)
    for tbl_name in table_names:
        inf = schema_info[tbl_name]
        lines.append(
            f"{tbl_name:<26} | {inf['rows']:>12,d} | {inf['cols']:>6d} | {inf['memory_mb']:>9.2f} | "
            f"{inf['cols_with_null']:>8d} | {inf['max_null_pct']:>9.1f}%"
        )
    lines.append(sep_single)

    lines.append("\n[MỤC 2] KHẢO SÁT CẶP KHÓA GHÉP NỐI & BẢN SỐ QUAN HỆ (CARDINALITY)")
    lines.append(sep_single)
    lines.append(f"• Khóa trên bảng chính [{main_table_name}]: {main_key_uniq:,d} unique / {main_rows:,d} dòng -> Chuẩn 1 dòng = 1 khách hàng")
    lines.append(sep_single)

    for item in join_analysis:
        lines.append(f"▶ Cặp ghép: [{main_table_name}] <---> [{item['table']}]")
        lines.append(f"   - Khóa ghép nối       : '{item['join_key']}' (Khớp dtype: {'CÓ' if item['key_match_type'] else 'KHÔNG'})")
        lines.append(f"   - Bản số quan hệ      : {item['cardinality']}")
        lines.append(f"   - Hồ sơ khách hàng    : {item['sec_uniq']:,d} unique keys trên {item['sec_rows']:,d} dòng")
        lines.append(f"   - Đánh giá rủi ro     : {item['merge_risk']}")
        lines.append("")

    lines.append(sep_single)
    lines.append("[MỤC 3] PHÁT HIỆN XUNG ĐỘT TÊN CỘT NGOÀI KHÓA CHÍNH (COLUMN NAME COLLISIONS)")
    lines.append(sep_single)

    has_collision = False
    for item in join_analysis:
        confs = item["conflicting_cols"]
        if confs:
            has_collision = True
            lines.append(f"⚠️  Bảng [{item['table']}] trùng {len(confs)} cột với [{main_table_name}]:")
            lines.append(f"   -> Cột trùng: {confs}")
            lines.append("   -> Khuyến nghị: Cần gắn tiền tố (prefix) trước khi ghép để tránh sinh hậu tố `_x`, `_y`.\n")

    if not has_collision:
        lines.append("✓ Không phát hiện xung đột tên cột ngoài khóa chính giữa các bảng phụ và bảng chính.\n")

    lines.append(sep_double)
    lines.append("TỔNG KẾT KHUYẾN NGHỊ KỸ THUẬT CHO PIPELINE GHÉP NỐI:")
    lines.append("1. Khử Trùng Lặp 1:N (Row Explosion):")
    lines.append("   - Toàn bộ 5 bảng phụ ĐỀU LÀ quan hệ 1:N đối với SK_ID_CURR.")
    lines.append("   - TUYỆT ĐỐI KHÔNG LEFT JOIN trực tiếp bảng phụ vào application_train.")
    lines.append("2. Giải Pháp Tổng Hợp (Feature Aggregation):")
    lines.append("   - Với biến định lượng: Tính toán [min, max, mean, sum].")
    lines.append("   - Với biến định tính: Áp dụng One-Hot Encoding sau đó tính [mean, sum].")
    lines.append("   - Sau khi gom về đúng 1 dòng / 1 SK_ID_CURR, mới tiến hành `merge(how='left', on='SK_ID_CURR')` vào bảng chính.")
    lines.append("3. Xử Lý Trùng Tên Cột:")
    lines.append("   - Đặt tiền tố/hậu tố rõ ràng cho các đặc trưng sau aggregation (ví dụ: `BUREAU_AMT_CREDIT_MEAN`, `PREV_AMT_CREDIT_SUM`)")
    lines.append("   - Tránh để phát sinh các cột `_x`, `_y` không rõ ngữ cảnh.")
    lines.append(sep_double)

    report_text = "\n".join(lines)

    # Tự động lưu ra file text
    if output_file is None:
        default_out = Path(__file__).resolve().parent / "schema_report.txt" if "__file__" in globals() else Path("schema_report.txt")
        output_file = default_out

    if output_file:
        out_p = Path(output_file).resolve()
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            f.write(report_text)
        if verbose:
            print(f"✓ Đã lưu kết quả báo cáo vào: {out_p}")

    if verbose:
        try:
            print(report_text)
        except UnicodeEncodeError:
            enc = sys.stdout.encoding or "utf-8"
            print(report_text.encode(enc, errors="replace").decode(enc))

    return report_text


if __name__ == "__main__":
    inspect_schema_for_merge()
