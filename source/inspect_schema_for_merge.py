"""
Module: inspect_schema_for_merge.py
Dự án: PRJ-01 - Dự đoán Rủi ro Vỡ nợ Tín dụng (Home Credit Default Risk)
Mô tả: Hàm phân tích schema, kiểm tra khóa ghép nối (Join Keys), phát hiện xung đột tên cột
       và đưa ra khuyến nghị merge dữ liệu đa bảng tối ưu bộ nhớ.
"""

import os
import gc
import sys
from pathlib import Path
from typing import Dict, Union, Optional, Any, List
import pandas as pd

# Đảm bảo console Windows hỗ trợ in Unicode tiếng Việt mượt mà
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
    main_table: str = "application_train",
    output_file: Optional[Union[str, Path]] = None,
    verbose: bool = True
) -> str:
    """
    Phân tích schema của 6 DataFrames trong bài toán Home Credit Default Risk phục vụ ghép nối (merge).

    Parameters:
    -----------
    data : Dict[str, pd.DataFrame] | str | Path | None
        - Nếu là dict: mapping {tên_bảng: DataFrame} đã load sẵn trong bộ nhớ.
        - Nếu là str hoặc Path: đường dẫn tới thư mục chứa các tệp CSV.
        - Nếu là None: tự động dò tìm thư mục 'Dataset - Home Credit Default Risk'.
    main_table : str, default 'application_train'
        Tên bảng chính làm gốc để phân tích quan hệ ghép nối.
    output_file : str | Path | None, optional
        Đường dẫn tệp text để lưu kết quả (tiện copy-paste).
    verbose : bool, default True
        Nếu True, in kết quả ra màn hình/console.

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
                Path(__file__).resolve().parent.parent / "Dataset - Home Credit Default Risk" if "__file__" in globals() else None
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

        # Tìm các file CSV trong thư mục
        for f in data_dir.glob("*.csv"):
            table_name = f.stem
            csv_files[table_name] = f

        if not csv_files:
            raise ValueError(f"Không tìm thấy tệp .csv nào trong thư mục: {data_dir}")

    # Chuẩn hóa tên bảng chính (loại bỏ đuôi .csv nếu có)
    main_table_clean = main_table.replace(".csv", "").strip()

    # ---------------------------------------------------------
    # 2. Thu thập siêu dữ liệu (Schema Metadata) từng bảng
    # ---------------------------------------------------------
    schema_info = {}
    # schema_info[tbl] = {
    #     'shape': (rows, cols),
    #     'memory_mb': float,
    #     'columns': {
    #          col: {'dtype': str, 'nunique': int, 'null_count': int, 'null_pct': float}
    #      }
    # }

    if is_dict_input:
        table_names = list(data.keys())
    else:
        table_names = list(csv_files.keys())

    # Đưa bảng chính lên đầu danh sách nếu có
    if main_table_clean in table_names:
        table_names.remove(main_table_clean)
        table_names.insert(0, main_table_clean)

    for tbl_name in table_names:
        if is_dict_input:
            df = data[tbl_name]
            clean_name = tbl_name.replace(".csv", "").strip()
            n_rows, n_cols = df.shape
            mem_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)

            cols_meta = {}
            for col in df.columns:
                n_null = int(df[col].isnull().sum())
                n_uniq = int(df[col].nunique())
                cols_meta[col] = {
                    "dtype": str(df[col].dtype),
                    "nunique": n_uniq,
                    "null_count": n_null,
                    "null_pct": (n_null / n_rows * 100) if n_rows > 0 else 0.0
                }
            schema_info[clean_name] = {
                "shape": (n_rows, n_cols),
                "memory_mb": mem_mb,
                "columns": cols_meta
            }
        else:
            clean_name = tbl_name
            csv_path = csv_files[tbl_name]
            # Đọc file CSV một cách tiết kiệm RAM
            df = pd.read_csv(csv_path)
            n_rows, n_cols = df.shape
            mem_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)

            cols_meta = {}
            for col in df.columns:
                n_null = int(df[col].isnull().sum())
                n_uniq = int(df[col].nunique())
                cols_meta[col] = {
                    "dtype": str(df[col].dtype),
                    "nunique": n_uniq,
                    "null_count": n_null,
                    "null_pct": (n_null / n_rows * 100) if n_rows > 0 else 0.0
                }
            schema_info[clean_name] = {
                "shape": (n_rows, n_cols),
                "memory_mb": mem_mb,
                "columns": cols_meta
            }
            # Giải phóng RAM ngay lập tức
            del df
            gc.collect()

    # ---------------------------------------------------------
    # 3. Phân tích Khóa ghép nối & Xung đột tên cột
    # ---------------------------------------------------------
    main_meta = schema_info.get(main_table_clean)
    if not main_meta:
        available_tables = list(schema_info.keys())
        raise KeyError(
            f"Không tìm thấy bảng chính '{main_table_clean}'. Các bảng khả dụng: {available_tables}"
        )

    main_cols = main_meta["columns"]
    main_rows = main_meta["shape"][0]

    # Phân tích quan hệ với từng bảng phụ
    join_keys_analysis = []
    col_collisions = []

    for sec_tbl, sec_meta in schema_info.items():
        if sec_tbl == main_table_clean:
            continue

        sec_cols = sec_meta["columns"]
        sec_rows = sec_meta["shape"][0]

        # Tìm các cột giao nhau
        common_cols = sorted(list(set(main_cols.keys()).intersection(set(sec_cols.keys()))))

        tbl_join_keys = []
        tbl_conflicts = []

        for col in common_cols:
            main_dtype = main_cols[col]["dtype"]
            sec_dtype = sec_cols[col]["dtype"]
            dtype_match = (main_dtype == sec_dtype)

            main_uniq = main_cols[col]["nunique"]
            sec_uniq = sec_cols[col]["nunique"]

            # Phán đoán có phải Khóa ghép nối hay không (thường chứa ID, SK_ID)
            is_id_pattern = any(k in col.upper() for k in ["SK_ID", "ID"])

            # Xác định bản số (Cardinality)
            is_main_unique = (main_uniq == main_rows)
            is_sec_unique = (sec_uniq == sec_rows)

            if is_main_unique and is_sec_unique:
                cardinality = "1 : 1"
            elif is_main_unique and not is_sec_unique:
                cardinality = "1 : N (Cần Aggregate trước khi merge)"
            elif not is_main_unique and is_sec_unique:
                cardinality = "N : 1"
            else:
                cardinality = "N : N (Nguy cơ Row Explosion cực cao!)"

            if is_id_pattern:
                tbl_join_keys.append({
                    "col": col,
                    "main_dtype": main_dtype,
                    "sec_dtype": sec_dtype,
                    "dtype_match": dtype_match,
                    "cardinality": cardinality,
                    "main_uniq": main_uniq,
                    "sec_uniq": sec_uniq
                })
            else:
                tbl_conflicts.append({
                    "col": col,
                    "main_dtype": main_dtype,
                    "sec_dtype": sec_dtype,
                    "dtype_match": dtype_match
                })

        join_keys_analysis.append({
            "table": sec_tbl,
            "shape": sec_meta["shape"],
            "join_keys": tbl_join_keys,
            "non_key_overlaps": tbl_conflicts
        })

    # Kiểm tra trùng tên chéo giữa các bảng phụ (Secondary vs Secondary)
    sec_tables = [t for t in schema_info.keys() if t != main_table_clean]
    cross_table_overlaps = []
    for i in range(len(sec_tables)):
        for j in range(i + 1, len(sec_tables)):
            t1, t2 = sec_tables[i], sec_tables[j]
            cols1 = set(schema_info[t1]["columns"].keys())
            cols2 = set(schema_info[t2]["columns"].keys())
            shared = sorted(list(cols1.intersection(cols2)))
            if shared:
                cross_table_overlaps.append((t1, t2, shared))

    # ---------------------------------------------------------
    # 4. Xây dựng Báo cáo dạng Text rõ ràng, dễ đọc
    # ---------------------------------------------------------
    lines = []
    sep_double = "=" * 85
    sep_single = "-" * 85

    lines.append(sep_double)
    lines.append("BÁO CÁO PHÂN TÍCH SCHEMA & KHÓA GHÉP NỐI (HOME CREDIT DEFAULT RISK)")
    lines.append(sep_double)
    lines.append(f"• Bảng chính (Main DataFrame): {main_table_clean}")
    lines.append(f"• Số lượng bảng phân tích: {len(schema_info)}")
    lines.append("")

    # Phần 1: Tóm tắt Kích thước các Bảng
    lines.append("1. TỔNG QUAN KÍCH THƯỚC (SHAPE & MEMORY)")
    lines.append(sep_single)
    lines.append(f"{'STT':<4} | {'Tên Bảng':<26} | {'Số Dòng':>12} | {'Số Cột':>8} | {'Dung Lượng (MB)':>16}")
    lines.append(sep_single)
    for idx, (tbl_name, meta) in enumerate(schema_info.items(), 1):
        rows, cols = meta["shape"]
        mem = meta["memory_mb"]
        is_main_flag = " (MAIN)" if tbl_name == main_table_clean else ""
        lines.append(f"{idx:<4} | {tbl_name + is_main_flag:<26} | {rows:>12,d} | {cols:>8,d} | {mem:>16.2f} MB")
    lines.append(sep_single)
    lines.append("")

    # Phần 2: Cặp Khóa ghép nối giữa Bảng chính và 5 Bảng phụ
    lines.append("2. PHÂN TÍCH CẶP KHÓA GHÉP NỐI (JOIN KEYS) VỚI BẢNG CHÍNH")
    lines.append(sep_single)
    for item in join_keys_analysis:
        tbl = item["table"]
        rows, cols = item["shape"]
        keys = item["join_keys"]
        lines.append(f"▶ Bảng chính [{main_table_clean}]  <--->  Bảng phụ [{tbl}] ({rows:,d} dòng, {cols} cột):")
        if keys:
            for k in keys:
                type_status = "Đồng nhất" if k["dtype_match"] else f"KHÁC BIỆT ({k['main_dtype']} vs {k['sec_dtype']})"
                lines.append(f"   • Khóa ghép: `{k['col']}`")
                lines.append(f"     - Kiểu dữ liệu: Main=[{k['main_dtype']}], Sub=[{k['sec_dtype']}] -> {type_status}")
                lines.append(f"     - Số unique : Main={k['main_uniq']:,d} / Sub={k['sec_uniq']:,d}")
                lines.append(f"     - Bản số (Cardinality): {k['cardinality']}")
        else:
            lines.append("   ⚠️ CẢNH BÁO: Không phát hiện khóa ID chung trực tiếp với bảng chính!")
        lines.append("")

    # Phần 3: Cảnh báo Xung đột Tên Cột (Non-Key Column Collisions)
    lines.append("3. KIỂM TRA TRÙNG TÊN CỘT (COLUMN OVERLAPS / CONFLICTS)")
    lines.append(sep_single)
    lines.append("A. Xung đột giữa Bảng chính và Bảng phụ (Nguy cơ sinh đuôi _x, _y khi merge):")
    has_conflict = False
    for item in join_keys_analysis:
        tbl = item["table"]
        overlaps = item["non_key_overlaps"]
        if overlaps:
            has_conflict = True
            lines.append(f"   • [{main_table_clean}] vs [{tbl}] trùng {len(overlaps)} cột ngoài khóa:")
            for c in overlaps:
                dtype_info = f"({c['main_dtype']})" if c["dtype_match"] else f"(Main: {c['main_dtype']} vs Sub: {c['sec_dtype']})"
                lines.append(f"     - `{c['col']}` {dtype_info}")
    if not has_conflict:
        lines.append("   ✓ Không phát hiện cột non-key nào bị trùng tên với bảng chính.")
    lines.append("")

    lines.append("B. Cột trùng tên chéo giữa các Bảng phụ (Cần lưu ý khi feature engineering):")
    if cross_table_overlaps:
        for t1, t2, shared in cross_table_overlaps:
            lines.append(f"   • [{t1}] vs [{t2}] trùng {len(shared)} cột: {shared}")
    else:
        lines.append("   ✓ Không có cột trùng chéo giữa các bảng phụ.")
    lines.append("")

    # Phần 4: Chi tiết Schema từng bảng
    lines.append("4. CHI TIẾT SCHEMA TỪNG BẢNG (Tên Cột, Dtype, Unique, Null Count, % Null)")
    lines.append(sep_single)
    for tbl_name, meta in schema_info.items():
        rows, cols = meta["shape"]
        lines.append(f"\n[BẢNG: {tbl_name}] - Shape: ({rows:,d} dòng, {cols} cột)")
        lines.append(f"{'STT':<4} | {'Tên Cột':<32} | {'Dtype':<12} | {'Unique':>10} | {'Số Null':>12} | {'% Null':>8}")
        lines.append("-" * 85)
        for c_idx, (col_name, c_meta) in enumerate(meta["columns"].items(), 1):
            lines.append(
                f"{c_idx:<4} | {col_name:<32} | {c_meta['dtype']:<12} | "
                f"{c_meta['nunique']:>10,d} | {c_meta['null_count']:>12,d} | {c_meta['null_pct']:>7.2f}%"
            )

    lines.append("")
    lines.append(sep_double)
    lines.append("5. KHUYẾN NGHỊ KỸ THUẬT (MERGE STRATEGY RECOMMENDATIONS)")
    lines.append(sep_double)
    lines.append("1. Tránh Nối Trực Tiếp (Zero-Direct-Merge):")
    lines.append("   - Mọi quan hệ giữa application_train và 5 bảng phụ đều là 1 : N (1 hồ sơ có nhiều khoản vay/giao dịch).")
    lines.append("   - Nối trực tiếp bằng `pd.merge()` sẽ gây BÙNG NỔ DÒNG (Row Explosion) từ 307K dòng lên hàng chục triệu dòng!")
    lines.append("2. Quy Tắc Aggregation Bắt Buộc:")
    lines.append("   - Luôn `groupby('SK_ID_CURR')` và tính toán thống kê (min, max, mean, sum, count, std) trên bảng phụ trước.")
    lines.append("   - Sau khi gom về đúng 1 dòng / 1 SK_ID_CURR, mới tiến hành `merge(how='left', on='SK_ID_CURR')` vào bảng chính.")
    lines.append("3. Xử Lý Trùng Tên Cột:")
    lines.append("   - Đặt tiền tố/hậu tố rõ ràng cho các đặc trưng sau aggregation (ví dụ: `BUREAU_AMT_CREDIT_MEAN`, `PREV_AMT_CREDIT_SUM`)")
    lines.append("   - Tránh để phát sinh các cột `_x`, `_y` không rõ ngữ cảnh.")
    lines.append(sep_double)

    report_text = "\n".join(lines)

    # Tự động lưu ra file text nếu không chỉ định, để người dùng dễ copy-paste
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
    # Chạy thử nghiệm trực tiếp từ script
    inspect_schema_for_merge()
