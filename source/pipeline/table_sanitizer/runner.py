"""
Module: runner.py
Package: source.pipeline.table_sanitizer
Mô tả: Hàm điều phối chính (Orchestrator / Runner) cho toàn bộ quy trình làm sạch cấp bảng:
       - Tự động định vị thư mục input/output.
       - Làm sạch bảng chính application_train và trích xuất danh sách khách hàng hợp lệ.
       - Lần lượt làm sạch 5 bảng phụ và khử khóa ngoại mồ côi.
       - Báo cáo tổng kết thống kê số lượng dòng, số cột chuyển đổi và dung lượng lưu trữ.
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pandas as pd

try:
    from .config import CORE_TABLES
    from .utils import configure_windows_console, resolve_paths
    from .application import sanitize_application_train
    from .secondary import sanitize_secondary_table
except (ImportError, ValueError):
    import sys
    _current = Path(__file__).resolve()
    for _parent in _current.parents:
        if (_parent / "source").exists():
            if str(_parent) not in sys.path:
                sys.path.insert(0, str(_parent))
            break
    from source.pipeline.table_sanitizer.config import CORE_TABLES
    from source.pipeline.table_sanitizer.utils import configure_windows_console, resolve_paths
    from source.pipeline.table_sanitizer.application import sanitize_application_train
    from source.pipeline.table_sanitizer.secondary import sanitize_secondary_table


def run_table_sanitizer(
    raw_dir: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """
    Thực thi trọn gói kịch bản Làm Sạch Cấp Bảng Độc Lập cho toàn bộ 6 bảng cốt lõi.
    
    Parameters:
        raw_dir: Đường dẫn thư mục dữ liệu thô (tùy chọn).
        output_dir: Đường dẫn thư mục xuất dữ liệu sạch (tùy chọn).
        
    Returns:
        Dict[str, Any]: Báo cáo trạng thái và bảng thống kê chi tiết.
    """
    configure_windows_console()
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
