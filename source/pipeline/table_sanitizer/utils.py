"""
Module: utils.py
Package: source.pipeline.table_sanitizer
Mô tả: Các tiện ích hạ tầng: xác định đường dẫn thư mục và cấu hình console.
"""

import sys
from pathlib import Path
from typing import Optional, Tuple, Union


def configure_windows_console() -> None:
    """Thiết lập encoding UTF-8 cho Windows console để tránh lỗi in ký tự tiếng Việt/ký tự đặc biệt."""
    if sys.platform.startswith("win"):
        try:
            if sys.stdout and hasattr(sys.stdout, "reconfigure"):
                sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            if sys.stderr and hasattr(sys.stderr, "reconfigure"):
                sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def resolve_paths(
    raw_dir: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None
) -> Tuple[Path, Path]:
    """
    Xác định đường dẫn thư mục raw đầu vào và thư mục processed đầu ra.
    
    Parameters:
        raw_dir: Đường dẫn thư mục dữ liệu thô (nếu None sẽ tự dò tìm data/raw).
        output_dir: Đường dẫn thư mục lưu parquet sạch (nếu None sẽ đặt tại data/processed).
        
    Returns:
        Tuple[Path, Path]: (raw_path, output_path)
    """
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
