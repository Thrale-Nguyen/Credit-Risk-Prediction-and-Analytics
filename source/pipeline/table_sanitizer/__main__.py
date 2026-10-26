"""
Executable entry point for table_sanitizer package.
Cho phép khởi chạy trực tiếp thông qua:
    python -m source.pipeline.table_sanitizer
"""

from .runner import run_table_sanitizer

if __name__ == "__main__":
    run_table_sanitizer()
