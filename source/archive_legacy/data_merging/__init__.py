"""
Package: source.data_merging
Dự án: PRJ-01 - Dự đoán Rủi ro Vỡ nợ Tín dụng (Home Credit Default Risk)
Mô tả: Package chuyên biệt thực thi quy trình Ghép nối Dữ liệu Đa bảng (Data Merging):
       1. inspect_schema_for_merge.py : Khảo sát schema, cardinality, phát hiện xung đột tên cột
       2. aggregate_secondary_table.py: Nén 1:N về 1:1, One-Hot Encoding, tính toán thống kê đa chiều
       3. flatten_aggregated_df.py    : Làm phẳng MultiIndex, gắn tiền tố, bảo toàn khóa SK_ID_CURR
       4. safe_merge.py               : Ghép nối an toàn 1:1, chốt chặn row explosion và xung đột _x, _y
       5. merge_pipeline.py           : Pipeline ghép nối tuần tự 5 bảng phụ trong RAM
       6. build_train_merged.py       : Kịch bản thực thi trọn gói streaming 5 bảng CSV ra Parquet
       7. eda_structural_summary.py   : Khám phá cấu trúc sơ bộ tệp train_merged_final.parquet
"""

from .inspect_schema_for_merge import inspect_schema_for_merge
from .aggregate_secondary_table import aggregate_secondary_table
from .flatten_aggregated_df import flatten_aggregated_df
from .safe_merge import safe_merge
from .merge_pipeline import sequential_merge_pipeline
from .build_train_merged import run_build_train_merged
from .eda_structural_summary import explore_merged_dataset

__all__ = [
    "inspect_schema_for_merge",
    "aggregate_secondary_table",
    "flatten_aggregated_df",
    "safe_merge",
    "sequential_merge_pipeline",
    "run_build_train_merged",
    "explore_merged_dataset",
]
