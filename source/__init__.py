"""
Package: source
Dự án: PRJ-01 - Dự đoán Rủi ro Vỡ nợ Tín dụng (Home Credit Default Risk)
"""

# Giai đoạn 2: Ghép nối dữ liệu đa bảng (Data Merging)
from .data_merging import (
    inspect_schema_for_merge,
    aggregate_secondary_table,
    flatten_aggregated_df,
    safe_merge,
    sequential_merge_pipeline,
    run_build_train_merged,
    explore_merged_dataset,
)

# Giai đoạn 3: Tiền xử lý & Làm sạch dữ liệu chuyên biệt cho LightGBM & CatBoost
from .data_cleaning import (
    clean_data_for_trees,
    get_lightgbm_data,
    get_catboost_data,
    TreeCleaningPipeline,
    CATEGORICAL_COLS,
    preprocess_data,
    DataCleaningPipeline,
)

__all__ = [
    # Data Merging
    "inspect_schema_for_merge",
    "aggregate_secondary_table",
    "flatten_aggregated_df",
    "safe_merge",
    "sequential_merge_pipeline",
    "run_build_train_merged",
    "explore_merged_dataset",
    # Data Cleaning for Trees
    "clean_data_for_trees",
    "get_lightgbm_data",
    "get_catboost_data",
    "TreeCleaningPipeline",
    "CATEGORICAL_COLS",
    "preprocess_data",
    "DataCleaningPipeline",
]
