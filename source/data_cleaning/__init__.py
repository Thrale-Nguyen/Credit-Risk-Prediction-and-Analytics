"""
Package: source.data_cleaning
Dự án: PRJ-01 - Dự đoán Rủi ro Vỡ nợ Tín dụng (Home Credit Default Risk)
Mô tả: Package chuyên biệt Tiền xử lý & Làm sạch dữ liệu tối ưu cho LightGBM & CatBoost:
       1. downcast.py           : Safe Dynamic Downcasting tối ưu RAM và chống tràn số int64.
       2. anomalies.py          : Xử lý dị thường DAYS_EMPLOYED = 365243 & dương hóa các cột thời gian DAYS_*.
       3. feature_filters.py    : Lọc bỏ đặc trưng kém chất lượng (>95% null, zero variance).
       4. business_rules.py     : Quy tắc nghiệp vụ thực tế cho AMT_INCOME_TOTAL (bỏ Winsorization 99%).
       5. structural_imputer.py : Khuyết thiếu có cấu trúc nhóm CC_ & bảo toàn Native Missing (np.nan).
       6. tree_adapters.py      : Adapter get_lightgbm_data() và get_catboost_data() chuẩn định dạng.
       7. pipeline.py           : Pipeline điều phối hoàn chỉnh clean_data_for_trees và TreeCleaningPipeline.
"""

from .downcast import safe_downcast_numeric
from .anomalies import handle_business_anomalies
from .feature_filters import filter_low_quality_features
from .business_rules import handle_business_outliers
from .outlier_handler import handle_financial_outliers
from .structural_imputer import handle_structural_missing
from .missing_imputer import handle_missing_data
from .tree_adapters import (
    CATEGORICAL_COLS,
    get_lightgbm_data,
    get_catboost_data,
    detect_categorical_columns,
)
from .pipeline import (
    clean_data_for_trees,
    TreeCleaningPipeline,
    preprocess_data,
    DataCleaningPipeline,
)

__all__ = [
    "safe_downcast_numeric",
    "handle_business_anomalies",
    "filter_low_quality_features",
    "handle_business_outliers",
    "handle_financial_outliers",
    "handle_structural_missing",
    "handle_missing_data",
    "CATEGORICAL_COLS",
    "get_lightgbm_data",
    "get_catboost_data",
    "detect_categorical_columns",
    "clean_data_for_trees",
    "TreeCleaningPipeline",
    "preprocess_data",
    "DataCleaningPipeline",
]
