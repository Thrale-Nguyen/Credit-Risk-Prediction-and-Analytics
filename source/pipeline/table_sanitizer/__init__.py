"""
Package: source.pipeline.table_sanitizer
Dự án: PRJ-01 - Dự đoán Rủi ro Vỡ nợ Tín dụng (Home Credit Default Risk)
Mô tả: [Bước 1 trong Pipeline Tuyến tính]
       Làm sạch dữ liệu cấp bảng độc lập (Table-level Data Sanitization):
       1. Khử nhiễm 104 cột object bị pha tạp đơn vị tiền tệ/đo lường (đ, VND, USD, kg, %) về số thực.
       2. Chuẩn hóa nhãn TARGET: Lọc sạch 24,885 missing và 48,554 nhãn rác, xác lập cỡ mẫu hợp lệ
          khoảng 234,072 dòng với tỷ lệ nợ xấu chuẩn mực 7.89%.
       3. Khử trùng lặp: Xóa dòng trùng lặp 100% tại POS, Installments, Credit Card.
       4. Lọc khóa ngoại mồ côi (Orphan Foreign Keys): Lọc bỏ triệt để các hồ sơ trong 5 bảng phụ
          không tồn tại trong tập khách hàng hợp lệ của application_train.
       5. Chuẩn hóa missing ẩn (Sentinel values) và cắt bỏ khoảng trắng thừa trên toàn bộ các cột phân loại.
       6. Lưu trữ 6 bảng đã được chuẩn hóa vào thư mục `data/processed/*.parquet`.
"""

from .config import (
    UNIT_CLEAN_REGEX,
    HIDDEN_MISSING_STRINGS,
    SENTINEL_TEXT,
    NUMERIC_KEYWORDS,
    CORE_TABLES,
    TEXT_MAPPING_RULES,
)
from .utils import (
    configure_windows_console,
    resolve_paths,
)
from .column_sanitizer import (
    sanitize_column_series,
)
from .application import (
    sanitize_application_train,
)
from .secondary import (
    sanitize_secondary_table,
)
from .runner import (
    run_table_sanitizer,
)

__all__ = [
    "run_table_sanitizer",
    "sanitize_application_train",
    "sanitize_secondary_table",
    "sanitize_column_series",
    "resolve_paths",
    "configure_windows_console",
    "CORE_TABLES",
    "UNIT_CLEAN_REGEX",
    "HIDDEN_MISSING_STRINGS",
    "SENTINEL_TEXT",
    "NUMERIC_KEYWORDS",
    "TEXT_MAPPING_RULES",
]
