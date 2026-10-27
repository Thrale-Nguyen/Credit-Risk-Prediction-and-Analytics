"""
Module: config.py
Package: source.pipeline.table_sanitizer
Mô tả: Các hằng số, biểu thức chính quy (Regex), danh mục missing ẩn
       và quy tắc chuẩn hóa cho quy trình làm sạch cấp bảng.
"""

import re

# Regex bóc tách đơn vị tiền tệ, đo lường và khoảng trắng rác (giữ lại dấu phẩy để xử lý số thập phân)
UNIT_CLEAN_REGEX = re.compile(r"[₫đVNDUSDkg%\s\xa0]", re.IGNORECASE)

# Danh mục các ký hiệu missing ẩn / sentinel values dạng chuỗi
HIDDEN_MISSING_STRINGS = {
    "?", "n/a", "na", "nan", "null", "none", "missing", "unknown",
    "#n/a", "#na", "n.a.", "n/k", "nil", "void", "empty",
    "-", "--", "---", ".", "", " ", "\t", "\n", "\xa0"
}

# Các từ khóa missing ẩn ngôn ngữ tự nhiên / biến thể Home Credit
SENTINEL_TEXT = {"chưa rõ", "chua ro", "chua ro -", "chưa rõ -", "xna", "xap"}

# Tiền tố và từ khóa nhận diện cột số đặc thù Home Credit
NUMERIC_KEYWORDS = (
    "AMT_", "CNT_", "DAYS_", "MONTHS_", "RATE_", "FLAG_", "OBS_", "DEF_",
    "FLOORS_", "YEARS_", "APARTMENTS_", "ELEVATORS_", "ENTRANCES_", "LIVINGAREA_",
    "NONLIVINGAREA_", "EXT_SOURCE_", "TOTALAREA_", "COMMONAREA_", "BASEMENTAREA_",
    "LANDAREA_", "OWN_CAR_AGE", "REGION_", "HOUR_APPR_PROCESS_START", "SCORE",
    "NUM_", "SK_ID", "SK_DPD", "SELLERPLACE_AREA"
)

# Danh sách 6 bảng cốt lõi
CORE_TABLES = [
    "application_train",
    "bureau",
    "previous_application",
    "POS_CASH_balance",
    "installments_payments",
    "credit_card_balance",
]

# -----------------------------------------------------------------------------
# QUY TẮC NGHIỆP VỤ & LÍNH CANH ĐẶC THÙ (WHITELIST & SENTINELS)
# -----------------------------------------------------------------------------

# Whitelist các thước đo tài sản & hạn mức giao dịch cốt lõi (Bắt buộc > 0)
# Tuyệt đối không dùng quy tắc phủ đầu (blanket rule) trên toàn bộ AMT_*
CORE_FINANCIAL_COLS = [
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
]

# Giá trị lính canh số kinh điển cho thời gian (1000 năm / chưa từng xảy ra)
TIME_SENTINEL_VALUE = 365243
TIME_SENTINEL_COLS = [
    "DAYS_EMPLOYED",
    "DAYS_FIRST_DRAWING",
    "DAYS_FIRST_DUE",
    "DAYS_LAST_DUE_1ST_VERSION",
    "DAYS_LAST_DUE",
    "DAYS_TERMINATION",
]

# Giá trị lính canh số nhân tạo cần loại bỏ trên các biến định lượng cốt lõi
NUMERIC_SENTINELS = {-1, -1.0, 1.0, 9999, 9999.0}

# Ánh xạ các lỗi chính tả / viết tắt thường gặp về chuẩn quốc tế của Home Credit
TEXT_MAPPING_RULES = {
    "Revolving Loa": "Revolving Loans",
    "Revolving Loa -": "Revolving Loans",
    "Revolving Loans -": "Revolving Loans",
    "Cash Loans -": "Cash Loans",
    "Commercial Associa": "Commercial Associate",
    "Secondary / Secondary Speci": "Secondary / secondary special",
    "Secondary / Secondary Specia": "Secondary / secondary special",
    "Secondary / Secondary Special": "Secondary / secondary special",
    "Secondary/Secondary Special": "Secondary / secondary special",
}

