"""
Package: source
Dự án: PRJ-01 - Dự đoán Rủi ro Vỡ nợ Tín dụng (Home Credit Default Risk)
"""

# Hệ thống Chẩn đoán Dữ liệu & Schema Tổng quát (Data Diagnostics Suite)
from .data_diagnostics import (
    diagnose_dataset,
    audit_single_table_health,
)

# Giai đoạn 2: Quy trình Tiền xử lý & Ghép nối dữ liệu (Pipeline Tuyến tính)
from .pipeline import run_table_sanitizer, run_merge_pipeline

# Giai đoạn 4: Phân tích Khám phá Dữ liệu & Kể chuyện Rủi ro Tín dụng (EDA & Storytelling)
from .eda import (
    plot_target_imbalance,
    plot_univariate_financial_distributions,
    compute_correlations,
    plot_correlation_matrix_heatmap,
    plot_external_sources_comparison,
    plot_age_distribution_and_default_rate,
    plot_bivariate_boxplots_and_violins,
    plot_overdue_delinquency_patterns,
    compute_multivariate_pivot_tables,
    plot_multivariate_heatmaps,
    plot_loan_purpose_risk_ranking,
)

__all__ = [
    # Data Diagnostics
    "diagnose_dataset",
    "audit_single_table_health",
    # Data Pipeline
    "run_table_sanitizer",
    "run_merge_pipeline",
    # EDA & Storytelling
    "plot_target_imbalance",
    "plot_univariate_financial_distributions",
    "compute_correlations",
    "plot_correlation_matrix_heatmap",
    "plot_external_sources_comparison",
    "plot_age_distribution_and_default_rate",
    "plot_bivariate_boxplots_and_violins",
    "plot_overdue_delinquency_patterns",
    "compute_multivariate_pivot_tables",
    "plot_multivariate_heatmaps",
    "plot_loan_purpose_risk_ranking",
]
