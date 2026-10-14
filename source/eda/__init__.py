"""
Package: source.eda
Dự án: PRJ-01 - Dự đoán Rủi ro Vỡ nợ Tín dụng (Home Credit Default Risk)
Mô tả: Module thực thi Phân tích Khám phá Dữ liệu (Exploratory Data Analysis - EDA)
       và Kể chuyện Dữ liệu Rủi ro Tín dụng (Credit Risk Storytelling):
       1. Phân tích đơn biến (Univariate Analysis): Thu nhập, Khoản vay, Mất cân bằng nhãn TARGET.
       2. Phân tích hai biến & Tương quan (Bivariate & Correlation): Pearson, Spearman, Boxplot, Violin plot,
          Bộ ba điểm tín dụng ngoài EXT_SOURCE, Độ tuổi, Lịch sử nợ quá hạn.
       3. Phân tích đa biến (Multivariate Analysis): Bóc tách rủi ro chéo qua pivot_table & crosstab
          theo Giới tính, Trình độ học vấn, Loại thu nhập, Mục đích vay.
"""

from .univariate import plot_target_imbalance, plot_univariate_financial_distributions
from .bivariate import (
    compute_correlations,
    plot_correlation_matrix_heatmap,
    plot_external_sources_comparison,
    plot_age_distribution_and_default_rate,
    plot_bivariate_boxplots_and_violins,
    plot_overdue_delinquency_patterns,
)
from .multivariate import (
    compute_multivariate_pivot_tables,
    plot_multivariate_heatmaps,
    plot_loan_purpose_risk_ranking,
)

__all__ = [
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
