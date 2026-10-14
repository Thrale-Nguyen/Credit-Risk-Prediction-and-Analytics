"""
Script: run_eda.py
Package: source.eda
Dự án: PRJ-01 - Dự đoán Rủi ro Vỡ nợ Tín dụng (Home Credit Default Risk)
Mô tả: Kịch bản thực thi trọn gói quy trình Phân tích Khám phá Dữ liệu (EDA)
       - Nạp ma trận dữ liệu sau ghép nối train_merged_final.parquet (307,511 dòng).
       - Thực thi đầy đủ 3 trụ cột EDA theo đúng yêu cầu đề bài:
         1. Phân tích đơn biến (Univariate Analysis).
         2. Phân tích hai biến & Tương quan (Bivariate & Correlation).
         3. Phân tích đa biến theo phân khúc nghiệp vụ (Multivariate Pivot Tables).
       - Xuất bản toàn bộ 9 biểu đồ chất lượng cao (DPI 300) vào thư mục figures/.
       - In bảng số liệu thống kê chi tiết ra màn hình console.
"""

from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd

# Đảm bảo console Windows in tiếng Việt chuẩn UTF-8
if sys.platform.startswith("win"):
    try:
        if sys.stdout and hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if sys.stderr and hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Import các hàm chức năng từ các module nội bộ của package eda
# Xử lý đường dẫn linh hoạt hỗ trợ cả chạy module python -m source.eda.run_eda lẫn chạy trực tiếp
try:
    from .univariate import (
        plot_target_imbalance,
        plot_univariate_financial_distributions,
    )
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
except ImportError:
    from univariate import (
        plot_target_imbalance,
        plot_univariate_financial_distributions,
    )
    from bivariate import (
        compute_correlations,
        plot_correlation_matrix_heatmap,
        plot_external_sources_comparison,
        plot_age_distribution_and_default_rate,
        plot_bivariate_boxplots_and_violins,
        plot_overdue_delinquency_patterns,
    )
    from multivariate import (
        compute_multivariate_pivot_tables,
        plot_multivariate_heatmaps,
        plot_loan_purpose_risk_ranking,
    )


def locate_file(filename: str) -> Path:
    """Tìm kiếm file theo nhiều cấp thư mục tương đối."""
    candidates = [
        Path(filename),
        Path("..") / filename,
        Path("..") / ".." / filename,
        Path(__file__).resolve().parent.parent.parent / filename,
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()
    return Path(filename)


def main():
    t_start = time.time()
    print("=" * 85)
    print("KHỞI CHẠY QUY TRÌNH PHÂN TÍCH KHÁM PHÁ DỮ LIỆU (EDA) - CRISP-DM GIAI ĐOẠN 4")
    print("=" * 85)

    # 1. Xác định đường dẫn tệp dữ liệu
    parquet_path = locate_file("train_merged_final.parquet")
    prev_path = locate_file("Dataset - Home Credit Default Risk/previous_application.csv")
    figures_dir = locate_file("figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    print(f"• Tệp dữ liệu chính : {parquet_path}")
    print(f"• Tệp đơn vay cũ    : {prev_path}")
    print(f"• Thư mục xuất ảnh  : {figures_dir}")

    # 2. Danh mục cột cần thiết cho EDA
    eda_columns = [
        "SK_ID_CURR",
        "TARGET",
        "AMT_INCOME_TOTAL",
        "AMT_CREDIT",
        "AMT_ANNUITY",
        "DAYS_BIRTH",
        "DAYS_EMPLOYED",
        "CODE_GENDER",
        "NAME_INCOME_TYPE",
        "NAME_EDUCATION_TYPE",
        "NAME_CONTRACT_TYPE",
        "EXT_SOURCE_1",
        "EXT_SOURCE_2",
        "EXT_SOURCE_3",
        "BUREAU_CREDIT_DAY_OVERDUE_MEAN",
        "BUREAU_AMT_CREDIT_SUM_OVERDUE_MEAN",
        "BUREAU_CREDIT_ACTIVE_CLOSED_MEAN",
        "BUREAU_DAYS_CREDIT_MEAN",
        "POS_SK_DPD_MEAN",
        "CC_SK_DPD_MEAN",
        "CC_CNT_DRAWINGS_ATM_CURRENT_MEAN",
        "CC_AMT_BALANCE_MEAN",
        "PREV_NAME_CONTRACT_STATUS_REFUSED_MEAN",
        "PREV_NAME_CONTRACT_STATUS_APPROVED_MEAN",
    ]

    print("\n[BƯỚC 1/4] NẠP DỮ LIỆU VÀ CHUẨN BỊ ĐẶC TRƯNG...")
    t0 = time.time()
    df = pd.read_parquet(parquet_path, columns=eda_columns)
    # Tính tuổi chính xác
    df["AGE_YEARS"] = np.abs(df["DAYS_BIRTH"]) / 365.25
    print(f"✓ Đã nạp {df.shape[0]:,} dòng và {df.shape[1]} cột trong {time.time() - t0:.2f} giây.")

    # ---------------------------------------------------------
    # TRỤ CỘT 1: PHÂN TÍCH ĐƠN BIẾN (UNIVARIATE ANALYSIS)
    # ---------------------------------------------------------
    print("\n[BƯỚC 2/4] THỰC THI PHÂN TÍCH ĐƠN BIẾN (UNIVARIATE ANALYSIS)...")
    fig1_path = figures_dir / "01_target_imbalance_distribution.png"
    target_counts = plot_target_imbalance(df, output_path=str(fig1_path))

    fig2_path = figures_dir / "02_univariate_income_credit_annuity.png"
    plot_univariate_financial_distributions(df, output_path=str(fig2_path))

    # ---------------------------------------------------------
    # TRỤ CỘT 2: PHÂN TÍCH HAI BIẾN & TƯƠNG QUAN (BIVARIATE ANALYSIS)
    # ---------------------------------------------------------
    print("\n[BƯỚC 3/4] THỰC THI PHÂN TÍCH HAI BIẾN & TƯƠNG QUAN...")
    
    # Tính tương quan Pearson & Spearman
    corr_features = [
        "EXT_SOURCE_3",
        "EXT_SOURCE_2",
        "EXT_SOURCE_1",
        "AGE_YEARS",
        "CC_CNT_DRAWINGS_ATM_CURRENT_MEAN",
        "PREV_NAME_CONTRACT_STATUS_REFUSED_MEAN",
        "BUREAU_CREDIT_ACTIVE_CLOSED_MEAN",
        "BUREAU_AMT_CREDIT_SUM_OVERDUE_MEAN",
        "BUREAU_CREDIT_DAY_OVERDUE_MEAN",
        "POS_SK_DPD_MEAN",
        "AMT_CREDIT",
        "AMT_INCOME_TOTAL",
        "AMT_ANNUITY",
        "CC_SK_DPD_MEAN",
    ]
    corr_df = compute_correlations(df, features=corr_features, target_col="TARGET")
    print("\n--- BẢNG SO SÁNH TƯƠNG QUAN PEARSON VS SPEARMAN VỚI NHÃN TARGET ---")
    display_cols = ["Feature", "Valid_Count", "Pearson", "Spearman", "Spearman_pval"]
    print(corr_df[display_cols].to_string(index=False))

    fig3_path = figures_dir / "03_correlation_matrix_heatmap.png"
    plot_correlation_matrix_heatmap(df, output_path=str(fig3_path))

    fig4_path = figures_dir / "04_external_sources_comparison.png"
    plot_external_sources_comparison(df, output_path=str(fig4_path))

    fig5_path = figures_dir / "05_age_distribution_and_default_rate.png"
    age_stats = plot_age_distribution_and_default_rate(df, output_path=str(fig5_path))

    fig6_path = figures_dir / "06_distributions_boxplots_target_comparison.png"
    plot_bivariate_boxplots_and_violins(df, output_path=str(fig6_path))

    fig7_path = figures_dir / "07_bivariate_overdue_and_credit_history.png"
    plot_overdue_delinquency_patterns(df, output_path=str(fig7_path))

    # ---------------------------------------------------------
    # TRỤ CỘT 3: PHÂN TÍCH ĐA BIẾN (MULTIVARIATE ANALYSIS)
    # ---------------------------------------------------------
    print("\n[BƯỚC 4/4] THỰC THI PHÂN TÍCH ĐA BIẾN THEO PHÂN KHÚC NGHIỆP VỤ...")
    pivots = compute_multivariate_pivot_tables(df)

    print("\n--- PIVOT TABLE: GIỚI TÍNH X TRÌNH ĐỘ HỌC VẤN (% VỠ NỢ) ---")
    print(pivots["gender_education_rate"])

    print("\n--- PIVOT TABLE: LOẠI THU NHẬP X TRÌNH ĐỘ HỌC VẤN (% VỠ NỢ) ---")
    print(pivots["income_education_rate"])

    print("\n--- PIVOT TABLE: LOẠI THU NHẬP X GIỚI TÍNH (% VỠ NỢ) ---")
    print(pivots["income_gender_rate"])

    fig8_path = figures_dir / "08_multivariate_pivot_heatmaps.png"
    plot_multivariate_heatmaps(pivots, output_path=str(fig8_path))

    fig9_path = figures_dir / "09_loan_purpose_risk_ranking.png"
    purpose_stats = plot_loan_purpose_risk_ranking(
        app_df=df, prev_csv_path=str(prev_path), output_path=str(fig9_path)
    )
    if not purpose_stats.empty:
        print("\n--- TOP MỤC ĐÍCH VAY CÓ NGUY CƠ VỠ NỢ CAO NHẤT ---")
        print(purpose_stats.tail(10)[["NAME_CASH_LOAN_PURPOSE", "count", "default_pct"]].to_string(index=False))

    print("\n" + "=" * 85)
    print(f"✓ HOÀN TẤT TOÀN DIỆN EDA TRONG {time.time() - t_start:.2f} GIÂY!")
    print(f"✓ ĐÃ XUẤT 9 TỆP HÌNH ẢNH BIỂU ĐỒ CHẤT LƯỢNG CAO (300 DPI) VÀO: {figures_dir.resolve()}")
    print("=" * 85)


if __name__ == "__main__":
    main()
