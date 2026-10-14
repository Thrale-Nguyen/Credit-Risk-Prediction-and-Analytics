"""
Module: univariate.py
Package: source.eda
Dự án: PRJ-01 - Dự đoán Rủi ro Vỡ nợ Tín dụng (Home Credit Default Risk)
Mô tả: Phân tích đơn biến (Univariate Analysis):
       - Trực quan hóa tỷ lệ mất cân bằng nhãn mục tiêu TARGET (Bar chart & Donut chart).
       - Trực quan hóa hình dạng phân phối (Histogram, KDE plot) của Thu nhập (AMT_INCOME_TOTAL),
         Khoản vay (AMT_CREDIT), và Khoản trả định kỳ (AMT_ANNUITY).
"""

from pathlib import Path
import sys
from typing import Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns

# Đảm bảo console Windows in tiếng Việt chuẩn UTF-8
if sys.platform.startswith("win"):
    try:
        if sys.stdout and hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if sys.stderr and hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Thiết lập phong cách đồ họa chuẩn mực học thuật và tài chính
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Segoe UI", "sans-serif"]
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8


def plot_target_imbalance(
    df: pd.DataFrame,
    output_path: Optional[str] = "figures/01_target_imbalance_distribution.png",
    dpi: int = 300,
) -> pd.Series:
    """
    Trực quan hóa phân phối nhãn mục tiêu TARGET với biểu đồ kép: Cột & Donut Chart.
    Minh họa tỷ lệ mất cân bằng lớp nghiêm trọng (Imbalance Ratio 1 : 11.4).
    """
    counts = df["TARGET"].value_counts().sort_index()
    percentages = df["TARGET"].value_counts(normalize=True).sort_index() * 100

    n_good = counts[0]
    n_bad = counts[1]
    pct_good = percentages[0]
    pct_bad = percentages[1]
    ratio = n_good / n_bad

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    colors = ["#2b5c8f", "#d9534f"]  # Navy Blue (Good) & Coral Red (Default)

    # 1. Bar Chart
    bars = axes[0].bar(
        ["0: Trả nợ tốt\n(Non-Default)", "1: Khó khăn / Vỡ nợ\n(Default)"],
        counts.values,
        color=colors,
        width=0.55,
        edgecolor="#333333",
        linewidth=1.2,
    )

    # Annotate bar counts and percentages
    for bar, count, pct in zip(bars, counts.values, percentages.values):
        yval = bar.get_height()
        axes[0].text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + 4000,
            f"{count:,} hồ sơ\n({pct:.2f}%)",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color="#222222",
        )

    axes[0].set_ylim(0, max(counts.values) * 1.15)
    axes[0].set_title(
        "Phân phối số lượng hồ sơ theo nhãn TARGET", fontsize=13, fontweight="bold", pad=12
    )
    axes[0].set_ylabel("Số lượng khách hàng (Hồ sơ)", fontsize=11)
    axes[0].yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"{int(x):,}"))
    axes[0].grid(axis="y", linestyle="--", alpha=0.4)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    # 2. Donut Chart
    wedges, texts, autotexts = axes[1].pie(
        counts.values,
        labels=["Trả nợ tốt (91.93%)", "Vỡ nợ (8.07%)"],
        autopct="%1.2f%%",
        startangle=40,
        colors=colors,
        pctdistance=0.75,
        explode=(0, 0.08),
        wedgeprops=dict(width=0.45, edgecolor="#ffffff", linewidth=2),
    )

    for at in autotexts:
        at.set_color("#ffffff")
        at.set_fontsize(11)
        at.set_fontweight("bold")
    for t in texts:
        t.set_fontsize(11)

    axes[1].text(
        0,
        0,
        f"Mất cân bằng\n1 : {ratio:.1f}\n(8.07%)",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color="#333333",
    )
    axes[1].set_title(
        "Tỷ lệ mất cân bằng cấu trúc danh mục tín dụng",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )

    fig.suptitle(
        "PHÂN TÍCH ĐƠN BIẾN: PHÂN PHỐI NHÃN MỤC TIÊU VÀ ĐỘ LỆCH DANH MỤC (CRISP-DM GIAI ĐOẠN 4)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )

    plt.tight_layout()

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=dpi, bbox_inches="tight")
        print(f"✓ Đã lưu biểu đồ phân phối TARGET: {out.resolve()}")

    plt.close(fig)
    return counts


def plot_univariate_financial_distributions(
    df: pd.DataFrame,
    output_path: Optional[str] = "figures/02_univariate_income_credit_annuity.png",
    dpi: int = 300,
) -> None:
    """
    Trực quan hóa hình thái phân phối (Histogram và KDE) của 3 biến tài chính cốt lõi:
    1. Thu nhập khách hàng (AMT_INCOME_TOTAL) - Phân tích phân phối gốc & Log-scale.
    2. Tổng hạn mức khoản vay (AMT_CREDIT).
    3. Khoản trả góp định kỳ (AMT_ANNUITY).
    """
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.25)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    # 1. Thu nhập AMT_INCOME_TOTAL (Cắt ngưỡng trực quan 600,000 để thấy rõ hình thái)
    income = df["AMT_INCOME_TOTAL"].dropna()
    inc_median = income.median()
    inc_mean = income.mean()
    inc_p99 = income.quantile(0.99)
    inc_display = income[income <= 600_000]

    sns.histplot(
        inc_display,
        kde=True,
        ax=ax1,
        color="#2b5c8f",
        bins=50,
        stat="density",
        edgecolor="#1b3c5f",
        line_kws=dict(linewidth=2, color="#e63946"),
    )
    ax1.axvline(
        inc_median,
        color="#2a9d8f",
        linestyle="--",
        linewidth=2,
        label=f"Median: {inc_median:,.0f} $",
    )
    ax1.axvline(
        inc_mean,
        color="#e76f51",
        linestyle=":",
        linewidth=2,
        label=f"Mean: {inc_mean:,.0f} $",
    )
    ax1.set_title(
        "A. Phân phối Thu nhập (AMT_INCOME_TOTAL ≤ 600k $)",
        fontsize=12,
        fontweight="bold",
    )
    ax1.set_xlabel("Thu nhập hàng năm (USD)", fontsize=10)
    ax1.set_ylabel("Mật độ phân phối (Density)", fontsize=10)
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"{int(x/1000):,}k"))
    ax1.legend(loc="upper right", frameon=True)
    ax1.grid(alpha=0.3)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # 2. Thu nhập trên thang đo Logarithm (Log-Income)
    log_income = np.log10(income[income > 0])
    sns.histplot(
        log_income,
        kde=True,
        ax=ax2,
        color="#457b9d",
        bins=60,
        stat="density",
        edgecolor="#1d3557",
        line_kws=dict(linewidth=2, color="#e63946"),
    )
    ax2.axvline(
        np.log10(inc_median),
        color="#2a9d8f",
        linestyle="--",
        linewidth=2,
        label=f"Log10(Median): {np.log10(inc_median):.2f}",
    )
    ax2.set_title(
        "B. Thu nhập trên thang Log10 (Dạng chuẩn Log-Normal)",
        fontsize=12,
        fontweight="bold",
    )
    ax2.set_xlabel("Log10(Thu nhập)", fontsize=10)
    ax2.set_ylabel("Mật độ phân phối (Density)", fontsize=10)
    ax2.legend(loc="upper right", frameon=True)
    ax2.grid(alpha=0.3)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # 3. Khoản vay AMT_CREDIT
    credit = df["AMT_CREDIT"].dropna()
    cred_median = credit.median()
    cred_mean = credit.mean()
    cred_p99 = credit.quantile(0.99)
    cred_display = credit[credit <= cred_p99]

    sns.histplot(
        cred_display,
        kde=True,
        ax=ax3,
        color="#1d3557",
        bins=50,
        stat="density",
        edgecolor="#0f1d2e",
        line_kws=dict(linewidth=2, color="#e63946"),
    )
    ax3.axvline(
        cred_median,
        color="#2a9d8f",
        linestyle="--",
        linewidth=2,
        label=f"Median: {cred_median:,.0f} $",
    )
    ax3.axvline(
        cred_mean,
        color="#e76f51",
        linestyle=":",
        linewidth=2,
        label=f"Mean: {cred_mean:,.0f} $",
    )
    ax3.set_title(
        "C. Phân phối Tổng khoản vay đề nghị (AMT_CREDIT ≤ 99th)",
        fontsize=12,
        fontweight="bold",
    )
    ax3.set_xlabel("Số tiền được duyệt cấp tín dụng (USD)", fontsize=10)
    ax3.set_ylabel("Mật độ phân phối (Density)", fontsize=10)
    ax3.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"{int(x/1000):,}k"))
    ax3.legend(loc="upper right", frameon=True)
    ax3.grid(alpha=0.3)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)

    # 4. Khoản trả định kỳ AMT_ANNUITY
    annuity = df["AMT_ANNUITY"].dropna()
    ann_median = annuity.median()
    ann_mean = annuity.mean()
    ann_p99 = annuity.quantile(0.99)
    ann_display = annuity[annuity <= ann_p99]

    sns.histplot(
        ann_display,
        kde=True,
        ax=ax4,
        color="#3d5a80",
        bins=50,
        stat="density",
        edgecolor="#1d3557",
        line_kws=dict(linewidth=2, color="#e63946"),
    )
    ax4.axvline(
        ann_median,
        color="#2a9d8f",
        linestyle="--",
        linewidth=2,
        label=f"Median: {ann_median:,.0f} $",
    )
    ax4.axvline(
        ann_mean,
        color="#e76f51",
        linestyle=":",
        linewidth=2,
        label=f"Mean: {ann_mean:,.0f} $",
    )
    ax4.set_title(
        "D. Phân phối Khoản trả góp định kỳ (AMT_ANNUITY ≤ 99th)",
        fontsize=12,
        fontweight="bold",
    )
    ax4.set_xlabel("Số tiền trả nợ hàng kỳ (USD)", fontsize=10)
    ax4.set_ylabel("Mật độ phân phối (Density)", fontsize=10)
    ax4.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"{int(x/1000):,}k"))
    ax4.legend(loc="upper right", frameon=True)
    ax4.grid(alpha=0.3)
    ax4.spines["top"].set_visible(False)
    ax4.spines["right"].set_visible(False)

    fig.suptitle(
        "HÌNH DẠNG PHÂN PHỐI CÁC ĐẶC TRƯNG TÀI CHÍNH CỐT LÕI (UNIVARIATE DISTRIBUTIONS & SKEWNESS)",
        fontsize=14,
        fontweight="bold",
        y=0.99,
    )

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=dpi, bbox_inches="tight")
        print(f"✓ Đã lưu biểu đồ phân phối tài chính: {out.resolve()}")

    plt.close(fig)
