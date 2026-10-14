"""
Module: bivariate.py
Package: source.eda
Dự án: PRJ-01 - Dự đoán Rủi ro Vỡ nợ Tín dụng (Home Credit Default Risk)
Mô tả: Phân tích hai biến & Ma trận tương quan (Bivariate & Correlation Analysis):
       - Khảo sát tương quan Pearson và Spearman giữa các đặc trưng với nhãn TARGET.
       - Trực quan hóa ma trận tương quan Heatmap.
       - Đánh giá phân hóa của bộ ba điểm tín dụng ngoài EXT_SOURCE_1, 2, 3.
       - Đánh giá mối quan hệ phi tuyến giữa độ tuổi (DAYS_BIRTH / AGE_YEARS) và rủi ro vỡ nợ.
       - So sánh phân phối giữa nhóm vỡ nợ (TARGET=1) và trả nợ tốt (TARGET=0) qua Boxplot & Violin plot.
       - Phân tích các chỉ số hành vi nợ quá hạn và lịch sử từ chối vay.
"""

from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from scipy import stats
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

plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Segoe UI", "sans-serif"]
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8


def compute_correlations(
    df: pd.DataFrame,
    features: Optional[List[str]] = None,
    target_col: str = "TARGET",
) -> pd.DataFrame:
    """
    Tính toán hệ số tương quan tuyến tính Pearson và tương quan hạng Spearman kèm p-value
    giữa các đặc trưng số với nhãn mục tiêu TARGET.
    """
    if features is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        features = [c for c in numeric_cols if c != target_col]

    records = []
    target_series = df[target_col]

    for feat in features:
        if feat not in df.columns:
            continue
        valid_mask = df[feat].notna() & target_series.notna()
        n_valid = int(valid_mask.sum())
        if n_valid < 30:
            continue

        x = df.loc[valid_mask, feat].values
        y = target_series.loc[valid_mask].values

        # Tránh chia cho 0 nếu độ lệch chuẩn = 0
        if np.std(x) == 0 or np.std(y) == 0:
            continue

        p_corr, p_pval = stats.pearsonr(x, y)
        s_corr, s_pval = stats.spearmanr(x, y)

        records.append({
            "Feature": feat,
            "Valid_Count": n_valid,
            "Pearson": float(p_corr),
            "Pearson_pval": float(p_pval),
            "Spearman": float(s_corr),
            "Spearman_pval": float(s_pval),
            "Abs_Spearman": float(abs(s_corr)),
        })

    res_df = pd.DataFrame(records).sort_values(by="Abs_Spearman", ascending=False)
    return res_df


def plot_correlation_matrix_heatmap(
    df: pd.DataFrame,
    output_path: Optional[str] = "figures/03_correlation_matrix_heatmap.png",
    dpi: int = 300,
) -> pd.DataFrame:
    """
    Vẽ biểu đồ Heatmap thể hiện ma trận tương quan giữa nhãn TARGET và các đặc trưng rủi ro cốt lõi.
    """
    df_calc = df.copy()
    if "AGE_YEARS" not in df_calc.columns and "DAYS_BIRTH" in df_calc.columns:
        df_calc["AGE_YEARS"] = np.abs(df_calc["DAYS_BIRTH"]) / 365.25

    key_features = [
        "TARGET",
        "EXT_SOURCE_3",
        "EXT_SOURCE_2",
        "EXT_SOURCE_1",
        "AGE_YEARS",
        "AMT_CREDIT",
        "AMT_INCOME_TOTAL",
        "AMT_ANNUITY",
    ]

    # Bổ sung các cột phụ trợ nếu có trong df
    optional_candidates = [
        "PREV_NAME_CONTRACT_STATUS_REFUSED_MEAN",
        "BUREAU_CREDIT_ACTIVE_CLOSED_MEAN",
        "BUREAU_AMT_CREDIT_SUM_OVERDUE_MEAN",
        "CC_CNT_DRAWINGS_ATM_CURRENT_MEAN",
    ]
    for c in optional_candidates:
        if c in df_calc.columns:
            key_features.append(c)

    # Đổi tên hiển thị ngắn gọn trực quan
    rename_dict = {
        "TARGET": "TARGET (Vỡ nợ)",
        "EXT_SOURCE_3": "EXT_SOURCE_3",
        "EXT_SOURCE_2": "EXT_SOURCE_2",
        "EXT_SOURCE_1": "EXT_SOURCE_1",
        "AGE_YEARS": "Độ tuổi (Năm)",
        "AMT_CREDIT": "Khoản vay (Credit)",
        "AMT_INCOME_TOTAL": "Thu nhập (Income)",
        "AMT_ANNUITY": "Khoản trả kỳ (Annuity)",
        "PREV_NAME_CONTRACT_STATUS_REFUSED_MEAN": "Tỷ lệ từ chối cũ (Refused)",
        "BUREAU_CREDIT_ACTIVE_CLOSED_MEAN": "Tỷ lệ tất toán CIC (Closed)",
        "BUREAU_AMT_CREDIT_SUM_OVERDUE_MEAN": "Nợ quá hạn CIC (Overdue)",
        "CC_CNT_DRAWINGS_ATM_CURRENT_MEAN": "Rút tiền ATM thẻ tín dụng",
    }

    sub_df = df_calc[key_features].rename(columns=rename_dict)
    corr_matrix = sub_df.corr(method="pearson")

    fig, ax = plt.subplots(figsize=(12, 10))

    # Mask tam giác trên để ma trận thanh thoát
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    cmap = sns.diverging_palette(240, 10, as_cmap=True)

    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt=".3f",
        cmap="coolwarm",
        vmin=-0.25,
        vmax=0.40,
        square=True,
        linewidths=1.0,
        cbar_kws={"shrink": 0.8, "label": "Hệ số tương quan Pearson (r)"},
        ax=ax,
        annot_kws={"size": 10, "weight": "bold"},
    )

    ax.set_title(
        "MA TRẬN TƯƠNG QUAN GIỮA NHÃN TARGET VÀ CÁC ĐẶC TRƯNG RỦI RO CỐT LÕI",
        fontsize=13,
        fontweight="bold",
        pad=16,
    )

    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=dpi, bbox_inches="tight")
        print(f"✓ Đã lưu Heatmap ma trận tương quan: {out.resolve()}")

    plt.close(fig)
    return corr_matrix


def plot_external_sources_comparison(
    df: pd.DataFrame,
    output_path: Optional[str] = "figures/04_external_sources_comparison.png",
    dpi: int = 300,
) -> None:
    """
    Trực quan hóa phân phối của bộ ba điểm tín dụng bên ngoài (EXT_SOURCE_1, 2, 3)
    phân rã theo nhóm trả nợ tốt (TARGET=0) và vỡ nợ (TARGET=1) qua KDE và Violin plot.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    sources = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
    colors = ["#2b5c8f", "#d9534f"]

    for idx, col in enumerate(sources):
        # 1. KDE Distribution (Hàng 1)
        ax_kde = axes[0, idx]
        valid_df = df[[col, "TARGET"]].dropna()

        for target_val, color, label in zip([0, 1], colors, ["Trả nợ tốt (0)", "Vỡ nợ (1)"]):
            subset = valid_df[valid_df["TARGET"] == target_val][col]
            sns.kdeplot(
                subset,
                ax=ax_kde,
                color=color,
                label=f"{label} - Med: {subset.median():.3f}",
                fill=True,
                alpha=0.35,
                linewidth=2,
            )

        ax_kde.set_title(f"Phân phối KDE: {col}", fontsize=12, fontweight="bold")
        ax_kde.set_xlabel(f"Điểm số {col}", fontsize=10)
        ax_kde.set_ylabel("Mật độ phân phối", fontsize=10)
        ax_kde.legend(loc="upper left", frameon=True, fontsize=9)
        ax_kde.grid(alpha=0.3)
        ax_kde.spines["top"].set_visible(False)
        ax_kde.spines["right"].set_visible(False)

        # 2. Violin Plot (Hàng 2)
        ax_violin = axes[1, idx]
        sns.violinplot(
            x="TARGET",
            y=col,
            hue="TARGET",
            data=valid_df,
            ax=ax_violin,
            palette=colors,
            inner="quartile",
            cut=0,
            legend=False,
        )
        ax_violin.set_title(f"Violin Plot: {col} theo TARGET", fontsize=12, fontweight="bold")
        ax_violin.set_xticks([0, 1])
        ax_violin.set_xticklabels(["Trả nợ tốt (0)", "Vỡ nợ (1)"], fontsize=10, fontweight="bold")
        ax_violin.set_xlabel("Trạng thái thanh toán", fontsize=10)
        ax_violin.set_ylabel(f"Điểm số {col}", fontsize=10)
        ax_violin.grid(axis="y", alpha=0.3)
        ax_violin.spines["top"].set_visible(False)
        ax_violin.spines["right"].set_visible(False)

    fig.suptitle(
        "NĂNG LỰC PHÂN BIỆT RỦI RO CỰC MẠNH CỦA BỘ BA ĐIỂM SỐ TÍN DỤNG NGOÀI (EXT_SOURCE_1, 2, 3)",
        fontsize=14,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout()

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=dpi, bbox_inches="tight")
        print(f"✓ Đã lưu biểu đồ phân tích EXT_SOURCE: {out.resolve()}")

    plt.close(fig)


def plot_age_distribution_and_default_rate(
    df: pd.DataFrame,
    output_path: Optional[str] = "figures/05_age_distribution_and_default_rate.png",
    dpi: int = 300,
) -> pd.DataFrame:
    """
    Phân tích tác động phi tuyến của độ tuổi (DAYS_BIRTH chuyển đổi thành AGE_YEARS)
    lên xác suất vỡ nợ của khách hàng qua biểu đồ KDE và Tỷ lệ vỡ nợ theo nhóm tuổi.
    """
    df_calc = df.copy()
    if "AGE_YEARS" not in df_calc.columns:
        df_calc["AGE_YEARS"] = np.abs(df_calc["DAYS_BIRTH"]) / 365.25

    # Phân nhóm tuổi
    bins = [20, 25, 35, 45, 55, 65, 75]
    labels = ["< 25", "25 - 34", "35 - 44", "45 - 54", "55 - 64", "≥ 65"]
    df_calc["AGE_GROUP"] = pd.cut(df_calc["AGE_YEARS"], bins=bins, labels=labels, right=False)

    age_stats = df_calc.groupby("AGE_GROUP")["TARGET"].agg(["count", "mean", "sum"]).reset_index()
    age_stats["default_pct"] = age_stats["mean"] * 100

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 1. Phân phối KDE tuổi theo TARGET
    colors = ["#2b5c8f", "#d9534f"]
    for target_val, color, label in zip([0, 1], colors, ["Trả nợ tốt (0)", "Vỡ nợ (1)"]):
        subset = df_calc[df_calc["TARGET"] == target_val]["AGE_YEARS"].dropna()
        sns.kdeplot(
            subset,
            ax=axes[0],
            color=color,
            label=f"{label} (Median: {subset.median():.1f} tuổi)",
            fill=True,
            alpha=0.35,
            linewidth=2,
        )

    axes[0].set_title(
        "A. Mật độ phân phối độ tuổi giữa nhóm Trả nợ tốt vs Vỡ nợ",
        fontsize=12,
        fontweight="bold",
    )
    axes[0].set_xlabel("Độ tuổi của người vay (Năm)", fontsize=11)
    axes[0].set_ylabel("Mật độ phân phối (Density)", fontsize=11)
    axes[0].legend(loc="upper right", frameon=True, fontsize=10)
    axes[0].grid(alpha=0.3)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    # 2. Tỷ lệ vỡ nợ theo từng nhóm tuổi (Default Rate by Age Group)
    bar_colors = ["#d9534f" if r > 8.07 else "#2a9d8f" for r in age_stats["default_pct"]]
    bars = axes[1].bar(
        age_stats["AGE_GROUP"],
        age_stats["default_pct"],
        color=bar_colors,
        edgecolor="#333333",
        width=0.55,
    )

    # Đường chuẩn danh mục 8.07%
    axes[1].axhline(
        8.07,
        color="#e63946",
        linestyle="--",
        linewidth=1.8,
        label="Tỷ lệ vỡ nợ trung bình toàn danh mục (8.07%)",
    )

    for bar, pct, cnt in zip(bars, age_stats["default_pct"], age_stats["count"]):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2.0,
            pct + 0.35,
            f"{pct:.2f}%\n({cnt:,} hồ sơ)",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
            color="#222222",
        )

    axes[1].set_ylim(0, max(age_stats["default_pct"]) * 1.25)
    axes[1].set_title(
        "B. Tỷ lệ vỡ nợ (%) theo các phân nhóm tuổi (Nghịch biến rõ nét)",
        fontsize=12,
        fontweight="bold",
    )
    axes[1].set_xlabel("Nhóm tuổi (Age Bracket)", fontsize=11)
    axes[1].set_ylabel("Tỷ lệ vỡ nợ (%)", fontsize=11)
    axes[1].legend(loc="upper right", frameon=True, fontsize=9.5)
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    fig.suptitle(
        "PHÂN TÍCH QUAN HỆ ĐỘ TUỔI VÀ RỦI RO TÍN DỤNG: NHÓM TRẺ TUỔI MANG NGUY CƠ CAO GẤP ĐÔI",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=dpi, bbox_inches="tight")
        print(f"✓ Đã lưu biểu đồ phân tích độ tuổi: {out.resolve()}")

    plt.close(fig)
    return age_stats


def plot_bivariate_boxplots_and_violins(
    df: pd.DataFrame,
    output_path: Optional[str] = "figures/06_distributions_boxplots_target_comparison.png",
    dpi: int = 300,
) -> None:
    """
    So sánh phân phối các biến tài chính cốt lõi giữa nhóm vỡ nợ (TARGET=1) và trả nợ tốt (TARGET=0)
    thông qua biểu đồ Boxplot và Violin plot:
    1. Khoản vay đề nghị (AMT_CREDIT).
    2. Thu nhập logarit (Log10 AMT_INCOME_TOTAL).
    3. Tỷ lệ Đòn bẩy nợ trên thu nhập (CREDIT_TO_INCOME = AMT_CREDIT / AMT_INCOME_TOTAL).
    4. Tỷ lệ Nghĩa vụ trả nợ trên thu nhập (DTI = AMT_ANNUITY / AMT_INCOME_TOTAL).
    """
    df_calc = df.copy()

    # Tính toán các tỷ lệ tài chính
    df_calc["LOG_INCOME"] = np.log10(df_calc["AMT_INCOME_TOTAL"].clip(lower=1000))
    df_calc["CREDIT_TO_INCOME"] = (df_calc["AMT_CREDIT"] / df_calc["AMT_INCOME_TOTAL"]).clip(upper=15)
    df_calc["ANNUITY_TO_INCOME"] = (df_calc["AMT_ANNUITY"] / df_calc["AMT_INCOME_TOTAL"]).clip(upper=1.0)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    colors = ["#2b5c8f", "#d9534f"]

    # 1. Boxplot Khoản vay AMT_CREDIT
    sns.boxplot(
        x="TARGET",
        y="AMT_CREDIT",
        hue="TARGET",
        data=df_calc,
        ax=axes[0, 0],
        palette=colors,
        showmeans=True,
        meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black"},
        showfliers=False,
        legend=False,
    )
    axes[0, 0].set_title("A. Khoản vay được cấp (AMT_CREDIT)", fontsize=12, fontweight="bold")
    axes[0, 0].set_xticks([0, 1])
    axes[0, 0].set_xticklabels(["Trả nợ tốt (0)", "Vỡ nợ (1)"], fontsize=10, fontweight="bold")
    axes[0, 0].set_ylabel("Hạn mức vay (USD)", fontsize=10)
    axes[0, 0].yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"{int(x/1000):,}k"))
    axes[0, 0].grid(axis="y", alpha=0.3)
    axes[0, 0].spines["top"].set_visible(False)
    axes[0, 0].spines["right"].set_visible(False)

    # 2. Violin Plot Log10 Thu nhập
    sns.violinplot(
        x="TARGET",
        y="LOG_INCOME",
        hue="TARGET",
        data=df_calc,
        ax=axes[0, 1],
        palette=colors,
        inner="quartile",
        cut=0,
        legend=False,
    )
    axes[0, 1].set_title("B. Thu nhập Log10 (AMT_INCOME_TOTAL)", fontsize=12, fontweight="bold")
    axes[0, 1].set_xticks([0, 1])
    axes[0, 1].set_xticklabels(["Trả nợ tốt (0)", "Vỡ nợ (1)"], fontsize=10, fontweight="bold")
    axes[0, 1].set_ylabel("Log10(Thu nhập)", fontsize=10)
    axes[0, 1].grid(axis="y", alpha=0.3)
    axes[0, 1].spines["top"].set_visible(False)
    axes[0, 1].spines["right"].set_visible(False)

    # 3. Boxplot Đòn bẩy nợ (CREDIT_TO_INCOME)
    sns.boxplot(
        x="TARGET",
        y="CREDIT_TO_INCOME",
        hue="TARGET",
        data=df_calc,
        ax=axes[1, 0],
        palette=colors,
        showmeans=True,
        meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black"},
        showfliers=False,
        legend=False,
    )
    axes[1, 0].set_title(
        "C. Tỷ lệ Đòn bẩy nợ / Thu nhập (Credit to Income)", fontsize=12, fontweight="bold"
    )
    axes[1, 0].set_xticks([0, 1])
    axes[1, 0].set_xticklabels(["Trả nợ tốt (0)", "Vỡ nợ (1)"], fontsize=10, fontweight="bold")
    axes[1, 0].set_ylabel("Hệ số Đòn bẩy (Lần)", fontsize=10)
    axes[1, 0].grid(axis="y", alpha=0.3)
    axes[1, 0].spines["top"].set_visible(False)
    axes[1, 0].spines["right"].set_visible(False)

    # 4. Boxplot Nghĩa vụ trả nợ DTI (ANNUITY_TO_INCOME)
    sns.boxplot(
        x="TARGET",
        y="ANNUITY_TO_INCOME",
        hue="TARGET",
        data=df_calc,
        ax=axes[1, 1],
        palette=colors,
        showmeans=True,
        meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black"},
        showfliers=False,
        legend=False,
    )
    axes[1, 1].set_title(
        "D. Tỷ lệ Gánh nặng trả góp / Thu nhập (DTI)", fontsize=12, fontweight="bold"
    )
    axes[1, 1].set_xticks([0, 1])
    axes[1, 1].set_xticklabels(["Trả nợ tốt (0)", "Vỡ nợ (1)"], fontsize=10, fontweight="bold")
    axes[1, 1].set_ylabel("Tỷ lệ DTI (%)", fontsize=10)
    axes[1, 1].yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    axes[1, 1].grid(axis="y", alpha=0.3)
    axes[1, 1].spines["top"].set_visible(False)
    axes[1, 1].spines["right"].set_visible(False)

    fig.suptitle(
        "SO SÁNH HÌNH THÁI PHÂN PHỐI GIỮA NHÓM VỠ NỢ VÀ TRẢ NỢ TỐT (BOXPLOT & VIOLIN PLOT)",
        fontsize=14,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout()

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=dpi, bbox_inches="tight")
        print(f"✓ Đã lưu biểu đồ Boxplot & Violin so sánh TARGET: {out.resolve()}")

    plt.close(fig)


def plot_overdue_delinquency_patterns(
    df: pd.DataFrame,
    output_path: Optional[str] = "figures/07_bivariate_overdue_and_credit_history.png",
    dpi: int = 300,
) -> None:
    """
    Trực quan hóa sự khác biệt về lịch sử tín dụng và nợ xấu giữa nhóm TARGET=0 và TARGET=1:
    1. Tỷ lệ đơn vay bị Home Credit từ chối trong quá khứ (PREV_NAME_CONTRACT_STATUS_REFUSED_MEAN).
    2. Tỷ lệ khoản vay đã tất toán thành công tại CIC (BUREAU_CREDIT_ACTIVE_CLOSED_MEAN).
    3. Tần suất rút tiền mặt thẻ tín dụng (CC_CNT_DRAWINGS_ATM_CURRENT_MEAN).
    4. Số ngày nợ quá hạn trung bình ngoài hệ thống (BUREAU_CREDIT_DAY_OVERDUE_MEAN).
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    colors = ["#2b5c8f", "#d9534f"]

    # 1. Tỷ lệ đơn vay bị từ chối
    refused_col = "PREV_NAME_CONTRACT_STATUS_REFUSED_MEAN"
    if refused_col in df.columns:
        mean_refused = df.groupby("TARGET")[refused_col].mean() * 100
        bars1 = axes[0, 0].bar(
            ["Trả nợ tốt (0)", "Vỡ nợ (1)"],
            mean_refused.values,
            color=colors,
            width=0.5,
            edgecolor="#333333",
        )
        for b, v in zip(bars1, mean_refused.values):
            axes[0, 0].text(
                b.get_x() + b.get_width() / 2.0,
                v + 0.3,
                f"{v:.2f}%",
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=11,
            )
        axes[0, 0].set_title(
            "A. Tỷ lệ đơn vay quá khứ bị từ chối (Refusal Rate)", fontsize=12, fontweight="bold"
        )
        axes[0, 0].set_ylabel("Tỷ lệ bị từ chối (%)", fontsize=10)
        axes[0, 0].set_ylim(0, max(mean_refused.values) * 1.25)
        axes[0, 0].grid(axis="y", alpha=0.3)
        axes[0, 0].spines["top"].set_visible(False)
        axes[0, 0].spines["right"].set_visible(False)

    # 2. Tỷ lệ tài khoản đã tất toán (Closed Credit Accounts at Bureau)
    closed_col = "BUREAU_CREDIT_ACTIVE_CLOSED_MEAN"
    if closed_col in df.columns:
        mean_closed = df.groupby("TARGET")[closed_col].mean() * 100
        bars2 = axes[0, 1].bar(
            ["Trả nợ tốt (0)", "Vỡ nợ (1)"],
            mean_closed.values,
            color=colors,
            width=0.5,
            edgecolor="#333333",
        )
        for b, v in zip(bars2, mean_closed.values):
            axes[0, 1].text(
                b.get_x() + b.get_width() / 2.0,
                v + 0.8,
                f"{v:.2f}%",
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=11,
            )
        axes[0, 1].set_title(
            "B. Tỷ lệ khoản vay đã tất toán tại CIC (Closed Ratio)",
            fontsize=12,
            fontweight="bold",
        )
        axes[0, 1].set_ylabel("Tỷ lệ đã tất toán (%)", fontsize=10)
        axes[0, 1].set_ylim(0, max(mean_closed.values) * 1.2)
        axes[0, 1].grid(axis="y", alpha=0.3)
        axes[0, 1].spines["top"].set_visible(False)
        axes[0, 1].spines["right"].set_visible(False)

    # 3. Tần suất rút tiền ATM thẻ tín dụng (CC Cash Advance Distress)
    atm_col = "CC_CNT_DRAWINGS_ATM_CURRENT_MEAN"
    if atm_col in df.columns:
        mean_atm = df.groupby("TARGET")[atm_col].mean()
        bars3 = axes[1, 0].bar(
            ["Trả nợ tốt (0)", "Vỡ nợ (1)"],
            mean_atm.values,
            color=colors,
            width=0.5,
            edgecolor="#333333",
        )
        for b, v in zip(bars3, mean_atm.values):
            axes[1, 0].text(
                b.get_x() + b.get_width() / 2.0,
                v + 0.04,
                f"{v:.3f} lần/tháng",
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=11,
            )
        axes[1, 0].set_title(
            "C. Số lần rút tiền mặt ATM từ thẻ tín dụng hàng tháng",
            fontsize=12,
            fontweight="bold",
        )
        axes[1, 0].set_ylabel("Số giao dịch rút tiền / tháng", fontsize=10)
        axes[1, 0].set_ylim(0, max(mean_atm.values) * 1.25)
        axes[1, 0].grid(axis="y", alpha=0.3)
        axes[1, 0].spines["top"].set_visible(False)
        axes[1, 0].spines["right"].set_visible(False)

    # 4. Số ngày quá hạn trung bình tại CIC (BUREAU_CREDIT_DAY_OVERDUE_MEAN)
    overdue_col = "BUREAU_CREDIT_DAY_OVERDUE_MEAN"
    if overdue_col in df.columns:
        mean_overdue = df.groupby("TARGET")[overdue_col].mean()
        bars4 = axes[1, 1].bar(
            ["Trả nợ tốt (0)", "Vỡ nợ (1)"],
            mean_overdue.values,
            color=colors,
            width=0.5,
            edgecolor="#333333",
        )
        for b, v in zip(bars4, mean_overdue.values):
            axes[1, 1].text(
                b.get_x() + b.get_width() / 2.0,
                v + 0.05,
                f"{v:.3f} ngày",
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=11,
            )
        axes[1, 1].set_title(
            "D. Số ngày nợ quá hạn trung bình ngoài hệ thống (Bureau DPD)",
            fontsize=12,
            fontweight="bold",
        )
        axes[1, 1].set_ylabel("Số ngày quá hạn bình quân", fontsize=10)
        axes[1, 1].set_ylim(0, max(mean_overdue.values) * 1.25)
        axes[1, 1].grid(axis="y", alpha=0.3)
        axes[1, 1].spines["top"].set_visible(False)
        axes[1, 1].spines["right"].set_visible(False)

    fig.suptitle(
        "DẤU HIỆU CẢNH BÁO SỚM TỪ LỊCH SỬ TÍN DỤNG, NỢ QUÁ HẠN VÀ HÀNH VI RÚT TIỀN THẺ",
        fontsize=14,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout()

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=dpi, bbox_inches="tight")
        print(f"✓ Đã lưu biểu đồ dấu hiệu cảnh báo lịch sử nợ: {out.resolve()}")

    plt.close(fig)
