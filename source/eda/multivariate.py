"""
Module: multivariate.py
Package: source.eda
Dự án: PRJ-01 - Dự đoán Rủi ro Vỡ nợ Tín dụng (Home Credit Default Risk)
Mô tả: Phân tích đa biến theo phân khúc nghiệp vụ (Multivariate & Business Segment Risk):
       - Bóc tách rủi ro chéo qua pivot_table và crosstab theo các chiều:
         + Loại thu nhập (NAME_INCOME_TYPE)
         + Trình độ học vấn (NAME_EDUCATION_TYPE)
         + Giới tính (CODE_GENDER)
         + Loại hợp đồng (NAME_CONTRACT_TYPE)
         + Mục đích vay tiền mặt từ lịch sử đơn vay (NAME_CASH_LOAN_PURPOSE)
       - Trực quan hóa Heatmap rủi ro chéo và Biểu đồ xếp hạng rủi ro theo mục đích vay.
"""

from pathlib import Path
import sys
from typing import Dict, Optional, Tuple, Union
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

plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Segoe UI", "sans-serif"]
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8


def compute_multivariate_pivot_tables(
    df: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    """
    Tính toán các bảng chéo (crosstab / pivot_table) bóc tách tỷ lệ vỡ nợ (%)
    và số lượng hồ sơ (count) theo các phân khúc nghiệp vụ cốt lõi.
    """
    clean_df = df.copy()
    if "CODE_GENDER" in clean_df.columns:
        clean_df = clean_df[~clean_df["CODE_GENDER"].astype(str).str.upper().isin(["XNA", "NAN"])]
        clean_df["CODE_GENDER"] = clean_df["CODE_GENDER"].astype(str).str.strip().str.upper()

    for col in ["NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE", "NAME_CONTRACT_TYPE"]:
        if col in clean_df.columns:
            clean_df[col] = clean_df[col].astype(str).str.strip().str.title()
            clean_df[col] = clean_df[col].replace({"Nan": np.nan, "Xna": np.nan, "Chưa Rõ": np.nan, "Chua Ro": np.nan})

    pivots = {}

    # 1. Giới tính x Học vấn
    if "CODE_GENDER" in clean_df.columns and "NAME_EDUCATION_TYPE" in clean_df.columns:
        p_rate = (
            clean_df.pivot_table(
                index="CODE_GENDER",
                columns="NAME_EDUCATION_TYPE",
                values="TARGET",
                aggfunc="mean",
            )
            * 100
        )
        p_count = clean_df.pivot_table(
            index="CODE_GENDER",
            columns="NAME_EDUCATION_TYPE",
            values="TARGET",
            aggfunc="count",
        )
        pivots["gender_education_rate"] = p_rate.round(2)
        pivots["gender_education_count"] = p_count

    # 2. Loại thu nhập x Học vấn
    if "NAME_INCOME_TYPE" in clean_df.columns and "NAME_EDUCATION_TYPE" in clean_df.columns:
        p_rate = (
            clean_df.pivot_table(
                index="NAME_INCOME_TYPE",
                columns="NAME_EDUCATION_TYPE",
                values="TARGET",
                aggfunc="mean",
            )
            * 100
        )
        p_count = clean_df.pivot_table(
            index="NAME_INCOME_TYPE",
            columns="NAME_EDUCATION_TYPE",
            values="TARGET",
            aggfunc="count",
        )
        pivots["income_education_rate"] = p_rate.round(2)
        pivots["income_education_count"] = p_count

    # 3. Giới tính x Loại thu nhập
    if "CODE_GENDER" in clean_df.columns and "NAME_INCOME_TYPE" in clean_df.columns:
        p_rate = (
            clean_df.pivot_table(
                index="NAME_INCOME_TYPE",
                columns="CODE_GENDER",
                values="TARGET",
                aggfunc="mean",
            )
            * 100
        )
        p_count = clean_df.pivot_table(
            index="NAME_INCOME_TYPE",
            columns="CODE_GENDER",
            values="TARGET",
            aggfunc="count",
        )
        pivots["income_gender_rate"] = p_rate.round(2)
        pivots["income_gender_count"] = p_count

    # 4. Loại hợp đồng x Giới tính
    if "NAME_CONTRACT_TYPE" in clean_df.columns and "CODE_GENDER" in clean_df.columns:
        p_rate = (
            clean_df.pivot_table(
                index="NAME_CONTRACT_TYPE",
                columns="CODE_GENDER",
                values="TARGET",
                aggfunc="mean",
            )
            * 100
        )
        p_count = clean_df.pivot_table(
            index="NAME_CONTRACT_TYPE",
            columns="CODE_GENDER",
            values="TARGET",
            aggfunc="count",
        )
        pivots["contract_gender_rate"] = p_rate.round(2)
        pivots["contract_gender_count"] = p_count

    return pivots


def plot_multivariate_heatmaps(
    pivots: Dict[str, pd.DataFrame],
    output_path: Optional[str] = "figures/08_multivariate_pivot_heatmaps.png",
    dpi: int = 300,
) -> None:
    """
    Trực quan hóa ma trận nhiệt (Heatmap) rủi ro vỡ nợ chéo trên 4 không gian phân khúc:
    A. Giới tính x Trình độ học vấn
    B. Loại thu nhập x Trình độ học vấn
    C. Loại thu nhập x Giới tính
    D. Loại hợp đồng x Giới tính
    """
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.28)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    edu_order = [
        "Lower Secondary",
        "Secondary / Secondary Special",
        "Incomplete Higher",
        "Higher Education",
        "Academic Degree",
    ]
    major_incomes = [
        "Working",
        "Commercial Associate",
        "State Servant",
        "Pensioner",
        "Unemployed",
    ]

    # 1. Giới tính x Học vấn
    if "gender_education_rate" in pivots:
        df1 = pivots["gender_education_rate"]
        cols1 = [c for c in edu_order if c in df1.columns]
        if cols1:
            df1 = df1[cols1]
        sns.heatmap(
            df1,
            annot=True,
            fmt=".2f",
            cmap="YlOrRd",
            vmin=0.0,
            vmax=16.0,
            linewidths=1.2,
            cbar_kws={"label": "Tỷ lệ vỡ nợ (%)"},
            ax=ax1,
            annot_kws={"size": 11, "weight": "bold"},
        )
        ax1.set_title(
            "A. Tỷ lệ vỡ nợ (%): Giới tính x Trình độ học vấn", fontsize=12, fontweight="bold"
        )
        ax1.set_ylabel("Giới tính (F: Nữ, M: Nam)", fontsize=10)
        ax1.set_xlabel("Trình độ học vấn", fontsize=10)
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=25, ha="right", fontsize=9)

    # 2. Loại thu nhập x Học vấn
    if "income_education_rate" in pivots:
        df2 = pivots["income_education_rate"]
        idx2 = [i for i in major_incomes if i in df2.index]
        cols2 = [c for c in edu_order if c in df2.columns]
        if idx2:
            df2 = df2.reindex(index=idx2)
        if cols2:
            df2 = df2[[c for c in cols2 if c in df2.columns]]
        df2 = df2.dropna(how="all")
        sns.heatmap(
            df2,
            annot=True,
            fmt=".2f",
            cmap="YlOrRd",
            vmin=0.0,
            vmax=16.0,
            linewidths=1.2,
            cbar_kws={"label": "Tỷ lệ vỡ nợ (%)"},
            ax=ax2,
            annot_kws={"size": 11, "weight": "bold"},
        )
        ax2.set_title(
            "B. Tỷ lệ vỡ nợ (%): Loại thu nhập x Trình độ học vấn",
            fontsize=12,
            fontweight="bold",
        )
        ax2.set_ylabel("Nguồn thu nhập chính", fontsize=10)
        ax2.set_xlabel("Trình độ học vấn", fontsize=10)
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=25, ha="right", fontsize=9)

    # 3. Loại thu nhập x Giới tính
    if "income_gender_rate" in pivots:
        df3 = pivots["income_gender_rate"]
        idx3 = [i for i in major_incomes if i in df3.index]
        if idx3:
            df3 = df3.reindex(index=idx3)
        df3 = df3.dropna(how="all")
        sns.heatmap(
            df3,
            annot=True,
            fmt=".2f",
            cmap="YlOrRd",
            vmin=0.0,
            vmax=16.0,
            linewidths=1.2,
            cbar_kws={"label": "Tỷ lệ vỡ nợ (%)"},
            ax=ax3,
            annot_kws={"size": 11, "weight": "bold"},
        )
        ax3.set_title(
            "C. Tỷ lệ vỡ nợ (%): Loại thu nhập x Giới tính", fontsize=12, fontweight="bold"
        )
        ax3.set_ylabel("Nguồn thu nhập chính", fontsize=10)
        ax3.set_xlabel("Giới tính (F: Nữ, M: Nam)", fontsize=10)

    # 4. Loại hợp đồng x Giới tính
    if "contract_gender_rate" in pivots:
        df4 = pivots["contract_gender_rate"]
        sns.heatmap(
            df4,
            annot=True,
            fmt=".2f",
            cmap="YlOrRd",
            vmin=0.0,
            vmax=16.0,
            linewidths=1.2,
            cbar_kws={"label": "Tỷ lệ vỡ nợ (%)"},
            ax=ax4,
            annot_kws={"size": 11, "weight": "bold"},
        )
        ax4.set_title(
            "D. Tỷ lệ vỡ nợ (%): Loại hợp đồng vay x Giới tính",
            fontsize=12,
            fontweight="bold",
        )
        ax4.set_ylabel("Hình thức hợp đồng", fontsize=10)
        ax4.set_xlabel("Giới tính (F: Nữ, M: Nam)", fontsize=10)

    fig.suptitle(
        "BÓC TÁCH RỦI RO CHÉO ĐA BIẾN THEO PHÂN KHÚC NGHIỆP VỤ (MULTIVARIATE CROSS-RISK HEATMAPS)",
        fontsize=14,
        fontweight="bold",
        y=0.99,
    )

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=dpi, bbox_inches="tight")
        print(f"✓ Đã lưu Heatmap rủi ro đa biến: {out.resolve()}")

    plt.close(fig)


def plot_loan_purpose_risk_ranking(
    app_df: pd.DataFrame,
    prev_csv_path: Optional[str] = "Dataset - Home Credit Default Risk/previous_application.csv",
    output_path: Optional[str] = "figures/09_loan_purpose_risk_ranking.png",
    dpi: int = 300,
) -> pd.DataFrame:
    """
    Phân tích tỷ lệ vỡ nợ theo mục đích vay tiền mặt (NAME_CASH_LOAN_PURPOSE)
    trích xuất từ lịch sử đơn vay cũ (previous_application) liên kết với nhãn TARGET của khách hàng.
    """
    prev_path = Path(prev_csv_path) if prev_csv_path else None
    if not prev_path or not prev_path.exists():
        # Tìm đường dẫn tương đối dự phòng
        candidates = [
            Path("data/processed/previous_application.parquet"),
            Path("data/raw/previous_application.csv"),
            Path("..") / "Dataset - Home Credit Default Risk" / "previous_application.csv",
        ]
        found = False
        for c in candidates:
            if c.exists():
                prev_path = c
                found = True
                break
        if not found:
            print(f"! Cảnh báo: Không tìm thấy tệp {prev_csv_path}. Bỏ qua plot loan purpose.")
            return pd.DataFrame()

    if prev_path.suffix.lower() == ".parquet":
        prev_df = pd.read_parquet(prev_path, columns=["SK_ID_CURR", "NAME_CASH_LOAN_PURPOSE"])
    else:
        prev_df = pd.read_csv(
            prev_path, usecols=["SK_ID_CURR", "NAME_CASH_LOAN_PURPOSE"]
        )

    # Chuẩn hóa về Title Case để gộp các nhóm bị phân tán chữ hoa/thường (vd: Car repairs và CAR REPAIRS)
    prev_df["NAME_CASH_LOAN_PURPOSE"] = prev_df["NAME_CASH_LOAN_PURPOSE"].astype(str).str.strip().str.title()
    # Loại trừ mã kỹ thuật không nêu mục đích cụ thể (Xap, Xna, Chưa rõ)
    invalid_purposes = {"Xap", "Xna", "Chưa Rõ", "Chua Ro", "Nan", ""}
    prev_filtered = prev_df[~prev_df["NAME_CASH_LOAN_PURPOSE"].isin(invalid_purposes)]

    # Merge với bảng đơn vay chính
    merged = prev_filtered.merge(
        app_df[["SK_ID_CURR", "TARGET"]], on="SK_ID_CURR", how="inner"
    )

    # Thống kê theo mục đích vay
    stats_df = (
        merged.groupby("NAME_CASH_LOAN_PURPOSE")["TARGET"]
        .agg(["count", "mean"])
        .reset_index()
    )
    # Lọc các mục đích có từ 100 hồ sơ trở lên để đảm bảo ý nghĩa thống kê
    stats_df = stats_df[stats_df["count"] >= 100].copy()
    stats_df["default_pct"] = stats_df["mean"] * 100
    stats_df = stats_df.sort_values(by="default_pct", ascending=True)

    fig, ax = plt.subplots(figsize=(14, 10))

    baseline = float(app_df["TARGET"].mean() * 100)

    # Đổi màu cảnh báo theo ngưỡng rủi ro
    colors = [
        "#d9534f" if r >= 13.0 else ("#f4a261" if r >= 10.0 else "#2a9d8f")
        for r in stats_df["default_pct"]
    ]

    bars = ax.barh(
        stats_df["NAME_CASH_LOAN_PURPOSE"],
        stats_df["default_pct"],
        color=colors,
        edgecolor="#333333",
        height=0.65,
    )

    # Đường mốc trung bình toàn danh mục động
    ax.axvline(
        baseline,
        color="#1d3557",
        linestyle="--",
        linewidth=2,
        label=f"Tỷ lệ vỡ nợ trung bình danh mục ({baseline:.2f}%)",
    )

    # Annotate giá trị trên từng thanh
    for bar, pct, cnt in zip(bars, stats_df["default_pct"], stats_df["count"]):
        xval = bar.get_width()
        ax.text(
            xval + 0.3,
            bar.get_y() + bar.get_height() / 2.0,
            f"{pct:.2f}% ({cnt:,} đơn)",
            ha="left",
            va="center",
            fontsize=9.5,
            fontweight="bold",
            color="#222222",
        )

    ax.set_xlim(0, max(stats_df["default_pct"]) * 1.25)
    ax.set_title(
        "BẢNG XẾP HẠNG MỨC ĐỘ RỦI RO THEO MỤC ĐÍCH VAY TIỀN MẶT (NAME_CASH_LOAN_PURPOSE)",
        fontsize=13,
        fontweight="bold",
        pad=16,
    )
    ax.set_xlabel("Tỷ lệ vỡ nợ (% Default Rate)", fontsize=11)
    ax.set_ylabel("Mục đích vay tiền mặt khai báo", fontsize=11)
    ax.legend(loc="lower right", frameon=True, fontsize=10)
    ax.grid(axis="x", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=dpi, bbox_inches="tight")
        print(f"✓ Đã lưu biểu đồ xếp hạng rủi ro mục đích vay: {out.resolve()}")

    plt.close(fig)
    return stats_df
