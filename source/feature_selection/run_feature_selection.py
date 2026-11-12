"""
Script điều phối thực thi toàn diện Giai đoạn 4:
Kiểm định giả thuyết thống kê & Chọn lọc đặc trưng tinh hoa (Dual-Track: Tree & Linear)
"""

import os
import sys
import json
import time
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Thiết lập đường dẫn thư mục gốc
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from source.feature_selection.statistical_tests import run_statistical_tests
from source.feature_selection.filter_methods import filter_by_variance_and_iv, remove_collinear_features
from source.feature_selection.embedded_selection import select_features_with_lightgbm
from source.feature_selection.linear_preprocessing import prepare_logistic_dataset

# Cấu hình logging
os.makedirs("data/processed", exist_ok=True)
os.makedirs("figures", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("data/processed/stage4_execution.log", mode="w", encoding="utf-8")
    ]
)
logger = logging.getLogger("STAGE_4_COORDINATOR")


def plot_top_iv(iv_dict, output_path="figures/10_top_iv_features.png", top_n=25):
    """Vẽ biểu đồ Barplot xếp hạng các đặc trưng có Information Value (IV) cao nhất."""
    logger.info(f"Đang vẽ biểu đồ Top {top_n} biến có IV cao nhất...")
    iv_df = pd.DataFrame(list(iv_dict.items()), columns=["feature", "iv"])
    iv_df = iv_df.sort_values(by="iv", ascending=False).head(top_n)
    
    plt.figure(figsize=(12, 8), dpi=300)
    sns.set_style("whitegrid")
    
    # Phân loại màu sắc theo sức mạnh dự báo của IV
    colors = []
    for val in iv_df["iv"]:
        if val > 0.3:
            colors.append("#1b5e20")  # Xanh lục đậm: Rất mạnh (> 0.3)
        elif val > 0.1:
            colors.append("#2e7d32")  # Xanh lục: Mạnh (0.1 - 0.3)
        elif val > 0.05:
            colors.append("#f57f17")  # Vàng cam: Trung bình (0.05 - 0.1)
        else:
            colors.append("#b71c1c")  # Đỏ: Yếu (0.02 - 0.05)
            
    bars = plt.barh(iv_df["feature"][::-1], iv_df["iv"][::-1], color=colors[::-1], edgecolor="black", height=0.65)
    
    plt.axvline(0.02, color="gray", linestyle="--", linewidth=1, label="Ngưỡng lọc tối thiểu (IV = 0.02)")
    plt.axvline(0.1, color="blue", linestyle=":", linewidth=1, label="Ngưỡng dự báo mạnh (IV = 0.10)")
    plt.axvline(0.3, color="green", linestyle="-.", linewidth=1, label="Ngưỡng phân tách cực mạnh (IV = 0.30)")
    
    plt.title(f"XẾP HẠNG TOP {top_n} ĐẶC TRƯNG CÓ INFORMATION VALUE (IV) CAO NHẤT", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Information Value (IV)", fontsize=12, labelpad=10)
    plt.ylabel("Đặc trưng (Features)", fontsize=12)
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Đã lưu biểu đồ: {output_path}")


def plot_tree_importance(importance_df, output_path="figures/11_tree_importance_ranking.png", top_n=25):
    """Vẽ biểu đồ Barplot xếp hạng đặc trưng theo LightGBM Gain Importance."""
    logger.info(f"Đang vẽ biểu đồ Top {top_n} biến theo LightGBM Gain Importance...")
    top_df = importance_df.head(top_n)
    
    plt.figure(figsize=(12, 8), dpi=300)
    sns.set_style("whitegrid")
    
    norm_gain = top_df["importance_gain"] / top_df["importance_gain"].max()
    palette = sns.color_palette("mako", n_colors=top_n)[::-1]
    
    plt.barh(top_df["feature"][::-1], top_df["importance_gain"][::-1], color=palette, edgecolor="black", height=0.65)
    
    plt.title(f"TOP {top_n} ĐẶC TRƯNG TINH HOA THEO LIGHTGBM GAIN IMPORTANCE", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Average Gain Importance (5-Fold Stratified CV)", fontsize=12, labelpad=10)
    plt.ylabel("Đặc trưng (Features)", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Đã lưu biểu đồ: {output_path}")


def main():
    start_time = time.time()
    logger.info("==========================================================================")
    logger.info("KHỞI ĐỘNG ĐIỀU PHỐI GIAI ĐOẠN 4: KIỂM ĐỊNH THỐNG KÊ & CHỌN LỌC ĐẶC TRƯNG")
    logger.info("==========================================================================")
    
    # 1. Nạp dữ liệu đầu vào đã ghép nối sạch từ Giai đoạn 2
    input_file = "data/processed/train_merged.parquet"
    if not os.path.exists(input_file):
        input_file = "train_merged_final.parquet"
        
    logger.info(f"Đang nạp dữ liệu từ: {input_file}...")
    df = pd.read_parquet(input_file)
    logger.info(f"Dữ liệu ban đầu: {df.shape[0]:,} dòng x {df.shape[1]:,} cột.")
    
    target_col = "TARGET"
    id_col = "SK_ID_CURR"
    initial_features = [c for c in df.columns if c not in [target_col, id_col]]
    
    # 2. Kiểm định giả thuyết thống kê (Hypothesis Testing)
    logger.info("--- BƯỚC 1: KIỂM ĐỊNH GIẢ THUYẾT THỐNG KÊ (MANN-WHITNEY U & CHI-SQUARE) ---")
    stats_df = run_statistical_tests(df, target_col=target_col, sample_size=50000, random_state=42)
    stats_output = "data/processed/statistical_tests_results.csv"
    stats_df.to_csv(stats_output, index=False, encoding="utf-8")
    logger.info(f"Đã lưu kết quả kiểm định thống kê vào: {stats_output}")
    
    # 3. Lọc tầng 1: Missing, Phương sai, Quasi-constant & Information Value (IV >= 0.02)
    logger.info("--- BƯỚC 2: LỌC TẦNG 1 (MISSING, QUASI-CONSTANT & IV >= 0.02) ---")
    passed_stage1, iv_dict, dropped_reasons = filter_by_variance_and_iv(
        df, initial_features, target_col=target_col, min_iv=0.02, max_missing=0.70, max_quasi_constant=0.99
    )
    
    # Lưu tóm tắt IV
    iv_df = pd.DataFrame(list(iv_dict.items()), columns=["feature", "iv"]).sort_values(by="iv", ascending=False)
    iv_df.to_csv("data/processed/iv_summary.csv", index=False, encoding="utf-8")
    
    # Vẽ biểu đồ Top IV
    plot_top_iv(iv_dict, output_path="figures/10_top_iv_features.png", top_n=25)
    
    # 4. Nhánh B: Chọn lọc đặc trưng cho Mô hình Cây (LightGBM & Random Forest)
    logger.info("--- BƯỚC 3: XÂY DỰNG NHÁNH ĐẶC TRƯNG MÔ HÌNH CÂY (TREE-READY) ---")
    # Khử đa cộng tuyến với ngưỡng |r| > 0.85
    tree_candidates, collinear_dropped_tree = remove_collinear_features(
        df, passed_stage1, threshold=0.85, iv_dict=iv_dict
    )
    
    # Lọc bằng LightGBM Gain Importance (Top 100)
    tree_selected_features, importance_df = select_features_with_lightgbm(
        df, tree_candidates, target_col=target_col, n_splits=5, top_k=100, random_state=42
    )
    
    # Lưu bảng Feature Importance và biểu đồ
    importance_df.to_csv("data/processed/tree_feature_importance.csv", index=False, encoding="utf-8")
    plot_tree_importance(importance_df, output_path="figures/11_tree_importance_ranking.png", top_n=25)
    
    # Lưu ma trận đặc trưng cây sạch (.parquet)
    tree_cols = [id_col] + tree_selected_features + [target_col]
    df_tree = df[tree_cols].copy()
    tree_output_path = "data/processed/train_tree_selected.parquet"
    df_tree.to_parquet(tree_output_path, index=False)
    logger.info(f"Đã lưu ma trận đặc trưng mô hình cây ({df_tree.shape[1]} cột) vào: {tree_output_path}")
    
    # 5. Nhánh A: Chọn lọc & Chuẩn hóa cho Logistic Regression (Linear-Ready)
    logger.info("--- BƯỚC 4: XÂY DỰNG NHÁNH ĐẶC TRƯNG LOGISTIC REGRESSION (LINEAR-READY) ---")
    # Lọc tương quan khắt khe hơn (|r| < 0.65)
    linear_candidates, collinear_dropped_linear = remove_collinear_features(
        df, passed_stage1, threshold=0.65, iv_dict=iv_dict
    )
    
    # Điền khuyết, RobustScaler, và Lọc VIF lặp (VIF < 5.0)
    df_linear, linear_selected_features, final_vif = prepare_logistic_dataset(
        df, linear_candidates, target_col=target_col, max_features=35, max_vif=5.0
    )
    
    # Bổ sung cột ID
    df_linear[id_col] = df[id_col].values
    cols_order = [id_col] + [c for c in df_linear.columns if c not in [id_col, target_col]] + [target_col]
    df_linear = df_linear[cols_order]
    
    linear_output_path = "data/processed/train_logistic_selected.parquet"
    df_linear.to_parquet(linear_output_path, index=False)
    logger.info(f"Đã lưu ma trận đặc trưng Logistic ({df_linear.shape[1]} cột) vào: {linear_output_path}")
    
    # 6. Xuất bản Metadata tổng kết Giai đoạn 4
    elapsed = time.time() - start_time
    summary_metadata = {
        "status": "success",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": round(elapsed, 2),
        "initial_features": len(initial_features),
        "stage1_passed_features": len(passed_stage1),
        "tree_branch": {
            "collinear_dropped": len(collinear_dropped_tree),
            "final_selected_features_count": len(tree_selected_features),
            "output_file": tree_output_path,
            "features_list": tree_selected_features
        },
        "logistic_branch": {
            "collinear_dropped": len(collinear_dropped_linear),
            "final_selected_features_count": len(linear_selected_features),
            "output_file": linear_output_path,
            "features_list": linear_selected_features,
            "max_vif_recorded": float(final_vif.max())
        }
    }
    
    with open("data/processed/stage4_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_metadata, f, indent=2, ensure_ascii=False)
        
    logger.info("==========================================================================")
    logger.info(f"HOÀN TẤT THÀNH CÔNG GIAI ĐOẠN 4 TRONG {elapsed:.2f} GIÂY ({elapsed/60:.2f} PHÚT)!")
    logger.info(f"- Đặc trưng Mô hình Cây (Tree-Ready)    : {len(tree_selected_features)} biến.")
    logger.info(f"- Đặc trưng Logistic Regression (VIF < 5): {len(linear_selected_features)} biến.")
    logger.info("==========================================================================")


if __name__ == "__main__":
    main()

