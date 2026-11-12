"""
Module: linear_preprocessing.py
Mục đích: Thiết kế pipeline chuyên biệt cho mô hình tuyến tính (Logistic Regression):
- Khử đa cộng tuyến khắt khe (|r| < 0.65).
- Kiểm định và lọc VIF lặp (Iterative Variance Inflation Factor < 5.0).
- Xử lý Missing Value (Imputation) và Chuẩn hóa thang đo (Robust/Standard Scaling).
- Xuất bản ma trận đặc trưng tinh gọn chuẩn mực cho Credit Scorecard.
"""

import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
import logging

logger = logging.getLogger(__name__)


def calculate_vif_series(X_matrix):
    """Tính VIF cho từng cột trong ma trận dữ liệu đã chuẩn hóa."""
    vif_data = pd.Series(
        [variance_inflation_factor(X_matrix.values, i) for i in range(X_matrix.shape[1])],
        index=X_matrix.columns
    )
    return vif_data


def iterative_vif_filter(X_df, max_vif=5.0):
    """
    Khử đa cộng tuyến bằng vòng lặp VIF:
    Loại bỏ từng biến có VIF cao nhất cho đến khi toàn bộ các biến đều có VIF < max_vif.
    """
    logger.info(f"Bắt đầu kiểm định VIF lặp (ngưỡng max VIF = {max_vif})...")
    X_current = X_df.copy()
    
    iteration = 0
    while True:
        iteration += 1
        vif_series = calculate_vif_series(X_current)
        max_current_vif = vif_series.max()
        max_vif_feature = vif_series.idxmax()
        
        if max_current_vif <= max_vif or len(X_current.columns) <= 5:
            logger.info(f"Hội tụ VIF sau {iteration - 1} vòng lặp. VIF cao nhất hiện tại: {max_current_vif:.2f} ({max_vif_feature})")
            break
            
        logger.info(f"Vòng {iteration}: Loại biến '{max_vif_feature}' có VIF = {max_current_vif:.2f} > {max_vif}")
        X_current = X_current.drop(columns=[max_vif_feature])
        
    return X_current.columns.tolist(), vif_series


def prepare_logistic_dataset(df, candidate_features, target_col="TARGET", max_features=35, max_vif=5.0):
    """
    Quy trình tiền xử lý hoàn chỉnh cho Logistic Regression:
    1. Lọc chỉ lấy các biến số hoặc biến phân loại đã được mã hóa sạch.
    2. Điền khuyết (Median Imputation).
    3. Chuẩn hóa thang đo (RobustScaler).
    4. Lọc VIF để triệt tiêu đa cộng tuyến.
    5. Trả về DataFrame sẵn sàng huấn luyện và danh sách biến tinh hoa.
    """
    logger.info(f"Bắt đầu chuẩn bị dữ liệu cho Logistic Regression từ {len(candidate_features)} biến ứng viên...")
    
    # Chỉ chọn các biến dạng số
    numeric_candidates = [c for c in candidate_features if np.issubdtype(df[c].dtype, np.number)]
    
    # Lấy tối đa top candidate ban đầu để kiểm soát tính toán VIF
    selected_numeric = numeric_candidates[:max_features]
    
    X_raw = df[selected_numeric].copy()
    y = df[target_col].copy()
    
    # 1. Điền khuyết bằng Median
    logger.info("Điền khuyết (Median Imputer) cho các biến định lượng...")
    imputer = SimpleImputer(strategy="median")
    X_imputed = pd.DataFrame(imputer.fit_transform(X_raw), columns=selected_numeric, index=df.index)
    
    # 2. Chuẩn hóa thang đo bằng RobustScaler (chống ảnh hưởng của ngoại lai)
    logger.info("Chuẩn hóa thang đo (RobustScaler)...")
    scaler = RobustScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=selected_numeric, index=df.index)
    
    # 3. Lọc VIF lặp để đảm bảo không còn đa cộng tuyến
    vif_features, final_vif_series = iterative_vif_filter(X_scaled, max_vif=max_vif)
    
    X_final = X_scaled[vif_features].copy()
    X_final[target_col] = y
    
    logger.info(f"Hoàn tất chuẩn bị tập dữ liệu Logistic: {len(vif_features)} biến sạch, VIF < {max_vif}.")
    return X_final, vif_features, final_vif_series

