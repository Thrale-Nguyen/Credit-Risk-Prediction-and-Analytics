"""
Module: embedded_selection.py
Mục đích: Lọc đặc trưng nâng cao dựa trên mô hình cây (LightGBM Feature Importance):
- Huấn luyện mô hình LightGBM Baseline trên Stratified K-Fold.
- Đo lường và xếp hạng đặc trưng theo Gain & Split Importance.
- Lọc ra tập đặc trưng tinh hoa (Tree-Ready Features).
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
import logging

logger = logging.getLogger(__name__)


def select_features_with_lightgbm(df, candidate_features, target_col="TARGET", n_splits=5, top_k=100, random_state=42):
    """
    Sử dụng LightGBM để tính toán Feature Importance và chọn lọc Top K đặc trưng.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dữ liệu đã qua lọc sơ bộ.
    candidate_features : list
        Danh sách các đặc trưng ứng viên.
    target_col : str
        Tên cột mục tiêu.
    n_splits : int
        Số lượng fold trong StratifiedKFold.
    top_k : int
        Số lượng đặc trưng tinh hoa cần giữ lại.
    random_state : int
        Seed ngẫu nhiên.
        
    Returns:
    --------
    selected_features : list
        Danh sách Top K đặc trưng được chọn.
    importance_df : pd.DataFrame
        Bảng chi tiết Feature Importance của tất cả các biến ứng viên.
    """
    logger.info(f"Bắt đầu huấn luyện LightGBM {n_splits}-Fold CV để đo Feature Importance cho {len(candidate_features)} biến...")
    
    X = df[candidate_features].copy()
    y = df[target_col].values
    
    # Ép kiểu category cho các biến object/category để LightGBM tự xử lý
    for col in X.columns:
        if X[col].dtype == "object":
            X[col] = X[col].astype("category")
            
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    feature_importance_gain = np.zeros(len(candidate_features))
    feature_importance_split = np.zeros(len(candidate_features))
    
    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "n_estimators": 150,
        "random_state": random_state,
        "n_jobs": -1,
        "verbose": -1
    }
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        logger.info(f"Đang huấn luyện Fold {fold + 1}/{n_splits}...")
        X_train, y_train = X.iloc[train_idx], y[train_idx]
        X_val, y_val = X.iloc[val_idx], y[val_idx]
        
        clf = lgb.LGBMClassifier(**params)
        clf.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
        )
        
        # Lấy importance theo split và gain
        feature_importance_split += clf.booster_.feature_importance(importance_type="split") / n_splits
        feature_importance_gain += clf.booster_.feature_importance(importance_type="gain") / n_splits
        
    importance_df = pd.DataFrame({
        "feature": candidate_features,
        "importance_gain": feature_importance_gain,
        "importance_split": feature_importance_split
    }).sort_values(by="importance_gain", ascending=False).reset_index(drop=True)
    
    # Loại bỏ các biến có gain = 0
    non_zero_df = importance_df[importance_df["importance_gain"] > 0]
    
    # Chọn Top K hoặc toàn bộ các biến có gain > 0 nếu số lượng nhỏ hơn top_k
    k_final = min(top_k, len(non_zero_df))
    selected_features = non_zero_df.head(k_final)["feature"].tolist()
    
    logger.info(f"Hoàn tất lọc mô hình cây: Giữ lại Top {len(selected_features)} đặc trưng tinh hoa (theo Gain Importance).")
    return selected_features, importance_df

