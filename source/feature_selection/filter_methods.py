"""
Module: filter_methods.py
Mục đích: Lọc đặc trưng đơn biến và khử đa cộng tuyến:
- Tính toán Weight of Evidence (WOE) và Information Value (IV).
- Lọc biến phương sai thấp, biến quasi-constant và missing quá cao.
- Khử đa cộng tuyến bằng ma trận tương quan và thuật toán Greedy Pairwise.
"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def calculate_feature_iv(df, feature, target_col="TARGET", n_bins=10, min_samples_bin=50):
    """
    Tính Information Value (IV) cho một đặc trưng đơn lẻ.
    Áp dụng được cho cả biến định lượng lẫn định tính.
    """
    series = df[feature]
    target = df[target_col]
    
    # Bỏ qua nếu toàn NaN hoặc chỉ có 1 giá trị duy nhất
    valid_mask = series.notna()
    if valid_mask.sum() < min_samples_bin * 2:
        return 0.0
        
    s_valid = series[valid_mask]
    y_valid = target[valid_mask]
    
    total_good = (y_valid == 0).sum()
    total_bad = (y_valid == 1).sum()
    
    if total_good == 0 or total_bad == 0:
        return 0.0
        
    # Phân khoảng nếu là biến số có nhiều giá trị duy nhất
    if np.issubdtype(s_valid.dtype, np.number) and s_valid.nunique() > n_bins:
        try:
            # Dùng qcut, nếu trùng biên độ thì fallback sang cut
            binned = pd.qcut(s_valid, q=n_bins, duplicates="drop")
        except Exception:
            binned = pd.cut(s_valid, bins=n_bins, duplicates="drop")
    else:
        binned = s_valid.astype(str)
        
    # Tính bảng phân phối Good/Bad
    grouped = pd.crosstab(binned, y_valid)
    if grouped.shape[1] < 2:
        return 0.0
        
    good_counts = grouped[0] if 0 in grouped.columns else 0
    bad_counts = grouped[1] if 1 in grouped.columns else 0
    
    # Tránh chia cho 0 với smoothing epsilon
    eps = 1e-4
    good_pct = (good_counts + eps) / (total_good + eps * len(grouped))
    bad_pct = (bad_counts + eps) / (total_bad + eps * len(grouped))
    
    woe = np.log(good_pct / bad_pct)
    iv = ((good_pct - bad_pct) * woe).sum()
    
    return float(iv) if not np.isnan(iv) else 0.0


def calculate_all_iv(df, features, target_col="TARGET", n_bins=10):
    """
    Tính IV hàng loạt cho danh sách biến đầu vào.
    """
    logger.info(f"Đang tính toán Information Value (IV) cho {len(features)} đặc trưng...")
    iv_dict = {}
    total = len(features)
    for i, col in enumerate(features):
        if (i + 1) % 100 == 0 or (i + 1) == total:
            logger.info(f"Đã tính IV: {i + 1}/{total} biến...")
        iv_val = calculate_feature_iv(df, col, target_col=target_col, n_bins=n_bins)
        iv_dict[col] = iv_val
        
    return iv_dict


def filter_by_variance_and_iv(df, features, target_col="TARGET", min_iv=0.02, max_missing=0.70, max_quasi_constant=0.99):
    """
    Lọc tầng 1: Loại bỏ các biến:
    - Missing vượt ngưỡng max_missing (mặc định 70%).
    - Quasi-constant vượt ngưỡng (giá trị phổ biến nhất > 99%).
    - Phương sai bằng 0.
    - Information Value (IV) < min_iv (mặc định 0.02 - không có năng lực dự báo).
    """
    logger.info(f"Bắt đầu lọc tầng 1 (Missing, Phương sai, Quasi-constant & IV >= {min_iv})...")
    passed_features = []
    dropped_reasons = {}
    
    iv_dict = calculate_all_iv(df, features, target_col=target_col)
    
    n_rows = len(df)
    for col in features:
        series = df[col]
        # 1. Kiểm tra Missing
        missing_rate = series.isna().mean()
        if missing_rate > max_missing:
            dropped_reasons[col] = f"Missing quá cao ({missing_rate*100:.1f}%)"
            continue
            
        # 2. Kiểm tra Quasi-constant
        top_freq = series.value_counts(normalize=True, dropna=False).values
        if len(top_freq) > 0 and top_freq[0] >= max_quasi_constant:
            dropped_reasons[col] = f"Quasi-constant ({top_freq[0]*100:.1f}%)"
            continue
            
        # 3. Kiểm tra IV
        col_iv = iv_dict.get(col, 0.0)
        if col_iv < min_iv:
            dropped_reasons[col] = f"IV quá thấp ({col_iv:.4f} < {min_iv})"
            continue
            
        passed_features.append(col)
        
    logger.info(f"Kết quả Lọc tầng 1: Giữ lại {len(passed_features)}/{len(features)} biến. Bỏ {len(dropped_reasons)} biến.")
    return passed_features, iv_dict, dropped_reasons


def remove_collinear_features(df, features, threshold=0.85, iv_dict=None):
    """
    Lọc tầng 2: Khử đa cộng tuyến bằng ma trận tương quan Pearson.
    Duyệt cặp biến có |r| > threshold. Giữ lại biến có IV cao hơn.
    """
    logger.info(f"Bắt đầu khử đa cộng tuyến với ngưỡng |r| > {threshold}...")
    
    # Chỉ tính tương quan cho các biến số
    numeric_cols = [c for c in features if np.issubdtype(df[c].dtype, np.number)]
    non_numeric_cols = [c for c in features if c not in numeric_cols]
    
    if len(numeric_cols) < 2:
        return features, []
        
    # Tính ma trận tương quan (vectorized trên pandas)
    logger.info(f"Đang tính ma trận tương quan cho {len(numeric_cols)} biến định lượng...")
    corr_matrix = df[numeric_cols].corr().abs()
    
    # Tìm các cặp có tương quan lớn hơn ngưỡng
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    to_drop = set()
    for col in upper_tri.columns:
        high_corr_partners = upper_tri.index[upper_tri[col] > threshold].tolist()
        for partner in high_corr_partners:
            if partner in to_drop or col in to_drop:
                continue
                
            # So sánh IV để giữ lại biến mạnh hơn
            iv_col = iv_dict.get(col, 0.0) if iv_dict else 0.0
            iv_partner = iv_dict.get(partner, 0.0) if iv_dict else 0.0
            
            if iv_partner >= iv_col:
                to_drop.add(col)
                break
            else:
                to_drop.add(partner)
                
    final_numeric = [c for c in numeric_cols if c not in to_drop]
    selected_features = final_numeric + non_numeric_cols
    
    logger.info(f"Kết quả Khử đa cộng tuyến: Loại bỏ {len(to_drop)} biến trùng lặp. Giữ lại {len(selected_features)} biến.")
    return selected_features, list(to_drop)

