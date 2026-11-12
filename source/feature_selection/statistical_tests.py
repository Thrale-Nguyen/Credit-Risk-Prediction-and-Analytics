"""
Module: statistical_tests.py
Mục đích: Thực hiện các kiểm định giả thuyết thống kê suy diễn:
- Mann-Whitney U Test & Two-Sample T-test cho các biến định lượng.
- Chi-Square Test of Independence & Cramér's V cho các biến định tính.
"""

import numpy as np
import pandas as pd
from scipy import stats
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def calculate_cohens_d(group1, group2):
    """Tính Cohen's d để đo lường độ lớn ảnh hưởng (Effect Size)."""
    n1, n2 = len(group1), len(group2)
    s1, s2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    if n1 + n2 <= 2:
        return 0.0
    pooled_std = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    if pooled_std == 0 or np.isnan(pooled_std):
        return 0.0
    return float((np.mean(group1) - np.mean(group2)) / pooled_std)


def calculate_cramers_v(contingency_table):
    """Tính Cramér's V cho bảng tương quan chéo."""
    chi2 = stats.chi2_contingency(contingency_table)[0]
    n = contingency_table.sum().sum()
    r, k = contingency_table.shape
    if n == 0 or min(r - 1, k - 1) == 0:
        return 0.0
    return float(np.sqrt(chi2 / (n * min(r - 1, k - 1))))


def run_statistical_tests(df, target_col="TARGET", sample_size=50000, random_state=42):
    """
    Thực hiện kiểm định thống kê toàn diện trên tập dữ liệu.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dữ liệu đầu vào chứa biến target và các biến đặc trưng.
    target_col : str
        Tên cột nhãn mục tiêu (0/1).
    sample_size : int or None
        Cỡ mẫu lấy ngẫu nhiên có phân tầng để tăng tốc tính toán kiểm định phi tham số.
    random_state : int
        Seed để tái lập mẫu.
        
    Returns:
    --------
    pd.DataFrame: Bảng tổng hợp kết quả kiểm định thống kê.
    """
    logger.info("Bắt đầu thực hiện kiểm định giả thuyết thống kê...")
    
    if target_col not in df.columns:
        raise ValueError(f"Không tìm thấy cột mục tiêu '{target_col}' trong dữ liệu.")
        
    # Lấy mẫu phân tầng nếu dữ liệu quá lớn để tối ưu thời gian tính toán
    if sample_size and len(df) > sample_size:
        logger.info(f"Lấy mẫu phân tầng {sample_size:,} dòng từ {len(df):,} dòng để kiểm định...")
        sample_df = df.groupby(target_col, group_keys=False).apply(
            lambda x: x.sample(int(np.rint(sample_size * len(x) / len(df))), random_state=random_state)
        ).reset_index(drop=True)
    else:
        sample_df = df

    y = sample_df[target_col].values
    mask_0 = (y == 0)
    mask_1 = (y == 1)
    
    results = []
    
    feature_cols = [c for c in df.columns if c not in [target_col, "SK_ID_CURR"]]
    total_cols = len(feature_cols)
    logger.info(f"Tổng số đặc trưng cần kiểm định: {total_cols}")
    
    for idx, col in enumerate(feature_cols):
        if (idx + 1) % 100 == 0 or (idx + 1) == total_cols:
            logger.info(f"Tiến độ kiểm định: {idx + 1}/{total_cols} cột...")
            
        series = sample_df[col]
        
        # Biến định tính / Categorical
        if series.dtype == "object" or series.dtype.name == "category":
            non_null_mask = series.notna()
            if non_null_mask.sum() == 0:
                continue
            s_clean = series[non_null_mask]
            y_clean = y[non_null_mask]
            
            ct = pd.crosstab(s_clean, y_clean)
            if ct.shape[0] < 2 or ct.shape[1] < 2:
                continue
                
            chi2_stat, p_val, dof, _ = stats.chi2_contingency(ct)
            v_val = calculate_cramers_v(ct)
            
            results.append({
                "feature": col,
                "type": "categorical",
                "test_name": "Chi-Square",
                "statistic": float(chi2_stat),
                "p_value": float(p_val),
                "is_significant_001": bool(p_val < 0.001),
                "effect_size": float(v_val),
                "effect_size_metric": "Cramer_V",
                "mean_class_0": np.nan,
                "mean_class_1": np.nan,
                "missing_pct": float(df[col].isna().mean() * 100)
            })
            
        # Biến định lượng / Numerical
        elif np.issubdtype(series.dtype, np.number):
            vals_0 = series[mask_0].dropna().values
            vals_1 = series[mask_1].dropna().values
            
            if len(vals_0) < 10 or len(vals_1) < 10:
                continue
                
            # Kiểm tra phương sai
            if np.var(vals_0) == 0 and np.var(vals_1) == 0:
                continue
                
            # Mann-Whitney U test (phù hợp với dữ liệu tài chính lệch nhiều)
            try:
                u_stat, p_val = stats.mannwhitneyu(vals_0, vals_1, alternative="two-sided")
                cohen_d = calculate_cohens_d(vals_1, vals_0)
                
                results.append({
                    "feature": col,
                    "type": "numerical",
                    "test_name": "Mann-Whitney U",
                    "statistic": float(u_stat),
                    "p_value": float(p_val),
                    "is_significant_001": bool(p_val < 0.001),
                    "effect_size": abs(cohen_d),
                    "effect_size_metric": "Abs_Cohen_d",
                    "mean_class_0": float(np.mean(vals_0)),
                    "mean_class_1": float(np.mean(vals_1)),
                    "missing_pct": float(df[col].isna().mean() * 100)
                })
            except Exception as e:
                logger.warning(f"Lỗi kiểm định cột {col}: {e}")
                continue

    res_df = pd.DataFrame(results)
    if not res_df.empty:
        res_df.sort_values(by=["is_significant_001", "effect_size"], ascending=[False, False], inplace=True)
    logger.info(f"Hoàn tất kiểm định. Tổng số biến kiểm định thành công: {len(res_df)}")
    return res_df

