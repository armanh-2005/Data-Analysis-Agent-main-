import pandas as pd
import numpy as np
from scipy import stats
from typing import Union, Dict, Any, List
from langchain_core.tools import tool

@tool
def cramers_v(confusion_matrix: Any) -> float:
    """
    Calculates Cramer's V statistic for categorical-categorical association.
    This is superior to correlation for nominal political data (e.g., Province vs. Party).
    
    Args:
        confusion_matrix: A pandas DataFrame (crosstab) or numpy array.
        
    Returns:
        float: Value between 0 (no association) and 1 (perfect association).
    """
    # Convert to numpy array if it's a dataframe
    if hasattr(confusion_matrix, "to_numpy"):
        confusion_matrix = confusion_matrix.to_numpy()
        
    chi2 = stats.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    
    if n == 0:
        return 0.0

    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    
    # Bias correction
    with np.errstate(divide='ignore', invalid='ignore'):
        phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
        rcorr = r - ((r-1)**2)/(n-1)
        kcorr = k - ((k-1)**2)/(n-1)
        
        denom = min((kcorr-1), (rcorr-1))
        if denom <= 0:
            return 0.0
            
        result = np.sqrt(phi2corr / denom)
        
    return float(result)


@tool
def weighted_frequency(df: Any, col: str, weight_col: str = None) -> Dict[str, Any]:
    """
    Calculates frequency and percentage of a categorical column.
    Uses survey weights if provided (crucial for polling data).
    
    Args:
        df: Input pandas DataFrame.
        col: The categorical column name.
        weight_col: Optional column name containing survey weights.
        
    Returns:
        Dict: A dictionary representation of the frequency table (keys: categories, values: counts/pct).
    """
    # Ensure df is a DataFrame (handling the Any type hint for LangChain)
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input 'df' must be a pandas DataFrame")

    if col not in df.columns:
        raise ValueError(f"Column {col} not found in DataFrame.")

    clean_df = df.dropna(subset=[col])
    
    if weight_col and weight_col in clean_df.columns:
        total_weight = clean_df[weight_col].sum()
        res = clean_df.groupby(col)[weight_col].sum().reset_index(name='count')
        res['percentage'] = (res['count'] / total_weight) * 100
    else:
        res = clean_df[col].value_counts().reset_index()
        res.columns = [col, 'count']
        res['percentage'] = (res['count'] / res['count'].sum()) * 100
    
    res = res.sort_values('percentage', ascending=False)
    return res.to_dict(orient='records')


@tool
def compare_groups(df: Any, numeric_col: str, group_col: str) -> Dict[str, Any]:
    """
    Smartly compares a numeric variable across groups using statistical tests.
    Auto-detects Parametric (T-Test/ANOVA) vs Non-Parametric (Mann-Whitney/Kruskal).
    
    Args:
        df: Input pandas DataFrame.
        numeric_col: Continuous variable (e.g., 'Age').
        group_col: Categorical variable defining groups (e.g., 'Gender').
        
    Returns:
        Dict: Test results including statistic, p_value, and conclusion.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input 'df' must be a pandas DataFrame")

    clean_df = df.dropna(subset=[numeric_col, group_col])
    groups = clean_df[group_col].unique()
    
    if len(groups) < 2:
        return {"error": "Need at least 2 groups to compare."}
    
    group_data = [clean_df[clean_df[group_col] == g][numeric_col] for g in groups]
    
    # 1. Check Normality
    is_normal = True
    for data in group_data:
        if len(data) >= 3:
            stat, p = stats.shapiro(data)
            if p < 0.05:
                is_normal = False
                break
    
    # 2. Select Test
    if len(groups) == 2:
        if is_normal:
            test_name = "Independent T-Test"
            stat, p_val = stats.ttest_ind(group_data[0], group_data[1], equal_var=False)
        else:
            test_name = "Mann-Whitney U Test"
            stat, p_val = stats.mannwhitneyu(group_data[0], group_data[1])
    else:
        if is_normal:
            test_name = "One-way ANOVA"
            stat, p_val = stats.f_oneway(*group_data)
        else:
            test_name = "Kruskal-Wallis H Test"
            stat, p_val = stats.kruskal(*group_data)
            
    return {
        "test_used": test_name,
        "is_normal_distribution": is_normal,
        "statistic": round(stat, 4),
        "p_value": round(p_val, 4),
        "significant": p_val < 0.05
    }