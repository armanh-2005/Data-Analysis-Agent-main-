import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Any, Union
from langchain_core.tools import tool

# Global Map
LIKERT_MAP = {
    "strongly disagree": 1, "disagree": 2, "neutral": 3, "neither": 3, "agree": 4, "strongly agree": 5,
    "kamelan mokhalef": 1, "mokhalef": 2, "bi-tara": 3, "movafeg": 4, "kamelan movafeg": 5,
    "very dissatisfied": 1, "dissatisfied": 2, "satisfied": 4, "very satisfied": 5,
    "strongly oppose": 1, "oppose": 2, "support": 4, "strongly support": 5,
}

def _to_numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series
    return series.astype(str).str.lower().map(LIKERT_MAP).fillna(3)

@tool
def calculate_net_support(df: Any, column: str) -> Dict[str, Any]:
    """
    Calculates 'Net Support' (Positive % - Negative %) for political analysis.
    Ignores neutrals to show directional opinion.
    
    Args:
        df: Input pandas DataFrame.
        column: The column containing opinion data.
        
    Returns:
        Dict: Contains net_score, positive_pct, negative_pct.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input 'df' must be a pandas DataFrame")

    clean_series = df[column].dropna()
    numeric_series = _to_numeric(clean_series)
    
    total = len(numeric_series)
    if total == 0:
        return {"net_support": 0, "error": "No data"}

    positives = numeric_series[numeric_series >= 4].count()
    negatives = numeric_series[numeric_series <= 2].count()
    
    pos_pct = (positives / total) * 100
    neg_pct = (negatives / total) * 100
    net_score = pos_pct - neg_pct
    
    return {
        "net_support_score": round(net_score, 2),
        "total_positive_percent": round(pos_pct, 2),
        "total_negative_percent": round(neg_pct, 2),
        "interpretation": "Positive" if net_score > 0 else "Negative"
    }

@tool
def polarization_index(df: Any, column: str) -> float:
    """
    Calculates a Polarization Index (0 to 1) for a Likert scale column.
    0 = Consensus, 1 = Extreme Polarization (U-shaped distribution).
    
    Args:
        df: Input pandas DataFrame.
        column: Likert scale column name.
        
    Returns:
        float: The polarization index.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input 'df' must be a pandas DataFrame")

    clean_series = df[column].dropna()
    data = _to_numeric(clean_series)
    
    if len(data) < 2:
        return 0.0
        
    actual_std = data.std()
    max_val = data.max()
    min_val = data.min()
    
    if max_val == min_val:
        return 0.0
        
    max_possible_std = (max_val - min_val) / 2.0
    pi = actual_std / max_possible_std
    
    return round(float(min(pi, 1.0)), 4)

@tool
def segmentation_clustering(df: Any, columns: List[str], n_groups: int = 3) -> Dict[str, Any]:
    """
    Performs K-Means clustering to identify ideological groups or personas.
    
    Args:
        df: Input pandas DataFrame.
        columns: List of columns (questions) to cluster on.
        n_groups: Number of clusters to find.
        
    Returns:
        Dict: Contains the cluster counts and centers (simplified for text output).
        Note: In code generation, assign the result to a variable to get the dataframe.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input 'df' must be a pandas DataFrame")

    data = df[columns].copy().dropna()
    if len(data) < n_groups:
        return {"error": "Not enough data"}

    encoded_data = pd.DataFrame()
    for col in columns:
        encoded_data[col] = _to_numeric(data[col])

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(encoded_data)
    
    kmeans = KMeans(n_clusters=n_groups, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(scaled_data)
    
    # Return summary stats for the tool output
    unique, counts = np.unique(clusters, return_counts=True)
    return {
        "clusters_found": len(unique),
        "counts": dict(zip([int(u) for u in unique], [int(c) for c in counts]))
    }