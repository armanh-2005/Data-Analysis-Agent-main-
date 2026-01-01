import numpy as np
import math
# ... keep existing imports ...

@tool
def plot_radar_chart(df: Any, categories_col: str, values_cols: List[str]) -> str:
    """
    Generates a Radar Chart (Spider Plot) to compare groups across multiple dimensions.
    
    Use this to compare "Personas" or "Parties" across multiple issues.
    Example: Compare 'Reformists' vs 'Conservatives' on [Economy, Freedom, Security, Tradition].
    
    Args:
        df: Input DataFrame.
        categories_col: The column defining the groups (e.g., 'Party', 'Cluster').
        values_cols: List of numeric columns (scores) to compare (e.g., ['score_econ', 'score_social']).
        
    Returns:
        str: "Plot created" message.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be DataFrame")

    # Calculate means for each group across the value columns
    # We dropna to ensure clean means
    summary = df.groupby(categories_col)[values_cols].mean().dropna()
    
    if summary.empty:
        return "Not enough data for radar chart."

    # Prepare data for plotting
    labels = list(summary.columns)
    num_vars = len(labels)
    
    # Compute angle for each axis
    angles = [n / float(num_vars) * 2 * math.pi for n in range(num_vars)]
    angles += angles[:1] # Close the loop
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    # Draw one line per group
    for group_name, row in summary.iterrows():
        values = row.tolist()
        values += values[:1] # Close the loop
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=str(group_name))
        ax.fill(angles, values, alpha=0.1)
    
    # Fix axis to top
    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)
    
    # Draw axis labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title(f"Comparison by {categories_col}")
    
    return "Radar chart created"


@tool
def plot_grouped_distribution(df: Any, metric_col: str, group_col: str) -> str:
    """
    Generates a Violin Plot combined with a Box Plot.
    
    This is superior to simple averages for political analysis because it shows 
    POLARIZATION. A wide violin means no consensus; a split violin means polarized groups.
    
    Args:
        df: Input DataFrame.
        metric_col: Numeric variable (e.g., 'Satisfaction Score', 'Age').
        group_col: Categorical variable (e.g., 'Gender', 'Vote History').
        
    Returns:
        str: "Plot created" message.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be DataFrame")

    plt.figure(figsize=(10, 6))
    
    # Violin plot shows the density/shape
    sns.violinplot(data=df, x=group_col, y=metric_col, inner=None, color=".8")
    
    # Strip plot shows the actual dots (jittered)
    sns.stripplot(data=df, x=group_col, y=metric_col, alpha=0.5, jitter=True)
    
    plt.title(f"Distribution of {metric_col} by {group_col}")
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    return "Violin distribution plot created"


@tool
def plot_stacked_bar(df: Any, primary_col: str, stack_col: str) -> str:
    """
    Generates a 100% Stacked Bar Chart.
    
    Essential for demographic profiles.
    Example: "What is the education breakdown (Stack) within each Income Bracket (Primary)?"
    
    Args:
        df: Input DataFrame.
        primary_col: The X-axis category (e.g., 'Income Level').
        stack_col: The category to stack (e.g., 'Education', 'Sentiment').
        
    Returns:
        str: "Plot created" message.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be DataFrame")

    # Create Crosstab
    ct = pd.crosstab(df[primary_col], df[stack_col], normalize='index') * 100
    
    # Plot
    ax = ct.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='viridis')
    
    plt.title(f"{stack_col} distribution by {primary_col}")
    plt.ylabel("Percentage (%)")
    plt.legend(title=stack_col, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Annotate percentages if readable
    for c in ax.containers:
        ax.bar_label(c, fmt='%.0f%%', label_type='center', padding=0, color='white', fontsize=8)
        
    return "Stacked bar plot created"