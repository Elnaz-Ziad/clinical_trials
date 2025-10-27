import pandas as pd
from src.constants import month_mapping

def create_standard_date(row):
    day = row['ae_grade_end_date_day']
    month = row['ae_grade_end_date_month']
    year = row['ae_grade_end_date_year']
    
    # Check for valid day, month, and year
    if pd.notnull(day) and pd.notnull(month) and pd.notnull(year):
        # Check if the values are 'Unknown'
        if day != 'Unknown' and month in month_mapping:
            try:
                # Convert to datetime
                return pd.to_datetime(f"{int(year)}-{month_mapping[month]}-{int(day)}", errors='raise')
            except ValueError:
                return pd.NaT
        else:
            return pd.NaT
    else:
        return pd.NaT




def detect_ctcae_version(text):
    if pd.isna(text):
        return None
    text_str = str(text).strip()
    if text_str.isupper():
        return 3
    elif text_str.endswith('.'):
        return 5
    else:
        return 4




def summarize_missing_values(df):
    """
    Compute and return missing counts and percentages for each column in a DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to analyze.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing columns:
        - 'missing_count': number of missing values
        - 'missing_percent': percentage of missing values (rounded to 1 decimal)
        Sorted in descending order by missing_count, excluding columns with no missing values.
    """
    missing_counts = df.isnull().sum()
    missing_perc = (missing_counts / len(df) * 100).round(1)

    missing_summary = (
        pd.DataFrame({
            "missing_count": missing_counts,
            "missing_percent": missing_perc
        })
        .query("missing_count > 0")
        .sort_values("missing_count", ascending=False)
    )

    return missing_summary




def parse_mixed_date(val):
    val_str = str(val)
    if '/' in val_str:
        return pd.to_datetime(val_str, dayfirst=True, errors='coerce')  # Slash ⇒ dd/mm/yyyy
    elif '-' in val_str:
        return pd.to_datetime(val_str, dayfirst=False, errors='coerce')  # Dash ⇒ yyyy-mm-dd
    else:
        return pd.to_datetime(val_str, errors='coerce')  # Catch-all fallback




def preprocessing_for_arm(
    df: pd.DataFrame,
    *,
    level: str = "mapped_term",   # choose "mapped_term" or "mapped_soc"
    basket_cols=("study_name", "mrn")
) -> pd.DataFrame:
    """
    Preprocess raw AE data into a basket × item boolean one-hot matrix.
    """
    df = df.copy()

    # Clean basket columns & item level
    for c in basket_cols:
        df[c] = df[c].astype("string").str.strip()
    df[level] = df[level].astype("string").str.strip()

    # Build basket_id
    df["basket_id"] = df[basket_cols[0]] + "_" + df[basket_cols[1]]

    # Drop duplicate item occurrences per basket
    unique_pairs = df[["basket_id", level]].drop_duplicates().dropna()

    # Pivot to basket × item matrix
    basket_matrix = (
        unique_pairs
        .assign(value=True)
        .pivot_table(index="basket_id", columns=level, values="value", fill_value=False)
        .astype(bool)
        .sort_index(axis=1)   # sort columns alphabetically
    )

    return basket_matrix
