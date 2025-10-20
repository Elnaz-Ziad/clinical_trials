import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Optional
import textwrap
from math import ceil
from matplotlib.patches import Patch


def plot_records_per_grade(df, grade_col='ae_grade'):
    """
    Plot the number of records per AE grade.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the AE grade column.
    grade_col : str, optional
        Column name containing AE grades (default: 'ae_grade').

    Returns
    -------
    None
        Displays a bar plot showing the number of records per grade.
    """
    # --- Count number of records per AE grade ---
    grade_counts = df[grade_col].value_counts().sort_index()

    # --- Define colors (green for <3, red for ≥3) ---
    colors = ['#C1E1C1' if grade < 3 else '#FFB6B6' for grade in grade_counts.index]

    # --- Plot ---
    plt.figure(figsize=(8, 6))
    bars = plt.bar(grade_counts.index.astype(str), grade_counts.values,
                   color=colors, edgecolor='black')

    # --- Add labels ---
    for bar in bars:
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{int(bar.get_height()):,}",
            ha='center', va='bottom', fontsize=12
        )

    # --- Labels and styling ---
    plt.xlabel("AE Grade", fontsize=14)
    plt.ylabel("Number of Records", fontsize=14)
    plt.title("Number of Records per AE Grade", fontsize=16, fontweight='bold', pad=10)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.show()




def plot_records_per_year_side_by_side(
    df_all,
    df_high,
    date_col='ae_grade_start_date',
    title="Number of Records per Year (Based on AE Grade Start Date)",
    color_all="#A7C7E7",
    color_high="#FFB6B6",
    figsize=(14, 6)
):
    """
    Plot the number of records per year for two datasets (e.g., all vs. high-grade),
    displayed side by side with independent axes.

    Parameters
    ----------
    df_all : pandas.DataFrame
        Full dataset.
    df_high : pandas.DataFrame
        Subset (e.g., grade ≥3 dataset).
    date_col : str, optional
        Column name containing the date of interest (default: 'ae_grade_start_date').
    title : str, optional
        Main title for the figure.
    color_all : str, optional
        Bar color for all records.
    color_high : str, optional
        Bar color for high-grade records.
    figsize : tuple, optional
        Figure size (default: (14, 6)).

    Returns
    -------
    None
        Displays a matplotlib bar plot.
    """

    def yearly_counts(df):
        temp = df.copy()
        temp[date_col] = pd.to_datetime(temp[date_col], errors='coerce')
        return temp[date_col].dt.year.value_counts().sort_index()

    # Compute yearly counts
    records_all = yearly_counts(df_all)
    records_high = yearly_counts(df_high)

    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # --- Left plot: All records ---
    axes[0].bar(records_all.index, records_all.values, color=color_all, edgecolor='black')
    for bar in axes[0].patches:
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{int(bar.get_height()):,}",
            ha='center', va='bottom', fontsize=11
        )
    axes[0].set_title("All Records", fontsize=15, fontweight='bold')
    axes[0].set_xlabel("Year", fontsize=13)
    axes[0].set_ylabel("Number of Records", fontsize=13)
    axes[0].tick_params(axis='x', rotation=45, labelsize=11)
    axes[0].tick_params(axis='y', labelsize=11)

    # --- Right plot: High-grade records ---
    axes[1].bar(records_high.index, records_high.values, color=color_high, edgecolor='black')
    for bar in axes[1].patches:
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{int(bar.get_height()):,}",
            ha='center', va='bottom', fontsize=11
        )
    axes[1].set_title("Grade ≥3 Records", fontsize=15, fontweight='bold')
    axes[1].set_xlabel("Year", fontsize=13)
    axes[1].set_ylabel("Number of Records", fontsize=13)
    axes[1].tick_params(axis='x', rotation=45, labelsize=11)
    axes[1].tick_params(axis='y', labelsize=11)

    # --- Shared title and layout ---
    fig.suptitle(title, fontsize=17, fontweight='bold', y=1.03)
    plt.tight_layout()
    plt.show()




def plot_ae_category_distribution(
    df_all,
    df_high,
    col="mapped_soc",
    title="Distribution of AE Categories",
    color_all="skyblue",
    color_high="#FFB6B6",
    figsize=(16, 20)
):
    """
    Plot the distribution of AE categories (e.g., mapped SOCs) for all records
    and for grade ≥3 records in two stacked horizontal bar charts.

    Parameters
    ----------
    df_all : pandas.DataFrame
        DataFrame containing all records.
    df_high : pandas.DataFrame
        DataFrame containing high-grade (e.g., grade ≥3) records.
    col : str, optional
        Column name to use for counting (default: 'mapped_soc').
    title : str, optional
        Main title of the figure.
    color_all : str, optional
        Color for the "All Records" bars (default: 'skyblue').
    color_high : str, optional
        Color for the "High-Grade Records" bars (default: '#FFB6B6').
    figsize : tuple, optional
        Figure size (default: (16, 20)).

    Returns
    -------
    None
        Displays the plot.
    """

    def soc_distribution(df, col):
        counts = df[col].value_counts(ascending=False)
        percentages = (counts / counts.sum() * 100).round(1)
        labels = [f"{count:,} ({pct}%)" for count, pct in zip(counts, percentages)]
        return counts, labels

    # --- Compute counts and labels ---
    counts_all, labels_all = soc_distribution(df_all, col)
    counts_high, labels_high = soc_distribution(df_high, col)

    # --- Create figure with two rows (stacked vertically) ---
    fig, axes = plt.subplots(2, 1, figsize=figsize)

    # ---------- Plot 1: All Records ----------
    bars = axes[0].barh(counts_all.index, counts_all.values,
                        color=color_all, edgecolor='black')
    for bar, label in zip(bars, labels_all):
        axes[0].text(bar.get_width() + counts_all.max() * 0.01,
                     bar.get_y() + bar.get_height() / 2,
                     label, va='center', fontsize=16)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Count", fontsize=16)
    axes[0].set_ylabel("AE Categories", fontsize=16)
    axes[0].set_title("All Records", fontsize=20, pad=10)
    axes[0].set_xlim(0, counts_all.max() * 1.3)
    axes[0].tick_params(axis='x', labelsize=16)
    axes[0].tick_params(axis='y', labelsize=16)

    # ---------- Plot 2: High-Grade Records ----------
    bars2 = axes[1].barh(counts_high.index, counts_high.values,
                         color=color_high, edgecolor='black')
    for bar, label in zip(bars2, labels_high):
        axes[1].text(bar.get_width() + counts_high.max() * 0.01,
                     bar.get_y() + bar.get_height() / 2,
                     label, va='center', fontsize=16)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Count", fontsize=16)
    axes[1].set_ylabel("AE Categories", fontsize=16)
    axes[1].set_title("Grade ≥3 Records", fontsize=20, pad=10)
    axes[1].set_xlim(0, counts_high.max() * 1.3)
    axes[1].tick_params(axis='x', labelsize=16)
    axes[1].tick_params(axis='y', labelsize=16)

    # ---------- Main Title ----------
    fig.suptitle(title, fontsize=22, fontweight='bold', y=0.95)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.show()


def build_rank_table(
    ct: pd.DataFrame,
    ct_high_grades: pd.DataFrame,
    category_col: str = "mapped_soc",
    save_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Build rank table for AE categories in all records vs. grade ≥3 records.
    Optionally save to CSV. Returns the full rank table.

    Parameters
    ----------
    ct : DataFrame
        Full dataset.
    ct_high_grades : DataFrame
        High-grade subset (e.g., ae_grade >= 3).
    category_col : str
        Column with AE category (default: 'mapped_soc').
    save_path : str or None
        If provided, CSV will be written to this path (index=False).

    Returns
    -------
    DataFrame
        Rank summary with clear, snake_case column names.
    """
    # Step 1: counts
    counts_all = ct[category_col].value_counts().reset_index()
    counts_all.columns = [category_col, "count_all"]

    counts_high = ct_high_grades[category_col].value_counts().reset_index()
    counts_high.columns = [category_col, "count_high"]

    # Step 2: ranks
    counts_all["rank_all"] = counts_all["count_all"].rank(method="dense", ascending=False).astype(int)
    counts_high["rank_high"] = counts_high["count_high"].rank(method="dense", ascending=False).astype(int)

    # Step 3: merge
    rank_table = pd.merge(counts_all, counts_high, on=category_col, how="outer").fillna(0)

    # Step 4: ints
    rank_table[["count_all", "count_high", "rank_all", "rank_high"]] = (
        rank_table[["count_all", "count_high", "rank_all", "rank_high"]].astype(int)
    )

    # Step 5: deltas & trend
    rank_table["rank_change"] = rank_table["rank_high"] - rank_table["rank_all"]
    rank_table["trend"] = rank_table["rank_change"].apply(
        lambda x: "↑ improved" if x < 0 else ("↓ dropped" if x > 0 else "→ same")
    )

    # Step 6: sort
    rank_table = rank_table.sort_values("rank_all").reset_index(drop=True)

    # Step 7: rename (final output only)
    rank_table = rank_table.rename(columns={
        category_col: "ae_category",
        "count_all": "n_all_grades",
        "rank_all": "rank_all_grades",
        "count_high": "n_grade_3plus",
        "rank_high": "rank_grade_3plus",
        "trend": "rank_trend",
        "rank_change": "rank_difference",
    })

    # optional save
    if save_path:
        rank_table.to_csv(save_path, index=False)

    return rank_table



def summarize_other_specify_terms(ct: pd.DataFrame, grade_col: str = "ae_grade", term_col: str = "mapped_term") -> pd.DataFrame:
    """
    Summarize how many AE terms end with "Other, specify" for all grades and for grade ≥3.

    Parameters
    ----------
    ct : DataFrame
        The full dataset containing AE grades and terms.
    grade_col : str, optional
        Column name for AE grade (default: "ae_grade").
    term_col : str, optional
        Column name for AE term (default: "mapped_term").

    Returns
    -------
    DataFrame
        Summary table with counts and percentages for all grades and for grade ≥3.
    """

    def summarize(df):
        counts = df[term_col].apply(
            lambda x: 'Ends with "Other, specify"' 
            if isinstance(x, str) and x.lower().strip().endswith("other, specify")
            else 'Does not end with "Other, specify"'
        ).value_counts()
        pct = counts / counts.sum() * 100
        return counts.astype(str) + ' (' + pct.round(1).astype(str) + '%)'

    # Summaries
    overall = summarize(ct)
    high_grade = summarize(ct[ct[grade_col] >= 3])

    # Combine into summary DataFrame
    summary_df = pd.DataFrame({
        "all_grades": overall,
        "grade_3plus": high_grade
    }).fillna("0 (0.0%)")

    return summary_df





def plot_top_terms_per_category(df, df_name="ct", top_soc=30, top_terms=7):
    """
    Plot top AE terms per AE category (mapped SOC), highlighting 'Other, specify' terms.
    Automatically adjusts color scheme depending on dataset type (all vs. high-grade).

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset containing mapped SOC and mapped term columns.
    df_name : str, optional
        Name of dataset ('ct' for all grades or 'ct_high_grades' for high-grade subset).
    top_soc : int, optional
        Number of top categories to show (default: 30).
    top_terms : int, optional
        Number of top terms per category to show (default: 7).

    Returns
    -------
    None
        Displays the bar chart grid.
    """

    # --- Automatically pick color scheme based on dataset type ---
    if "high" in df_name.lower():
        normal_color = "#FFB6B6"   # light red for high grades
    else:
        normal_color = "skyblue"   # blue for all grades

    other_spec_color = "#FFD580"   # pastel orange

    # --- Helper for wrapping long labels ---
    def wrap_labels(labels, width=28):
        return ['\n'.join(textwrap.wrap(str(x), width=width)) for x in labels]

    # --- Top categories ---
    top_soc_list = df['mapped_soc'].value_counts().head(top_soc).index.tolist()

    # --- Precompute max count for consistent x-axis ---
    global_max = 0
    per_soc_term_counts = {}
    for soc in top_soc_list:
        sub = df[df['mapped_soc'] == soc]
        term_counts = sub['mapped_term'].value_counts().head(top_terms)
        per_soc_term_counts[soc] = (sub, term_counts)
        if len(term_counts):
            global_max = max(global_max, term_counts.max())

    # --- Create 2-column grid ---
    n = len(top_soc_list)
    rows = ceil(n / 2)
    fig, axes = plt.subplots(rows, 2, figsize=(14, 4.2 * rows))
    axes = np.atleast_1d(axes).ravel()

    for i, soc in enumerate(top_soc_list):
        ax = axes[i]
        sub, term_counts = per_soc_term_counts[soc]
        term_pct = (term_counts / len(sub) * 100).round(1)

        # --- Color terms (highlight 'Other, specify') ---
        colors = [
            other_spec_color if str(term).lower().strip().endswith("other, specify") else normal_color
            for term in term_counts.index
        ]

        # --- Plot bars ---
        bars = ax.barh(
            wrap_labels(term_counts.index, width=28),
            term_counts.values,
            color=colors,
            edgecolor="black"
        )

        # --- Add labels ---
        for bar, count, pct in zip(bars, term_counts.values, term_pct.values):
            ax.text(
                bar.get_width() + global_max * 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"{int(count):,} ({pct}%)",
                va='center',
                fontsize=13
            )

        ax.invert_yaxis()
        ax.set_xlabel("Count", fontsize=14)
        ax.set_ylabel("AE term", fontsize=14)
        ax.set_title(f'{soc}  (n={len(sub):,})', fontsize=13, fontweight='bold', pad=10)
        ax.tick_params(axis='both', labelsize=12)
        ax.set_xlim(0, global_max * 1.25)

    # Hide empty subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # --- Main title and legend ---
    plt.suptitle(
        "Top AE Terms per Category",
        fontsize=22,
        fontweight='bold',
        y=1,
        ha='center'
    )

    legend_patches = [
        Patch(facecolor=normal_color, edgecolor='black', label='Other terms'),
        Patch(facecolor=other_spec_color, edgecolor='black', label='Ends with "Other, specify"'),
    ]
    fig.legend(
        handles=legend_patches,
        loc='upper center',
        ncol=2,
        frameon=False,
        fontsize=11,
        bbox_to_anchor=(0.5, 0.99)
    )

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()








def plot_number_of_records_and_studies_by_disease_site(
    df,
    site_col="disease_site_group",
    study_col="study_name",
    figsize=(12, 6)
):
    """
    Plot the number of total records and unique studies per disease site side by side.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing disease site and study columns.
    site_col : str, optional
        Column name representing disease site (default: 'disease_site_group').
    study_col : str, optional
        Column name representing study name (default: 'study_name').
    figsize : tuple, optional
        Figure size (default: (12, 6)).

    Returns
    -------
    None
        Displays the plots.
    """

    # --- Count total records per disease site ---
    record_counts = df[site_col].value_counts()

    # --- Count unique studies per disease site ---
    unique_studies = (
        df.groupby(site_col)[study_col]
        .nunique()
        .sort_values(ascending=False)
    )

    # --- Create subplots (1 row, 2 columns) ---
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # --- Left: total records ---
    record_counts.plot(kind='barh', ax=axes[0], color='lightpink', edgecolor='black')
    axes[0].set_title('Number of Records per Disease Site', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Count', fontsize=12)
    axes[0].set_ylabel('Disease Site', fontsize=12)
    axes[0].invert_yaxis()

    # Add labels
    for i, v in enumerate(record_counts):
        axes[0].text(v + record_counts.max() * 0.01, i, str(v), va='center', fontsize=10)

    # --- Right: unique studies ---
    unique_studies.plot(kind='barh', ax=axes[1], color='thistle', edgecolor='black')
    axes[1].set_title('Number of Unique Studies per Disease Site', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Unique Study Count', fontsize=12)
    axes[1].invert_yaxis()

    # Add labels
    for i, v in enumerate(unique_studies):
        axes[1].text(v + unique_studies.max() * 0.01, i, str(v), va='center', fontsize=10)

    # Adjust layout and limits
    axes[0].set_xlim(0, record_counts.max() * 1.25)
    axes[1].set_xlim(0, unique_studies.max() * 1.25)
    plt.tight_layout()
    plt.show()
