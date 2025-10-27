import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, apriori, association_rules

def find_frequent_itemsets_and_rules(
    basket_matrix: pd.DataFrame,
    *,
    algo: str = "fpgrowth",       # choose here
    min_support: float = 0.01,
    metric: str = "confidence",
    min_threshold: float = 0.6
):
    """
    Mine frequent itemsets and association rules.
    Pass algo="fpgrowth" (fast) or algo="apriori".
    """
    X = basket_matrix.astype(bool)

    if algo == "fpgrowth":
        frequent_itemsets = fpgrowth(X, min_support=min_support, use_colnames=True)
    elif algo == "apriori":
        frequent_itemsets = apriori(X, min_support=min_support, use_colnames=True)
    else:
        raise ValueError("algo must be 'fpgrowth' or 'apriori'")

    if frequent_itemsets.empty:
        return frequent_itemsets, pd.DataFrame()

    rules = association_rules(frequent_itemsets, metric=metric, min_threshold=min_threshold)

    # Add readable string columns
    rules["antecedents_str"] = rules["antecedents"].apply(lambda x: ", ".join(sorted(x)))
    rules["consequents_str"] = rules["consequents"].apply(lambda x: ", ".join(sorted(x)))

    return frequent_itemsets, rules