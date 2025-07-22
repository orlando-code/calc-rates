import itertools
import string

import pandas as pd


def uniquify_repeated_values(vals: list, uniquify_str: str = "LOC") -> list:
    """
    Append a unique suffix to repeated values in a list.

    Parameters:
        vals (list): List of values.

    Returns:
        list: List of values with unique suffixes.
    """

    def zip_letters(letters: list[str]) -> list[str]:
        """Zip a list of strings with uppercase letters."""
        al = string.ascii_uppercase
        return (
            [f"-{uniquify_str}-".join(i) for i in zip(letters, al)]
            if len(letters) > 1
            else letters
        )

    return [j for _, i in itertools.groupby(vals) for j in zip_letters(list(i))]


def safe_to_numeric(col):
    """Convert column to numeric if possible, otherwise return as is."""
    try:
        return pd.to_numeric(col)
    except (ValueError, TypeError):
        return col  # return original column if conversion fails
