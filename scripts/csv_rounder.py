import argparse
from pathlib import Path

import pandas as pd


def round_sig(x, sig=3):
    """Round a number to N decimal places."""
    if pd.isnull(x):
        return x
    try:
        x = float(x)
    except Exception:
        return x
    return round(x, sig)


def round_csv(input_path, output_path, sig=3):
    df = pd.read_csv(input_path)
    df_rounded = df.applymap(lambda x: round_sig(x, sig))
    df_rounded.to_csv(output_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Round all values in a CSV to N significant digits."
    )
    parser.add_argument("input_csv", type=Path, help="Input CSV file path")
    parser.add_argument("output_csv", type=Path, help="Output CSV file path")
    parser.add_argument(
        "--sig", type=int, default=3, help="Number of significant digits (default: 3)"
    )
    args = parser.parse_args()
    round_csv(args.input_csv, args.output_csv, args.sig)
