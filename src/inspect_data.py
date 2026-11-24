"""Step 3 - look at the raw data BEFORE touching it.  Run:  python -m src.inspect_data"""
import pandas as pd

from src.config import RAW_PATH


def main() -> None:
    df = pd.read_csv(RAW_PATH)  # let pandas guess types, so you can see what goes wrong

    print("SHAPE:", df.shape)
    print("\nDTYPES (numbers stored as text show up as object/str):")
    print(df.dtypes)
    print("\nFIRST ROWS:")
    print(df.head(8).to_string())
    print("\nMISSING VALUES PER COLUMN:")
    print(df.isna().sum())
    print("\nEXACT DUPLICATE ROWS:", df.duplicated().sum())

    for col in ["Gender", "Contract", "Payment Method", "Internet Service", "Churned"]:
        print(f"\nVALUES IN '{col}':")
        print(df[col].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()