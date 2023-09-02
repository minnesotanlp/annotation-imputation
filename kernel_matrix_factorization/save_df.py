from make_path import make_path
import pandas as pd

# constants for saving the datasets
QUOTE_CHAR = '"'
# Quoting constants from https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html
QUOTE_ALL = 1

def save_df(df: pd.DataFrame, file: str):
    make_path(file)
    while True:
        try:
            df.to_csv(file, index=False, quoting=QUOTE_ALL, quotechar=QUOTE_CHAR)
            break
        except Exception as e:
            print(f"Error saving dataframe to {file}: {e}")
            user_response = input("Would you like to try again? (y/n): ").lower()
            if user_response == "n":
                print("Will not try again")
                break
            else:
                print("Trying again...")

