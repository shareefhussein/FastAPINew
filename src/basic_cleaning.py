"""
Cleaning data procedure
"""

import pandas as pd


def clean_data(df):
    """
    clean the data for eda and model
    """

    df_columns = [x.strip(' ') for x in df.columns.tolist()]
    
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)


    df.columns = df_columns
    df.replace({'?': None}, inplace=True)
    df.dropna(inplace=True)
    df.drop("fnlgt", axis="columns", inplace=True)
    df.drop("education-num", axis="columns", inplace=True)
    df.drop("capital-gain", axis="columns", inplace=True)
    df.drop("capital-loss", axis="columns", inplace=True)
    return df

def execute_cleaning():
    """
    Execute data cleaning
    """
    df = pd.read_csv("./data/raw/census.csv")
    
    df = clean_data(df)

    print('Export prepared data')
    df.to_csv("./data/prepared_data/census.csv", index=False)
    
