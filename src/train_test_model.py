"""
Train model procedure 
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from joblib import dump
import src.all_functions

def train_test_model():
    """
    Execute model trainig 
    """

    df = pd.read_csv("./data/prepared_data/census.csv")

    train, _ = train_test_split(df, test_size=0.20)

    X_train, y_train, encoder, lb = src.all_functions.process_data(
    train, categorical_features=src.all_functions.get_cat_features(), label="salary", training=True
    )

    trained_model = src.all_functions.train_model(X_train, y_train)

    dump(trained_model, './model/model.joblib')
    dump(encoder, './model/encoder.joblib')
    dump(lb, './model/lb.joblib')


