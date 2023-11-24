"""
Check score procedure
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from joblib import load
import src.all_functions
import logging


def check_score():
    
    """
    model score checking 
    """
    
    df = pd.read_csv('./data/prepared_data/census.csv')
    
    _, test = train_test_split(df, test_size=0.20)
    
    trained_model = load('./model/model.joblib')
    lb = load('./model/lb.joblib')
    encoder = load('./model/encoder.joblib')
    
    
    slice_lst = []

    for col in src.all_functions.get_cat_features():
      for cat in test[col].unique():
        
        temp_df = test[test[col]==cat]
        
        X_test, y_test, _ ,_ = src.all_functions.process_data(temp_df, categorical_features =src.all_functions.get_cat_features(),
                                  label='salary', encoder=encoder, lb=lb, training=False)
                                  
        y_pred = trained_model.predict(X_test)
        
        pr, rcl, fb  = src.all_functions.compute_model_metrics(y_test, y_pred)
        
        
        line = "[%s->%s] Precision: %s " \
                   "Recall: %s FBeta: %s" % (col, cat, pr, rcl, fb)
        
        logging.info(line)
        slice_lst.append(line)
        
  
    with open('./model/slice_output.txt', 'w') as f:
      for slice_value in slice_lst:
        f.write(slice_value + '\n')
    
    
        
