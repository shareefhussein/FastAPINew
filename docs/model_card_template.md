# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
### the model is a Gradinet Boosting Classifer, and uses the default hyperparameters in Scikit learn.

## Intended Use
### the model is used to predict the salray of a person based on financial features of this person.

## Training Data
### the data is downloaded from https://archive.ics.uci.edu/ml/datasets/census+income with training size 80% 

## Evaluation Data
### the data is downloaded from https://archive.ics.uci.edu/ml/datasets/census+income with test size 20% 

## Metrics
### the model was evaluated using accuracy score, the score value around 0.81 

## Ethical Considerations
### the dataset contains data related to race, gender, and sex.

## Caveats and Recommendations
### gender classes are given as (male/not male), which we include and (male/female).
