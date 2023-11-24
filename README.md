# ML pipeline to expose API on Heroku

Udacity project about creating a pipeline to train a model and publish it with a public API on Heroku.

## Developer environment

### Githooks

Flake8 githooks needs to be installed on local development environment with following steps:

* Install precommit binary following https://pre-commit.com/#installation
* Execute `pre-commit install`

## Code/Model testing

Test suite can be executed by `pytest`

## Procedures

### Basic cleaning procedure

Cleaning data can be done by `python main.py --action basic_cleaning`

### Train/test model procedure

Model training and test can be done by `python main.py --action train_test_model`

### Check model score procedure

Check score on latest dvs saved model can be done by `python main.py --action check_score`

### Run entire pipeline

To run the entire pipeline in sequence, use `python main.py --action all` or `python main.py`

### Serve the API on local

If testing FastAPi serving on local is needed, execute `uvicorn api_server:app --reload`

### Check Heroku deployed API

Check Heroku deployed APi using `python check_heroku_api.py`

## CI/CD

Every step is automated so on Pull Request [Test pipeline](.github/workflows/test.yaml) is triggered.
Pipeline pulls data from DVC and execute Flake8 + pytest doing every test.

On Merge [Deploy pipeline](.github/workflows/deploy.yaml) is executed.
Model is trained, score is checked, data.dvc is then autocommitted and Heroku will be able to automatically build the app.
