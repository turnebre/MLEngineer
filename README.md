# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
This project contains an ML model that was migrated from a .ipynb to a .py file for production use. It followns PEP8 coding practices and documentation on how to run the model and tests from the cli.

## Files and data description
1. #### **churn_library.py**
    The churn_library.py is a library of functions to find customers who are likely to churn.

2. #### **churn_script_logging_and_tests.py**
    Contain unit tests for the churn_library.py functions.  Uses the basic assert statements that test functions work properly. The goal of test functions is to checking the returned items aren't empty or folders where results should land have results after the function has been run.

3. #### **conftest.py**
    Contains logging configuration needed to output logs using pytest.

4. #### **churn_notebook.ipynb**
    Original notebook for training model.

## Running Files
To run the model and tests, perform the following:
1. Install **requirements.txt** by running `pip install -r requirements.txt`
2. Run model by running `python churn_library.py`
3. Test model ran correctly by running `pytest churn_script_logging_and_tests.py`

The output should save images to the `images/` directory, save the models themselves to the `models/` directory, and log test results to the `logs/` directory. 



