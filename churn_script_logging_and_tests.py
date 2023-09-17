"""
Test file to ensure churn_library.py correctly ran and outputted results to the appropriate directories.
Author: Brendan Turner
Date: 09/17/2023
"""
import os
import logging
import churn_library as cl
import pytest


@pytest.fixture
def import_data():
    """
    Fixture for import_data function
    """
    return cl.import_data("data/BankChurners.csv")


def test_import(import_data):
    """
    test data import - this example is completed for you to assist with the other test functions
    """
    try:
        df = import_data
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns"
        )
        raise err


@pytest.fixture
def encoder_helper(import_data):
    """
    Fixture for encoder_helper function
    """
    return cl.encoder_helper(import_data)


def test_encoder_helper(encoder_helper):
    """
    test encoder helper
    """
    df = encoder_helper

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except Exception as err:
        logging.error(
            "ERROR: The file doesn't appear to have rows and columns")
        raise err


@pytest.fixture
def perform_eda(encoder_helper):
    """
    Fixture for perform_eda function
    """
    return cl.perform_eda(encoder_helper)


def test_eda(perform_eda):
    """
    test perform eda function
    """

    perform_eda

    # Define the directory path
    directory_path = "images"

    figures = [
        "churn_histogram",
        "customer_age_histogram",
        "marital_status_count",
        "transactions_distribution",
        "confusion_matrix",
    ]

    for fig in figures:
        file_name = f"{fig}.png"
        file_path = os.path.join(directory_path, file_name)

        # Check if the file exists in the directory
        try:
            assert os.path.isfile(file_path)
            logging.info(f"SUCCESS: The file {fig} exists in the directory.")
        except AssertionError as err:
            logging.error(
                f"FAILED: The file {fig} does not exist in the directory.")
            raise err


@pytest.fixture
def perform_feature_engineering(encoder_helper):
    """
    Fixture for perform_feature_engineering function
    """
    return cl.perform_feature_engineering(encoder_helper)


def test_perform_feature_engineering(perform_feature_engineering):
    """
    test perform_feature_engineering
    """

    # Check train and tests sets aren't empty
    try:
        X_train, X_test, y_train, y_test = perform_feature_engineering
        assert X_train.shape[0] > 0 and X_train.shape[1] > 0
        assert X_test.shape[0] > 0 and X_test.shape[1] > 0
        assert y_train.shape[0] > 0
        assert y_test.shape[0] > 0
        logging.info("SUCCESS: All datasets successfully loaded.")
    except Exception as err:
        logging.error(
            "ERROR: Could not load all datasets.  Refer to error for more details"
        )
        raise err


@pytest.fixture
def train_models(perform_feature_engineering):
    """
    Fixture for train_models function
    """
    X_train, X_test, y_train, y_test = perform_feature_engineering
    return cl.train_models(X_train, X_test, y_train, y_test)


def test_train_models(train_models):
    """
    test train_models
    """

    # Check models have been saved
    try:
        train_models
        rfc_model = os.path.join("models", "rfc_model.pkl")
        assert os.path.isfile(rfc_model)
        logistic_model = os.path.join("models", "logistic_model.pkl")
        assert os.path.isfile(logistic_model)
        logging.info("SUCCESS: Models successfully saved")
    except Exception as err:
        logging.error(
            "ERROR: Models did not save. Refer to error for more details.")

    # Check classification_report_image saved reports
    try:
        random_forest_report = os.path.join(
            "images", "random_forest_report.png")
        assert os.path.isfile(random_forest_report)
        logistic_regression_report = os.path.join(
            "images", "logistic_regression_report.png"
        )
        assert os.path.isfile(logistic_regression_report)
        logging.info("SUCCESS: Classification Reports successfully saved.")
    except Exception as err:
        logging.error(
            "ERROR: Classification Reports did not save. Refer to error for more details."
        )

    # Check feature_importance_plot saved plot
    try:
        feature_importance_plot = os.path.join(
            "images", "feature_importance_plot.png")
        assert os.path.isfile(feature_importance_plot)
        logging.info("SUCCESS: Feature Importance Plot successfully saved.")
    except Exception as err:
        logging.error(
            "ERROR: Feature Importance Plot did not save. Refer to error for more details."
        )
        raise err


if __name__ == "__main__":
    pass
