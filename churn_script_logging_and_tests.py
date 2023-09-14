import os
import logging
import churn_library as cl
import pytest

logging.basicConfig(
    filename="logs/churn_library.log",
    level=logging.INFO,
    filemode="w",
    format="%(name)s - %(levelname)s - %(message)s",
)


@pytest.fixture
def import_data():
    return cl.import_data("data/BankChurners.csv")


@pytest.fixture
def encoder_helper(import_data):
    return cl.encoder_helper(import_data)


@pytest.fixture
def perform_eda(encoder_helper):
    return cl.perform_eda(encoder_helper)


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
            logging.error(f"FAILED: The file {fig} does not exist in the directory.")
            raise err


def test_encoder_helper(encoder_helper):
    """
    test encoder helper
    """
    try:
        df = encoder_helper
        logging.info("Testing encoder_helper: SUCCESS")
    except Exception as err:
        logging.error("ERROR: Testing import_eda:could not run encoder_helper")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing encoder_helper: The file doesn't appear to have rows and columns"
        )
        raise err


def test_perform_feature_engineering(perform_feature_engineering):
    """
    test perform_feature_engineering
    """


def test_train_models(train_models):
    """
    test train_models
    """


if __name__ == "__main__":
    pass
