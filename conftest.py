import logging

# Configure logger and log directory
logging.basicConfig(
    filename="logs/churn_library.log",
    level=logging.INFO,
    filemode="w",
    format="%(name)s - %(levelname)s - %(message)s",
)
