import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

# Set up logger to print to console when running this script
logr = logging.getLogger(__name__)
logr.setLevel(logging.DEBUG)
logr.addHandler(logging.StreamHandler())
logr.handlers[0].setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)

# Load environment variables
load_dotenv()

# Constants
DB_CONFIG = {
    "database": "yelp_db",
    "user": "postgres",
    "password": os.getenv("db_password"),
    "host": "localhost",
    "port": "5432",
}
DATA_PATH = Path(os.getenv("data_dir")) / "cleaned"


def push_to_db(data_dir: Path, db_config: dict):
    """Push the cleaned data to the database.

    Args:
        data_dir (Path): path to the directory containing the cleaned data
        db_config (dict): configuration for the database connection
    """

    engine = create_engine(
        f'postgresql+psycopg2://{db_config["user"]}:{db_config["password"]}@'
        f'{db_config["host"]}:{db_config["port"]}/{db_config["database"]}'
    )

    # Loading the businesses data
    try:
        logr.info("Loading the businesses data")
        business_df = pd.read_csv(data_dir / "business.csv")

        cats = [i for i in business_df.columns if i.startswith("cat_")]
        attrs = [i for i in business_df.columns if i.startswith("attr_")]
        main_cols = [i for i in business_df.columns if i not in cats + attrs]

        # Divide cats into two groups to avoid exceeding the maximum number of parameters in a query by
        # randomly selecting half of the columns
        cats_1 = list(np.random.choice(cats, int(len(cats) / 2), replace=False))
        cats_2 = [i for i in cats if i not in cats_1]

        attrs_df = business_df[["business_id"] + attrs].copy()
        cats_1_df = business_df[["business_id"] + cats_1].copy()
        cats_2_df = business_df[["business_id"] + cats_2].copy()
        business_df = business_df[main_cols].copy()

        logr.info("Pushing the businesses data to the database")
        business_df.to_sql(
            "business", con=engine, if_exists="replace", index=False, chunksize=1000
        )
        attrs_df.to_sql(
            "business_attr", con=engine, if_exists="replace", index=False, chunksize=1000
        )
        cats_1_df.to_sql(
            "business_cat1", con=engine, if_exists="replace", index=False, chunksize=1000
        )
        cats_2_df.to_sql(
            "business_cat2", con=engine, if_exists="replace", index=False, chunksize=1000
        )
        logr.info("Finished pushing the businesses data to the database")

        # Free up memory
        del business_df, attrs_df, cats_1_df, cats_2_df
    except FileNotFoundError:
        logr.error("Could not find 'business.csv' in the data directory")
        raise

    # Loading the reviews data
    try:
        logr.info("Loading the reviews data")
        review_df = pd.read_csv(data_dir / "review.csv")

        logr.info("Pushing the reviews data to the database")
        review_df.to_sql("review", con=engine, if_exists="replace", index=False, chunksize=1000)

        logr.info("Finished pushing the reviews data to the database")
        del review_df
    except FileNotFoundError:
        logr.error("Could not find 'review.csv' in the data directory")
        raise

    # Loading the users data
    try:
        logr.info("Loading the users data")
        user_df = pd.read_csv(data_dir / "user.csv")

        logr.info("Pushing the users data to the database")
        user_df.to_sql("user", con=engine, if_exists="replace", index=False, chunksize=1000)

        logr.info("Finished pushing the users data to the database")
        del user_df
    except FileNotFoundError:
        logr.error("Could not find 'user.csv' in the data directory")
        raise


if __name__ == "__main__":
    logr.info("Starting preprocessor script")
    logr.debug(f"Data path: {DATA_PATH}")
    logr.debug(f"Database config: {DB_CONFIG}")

    try:
        push_to_db(DATA_PATH, DB_CONFIG)
    except Exception as e:
        logr.error(f"An error occurred: {e}")
