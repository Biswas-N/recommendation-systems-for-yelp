import logging
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from pyspark.sql import SparkSession
from pyspark.sql.functions import concat, explode, lit, lower, regexp_replace, split, trim
from pyspark.sql.types import IntegerType, StringType, StructField, StructType

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
PROJECT_DIR = Path(os.getcwd()).parent
DATA_DIR = PROJECT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
CLEANED_DATA_DIR = DATA_DIR / "cleaned"


def main():
    file_stats = ""
    for file in RAW_DATA_DIR.glob("*.json"):
        file_stats += f"{file.name} - {round(os.path.getsize(file) / 1e9, 2)}GB\n"

    logr.info(f"File stats:\n{file_stats}")

    # Create a SparkSession object with the desired configuration options
    spark = (
        SparkSession.builder.appName("YelpDataEda")
        .master("local[*]")
        .config("spark.driver.memory", "2g")
        .config("spark.executor.memory", "2g")
        .config("spark.executor.cores", "2")
        .config("spark.driver.maxResultSize", "1g")
        .getOrCreate()
    )

    # ### Wrangling Businesses Data (yelp_academic_dataset_business.json)
    business_schema = StructType(
        [
            StructField("business_id", StringType(), True),
            StructField("name", StringType(), True),
            StructField("address", StringType(), True),
            StructField("city", StringType(), True),
            StructField("state", StringType(), True),
            StructField("postal_code", StringType(), True),
            StructField("latitude", StringType(), True),
            StructField("longitude", StringType(), True),
            StructField("stars", StringType(), True),
            StructField("review_count", IntegerType(), True),
            StructField("is_open", IntegerType(), True),
            StructField("categories", StringType(), True),
            StructField(
                "attributes",
                StructType(
                    [
                        StructField("Alcohol", StringType(), True),
                        StructField(
                            "Ambience",
                            StructType(
                                [
                                    StructField("casual", StringType(), True),
                                    StructField("classy", StringType(), True),
                                    StructField("divey", StringType(), True),
                                    StructField("hipster", StringType(), True),
                                    StructField("intimate", StringType(), True),
                                    StructField("romantic", StringType(), True),
                                    StructField("touristy", StringType(), True),
                                    StructField("trendy", StringType(), True),
                                    StructField("upscale", StringType(), True),
                                ]
                            ),
                            True,
                        ),
                        StructField("BikeParking", StringType(), True),
                        StructField("BusinessAcceptsCreditCards", StringType(), True),
                        StructField(
                            "BusinessParking",
                            StructType(
                                [
                                    StructField("garage", StringType(), True),
                                    StructField("lot", StringType(), True),
                                    StructField("street", StringType(), True),
                                    StructField("valet", StringType(), True),
                                ]
                            ),
                            True,
                        ),
                        StructField("GoodForKids", StringType(), True),
                        StructField("HasTV", StringType(), True),
                        StructField("NoiseLevel", StringType(), True),
                        StructField("OutdoorSeating", StringType(), True),
                        StructField("RestaurantsAttire", StringType(), True),
                        StructField("RestaurantsDelivery", StringType(), True),
                        StructField("RestaurantsGoodForGroups", StringType(), True),
                        StructField("RestaurantsPriceRange2", StringType(), True),
                        StructField("RestaurantsReservations", StringType(), True),
                        StructField("RestaurantsTakeOut", StringType(), True),
                        StructField("WiFi", StringType(), True),
                    ]
                ),
                True,
            ),
            StructField(
                "hours",
                StructType(
                    [
                        StructField("Monday", StringType(), True),
                        StructField("Tuesday", StringType(), True),
                        StructField("Wednesday", StringType(), True),
                        StructField("Thursday", StringType(), True),
                        StructField("Friday", StringType(), True),
                        StructField("Saturday", StringType(), True),
                        StructField("Sunday", StringType(), True),
                    ]
                ),
                True,
            ),
        ]
    )

    business_df = spark.read.json(
        str(RAW_DATA_DIR / "yelp_academic_dataset_business.json"), schema=business_schema
    )

    # Extract attributes
    business_df = (
        business_df.withColumn("attr_alcohol", business_df.attributes.Alcohol)
        .withColumn("attr_bike_parking", business_df.attributes.BikeParking)
        .withColumn(
            "attr_business_accepts_credit_cards", business_df.attributes.BusinessAcceptsCreditCards
        )
        .withColumn("attr_good_for_kids", business_df.attributes.GoodforKids)
        .withColumn("attr_has_tv", business_df.attributes.HasTV)
        .withColumn("attr_noise_level", business_df.attributes.NoiseLevel)
        .withColumn("attr_outdoor_seating", business_df.attributes.OutdoorSeating)
        .withColumn("attr_restaurants_attire", business_df.attributes.RestaurantsAttire)
        .withColumn("attr_restaurants_delivery", business_df.attributes.RestaurantsDelivery)
        .withColumn(
            "attr_restaurants_good_for_groups", business_df.attributes.RestaurantsGoodforGroups
        )
        .withColumn("attr_restaurants_price_range2", business_df.attributes.RestaurantsPriceRange2)
        .withColumn(
            "attr_restaurants_reservations", business_df.attributes.RestaurantsReservations
        )
        .withColumn("attr_restaurants_takeout", business_df.attributes.RestaurantsTakeOut)
        .withColumn("attr_wifi", business_df.attributes.WiFi)
        .withColumn("attr_ambience_casual", business_df.attributes.Ambience.casual)
        .withColumn("attr_ambience_classy", business_df.attributes.Ambience.classy)
        .withColumn("attr_ambience_divey", business_df.attributes.Ambience.divey)
        .withColumn("attr_ambience_hipster", business_df.attributes.Ambience.hipster)
        .withColumn("attr_ambience_intimate", business_df.attributes.Ambience.intimate)
        .withColumn("attr_ambience_romantic", business_df.attributes.Ambience.romantic)
        .withColumn("attr_ambience_touristy", business_df.attributes.Ambience.touristy)
        .withColumn("attr_ambience_trendy", business_df.attributes.Ambience.trendy)
        .withColumn("attr_ambience_upscale", business_df.attributes.Ambience.upscale)
        .withColumn("attr_business_parking_garage", business_df.attributes.BusinessParking.garage)
        .withColumn("attr_business_parking_lot", business_df.attributes.BusinessParking.lot)
        .withColumn("attr_business_parking_street", business_df.attributes.BusinessParking.street)
        .withColumn("attr_business_parking_valet", business_df.attributes.BusinessParking.valet)
    )

    # Extract hours
    business_df = (
        business_df.withColumn("hours_monday", business_df.hours.Monday)
        .withColumn("hours_tuesday", business_df.hours.Tuesday)
        .withColumn("hours_wednesday", business_df.hours.Wednesday)
        .withColumn("hours_thursday", business_df.hours.Thursday)
        .withColumn("hours_friday", business_df.hours.Friday)
        .withColumn("hours_saturday", business_df.hours.Saturday)
        .withColumn("hours_sunday", business_df.hours.Sunday)
    )

    # Drop original nested columns
    business_df = business_df.drop("attributes", "hours")

    # Long-form to Wide-form on categories column
    business_df.select("categories").show(5, False)  # Current categories column

    business_df = business_df.withColumn("category", split("categories", ", "))

    # Explode categories column
    exploded_categories = business_df.select("business_id", "category", "categories").withColumn(
        "category", explode("category")
    )
    exploded_categories = exploded_categories.withColumn(
        "category", regexp_replace("category", "&", " ")
    )
    exploded_categories = exploded_categories.withColumn(
        "category", regexp_replace("category", " +", " ")
    )
    exploded_categories = exploded_categories.withColumn(
        "category", regexp_replace("category", " ", "_")
    )
    exploded_categories = exploded_categories.withColumn(
        "category", regexp_replace("category", "-", "_")
    )
    exploded_categories = exploded_categories.withColumn(
        "category", regexp_replace("category", "_+", "_")
    )
    exploded_categories = exploded_categories.withColumn(
        "category", regexp_replace("category", "'", "")
    )
    exploded_categories = exploded_categories.withColumn(
        "category", regexp_replace("category", "&", "and")
    )
    exploded_categories = exploded_categories.withColumn(
        "category", regexp_replace("category", "/", "_or_")
    )
    exploded_categories = exploded_categories.withColumn(
        "category", regexp_replace("category", "\(", "")
    )
    exploded_categories = exploded_categories.withColumn(
        "category", regexp_replace("category", "\)", "")
    )
    exploded_categories = exploded_categories.withColumn("category", trim("category"))
    exploded_categories = exploded_categories.withColumn("category", lower("category"))
    exploded_categories = exploded_categories.withColumn(
        "category", concat(lit("cat_"), exploded_categories.category)
    )

    exploded_categories.select("business_id", "category", "categories").show(
        10, False
    )  # Exploded categories column

    # Pivot the resulting rows into columns
    pivoted_df = exploded_categories.groupBy("business_id").pivot("category").count().na.fill(0)
    # Join the pivoted DataFrame back to the original DataFrame
    joined_df = business_df.join(pivoted_df, "business_id", "left").drop("categories", "category")

    business_pdf = joined_df.toPandas()

    # Checks to see if rows count match
    (business_df.count(), business_pdf.shape[0])

    business_pdf.to_csv(str(CLEANED_DATA_DIR / "business.csv"), index=False)

    # Free up memory
    del business_pdf
    del business_df
    del joined_df
    del exploded_categories
    del pivoted_df

    # Read the review data from JSON file chunk by chunk
    review_df = pd.read_json(
        str(RAW_DATA_DIR / "yelp_academic_dataset_review.json"), lines=True, chunksize=1_000_000
    )

    # Save the review data to CSV file chunk by chunk
    for i, chunk in enumerate(review_df):
        print(f"Processing chunk {i+1}")

        if i == 0:
            chunk.to_csv(str(CLEANED_DATA_DIR / "review.csv"), mode="w", header=True, index=False)
        else:
            chunk.to_csv(str(CLEANED_DATA_DIR / "review.csv"), mode="a", header=False, index=False)

    del review_df
    del chunk

    # Read the user data from JSON file chunk by chunk
    user_df = pd.read_json(
        str(RAW_DATA_DIR / "yelp_academic_dataset_user.json"), lines=True, chunksize=1_000_000
    )

    # Save the user data to CSV file chunk by chunk
    for i, chunk in enumerate(user_df):
        print(f"Processing chunk {i+1}")

        if i == 0:
            chunk.to_csv(str(CLEANED_DATA_DIR / "user.csv"), mode="w", header=True, index=False)
        else:
            chunk.to_csv(str(CLEANED_DATA_DIR / "user.csv"), mode="a", header=False, index=False)

    del user_df
    del chunk


if __name__ == "__main__":
    logr.info("Starting data wrangling script")

    main()
