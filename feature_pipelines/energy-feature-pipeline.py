import time
from os.path import exists

import hopsworks
import pandas as pd
from entsoe import EntsoePandasClient
from pandera import Check, Column, DataFrameSchema
import datetime

from utils import entsoe_key


energy_data = pd.DataFrame()


if not exists("../data/energy.csv"):
    client = EntsoePandasClient(api_key=entsoe_key)

    start_date = pd.Timestamp("20201212", tz="Europe/Berlin")
    end_date = pd.Timestamp("20221219", tz="Europe/Berlin")
    country_code = "SE_3"

    batch_end_date = start_date + pd.Timedelta(days=100)

    while start_date < end_date:
        print(f"{start_date} - {batch_end_date}")

        day_ahead_prices = client.query_day_ahead_prices(
            country_code, start=start_date, end=batch_end_date
        )
        load = client.query_load(country_code, start=start_date, end=batch_end_date)
        aggregate_water_reservoirs_and_hydro_storage = (
            client.query_aggregate_water_reservoirs_and_hydro_storage(
                country_code, start=start_date, end=batch_end_date
            )
        )

        result = pd.concat(
            [day_ahead_prices, load, aggregate_water_reservoirs_and_hydro_storage],
            axis=1,
        )
        # forward fill to fill in missing values
        result = result.ffill()
        # concat the result to the energy_data dataframe
        energy_data = pd.concat([energy_data, result], axis=0)

        start_date = batch_end_date + pd.Timedelta(days=1)
        batch_end_date = start_date + pd.Timedelta(days=100)
        if batch_end_date > end_date:
            batch_end_date = end_date

        time.sleep(61)

    energy_data.to_csv("../data/energy.csv")
else:
    energy_data = pd.read_csv("../data/energy.csv")

# drop the rows with missing values
energy_data = energy_data.dropna(axis=0)
energy_data.columns = ["date", "price", "load", "filling_rate"]
energy_data["date"] = pd.to_datetime(energy_data["date"])
# set the date column as the index
energy_data = energy_data.set_index("date")
# resample the data to daily mean
energy_data = energy_data.resample("D").mean()
# reset the index
energy_data = energy_data.reset_index()
energy_data["date"] = energy_data["date"].dt.strftime("%Y-%m-%d")

# shift the price column to create the lag features
energy_data["p_1"] = energy_data["price"].shift()
energy_data["p_2"] = energy_data["price"].shift(2)
energy_data["p_3"] = energy_data["price"].shift(3)
energy_data["p_4"] = energy_data["price"].shift(4)
energy_data["p_5"] = energy_data["price"].shift(5)
energy_data["p_6"] = energy_data["price"].shift(6)
energy_data["p_7"] = energy_data["price"].shift(7)

energy_data = energy_data.dropna()
# reset the index
energy_data = energy_data.reset_index(drop=True)
# drop the first row since it is the first day of the dataset
energy_data = energy_data.drop(index=0)
# reset the index
energy_data = energy_data.reset_index(drop=True)

schema = DataFrameSchema(
        {
            "date": Column(
                checks=[
                    Check(
                        lambda d: bool(datetime.datetime.strptime(d, "%Y-%m-%d")),
                        element_wise=True,
                    ),
                ]
            ),
            "price": Column(float),
            "load": Column(float, Check.greater_than_or_equal_to(0)),
            "filling_rate": Column(float, Check.greater_than_or_equal_to(0)),
            "p_1": Column(float),
            "p_2": Column(float),
            "p_3": Column(float),
            "p_4": Column(float),
            "p_5": Column(float),
            "p_6": Column(float),
            "p_7": Column(float),
        }
    )
schema.to_json('../pandera_schemas/energy-feature-pipeline-daily-schema.json')

energy_data = schema.validate(energy_data)

project = hopsworks.login()
fs = project.get_feature_store()

energy_fg = fs.get_or_create_feature_group(
    name="energy_prices",
    version=1,
    primary_key=[
        "date",
        "load",
        "filling_rate",
        "p_1",
        "p_2",
        "p_3",
        "p_4",
        "p_5",
        "p_6",
        "p_7",
    ],
    description="daily energy prices",
)
energy_fg.insert(energy_data, write_options={"wait_for_job": False})
