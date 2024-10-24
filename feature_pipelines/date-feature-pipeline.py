import datetime
import io
import sys
import urllib.request
from os.path import exists

import hopsworks
import pandas as pd
from pandera import Check, Column, DataFrameSchema



if not exists("../data/dates.csv"):
    # TODO: make this dynamic by calling get_dates_data() from utils.utils
    start_date = "2020-12-19"
    end_date = "2022-12-19"
    date_list = pd.date_range(start=start_date, end=end_date)
    dates_data = pd.DataFrame(data={"date": date_list})
    dates_data["dayofyear"] = dates_data["date"].dt.dayofyear
    dates_data["dayofweek"] = dates_data["date"].dt.dayofweek
    dates_data["month"] = dates_data["date"].dt.month
    dates_data["week"] = dates_data["date"].dt.isocalendar().week.astype("int64")
    dates_data.to_csv("data/dates.csv")
else:
    dates_data = pd.read_csv("../data/dates.csv", index_col=0)


dates_data = dates_data.dropna(axis=0)
dates_data["date"] = pd.to_datetime(dates_data["date"]) 
dates_data = dates_data.set_index("date")
dates_data = dates_data.reset_index()

dates_data["date"] = dates_data["date"].dt.strftime("%Y-%m-%d")

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
            "dayofyear": Column(int, Check.in_range(1, 366)),
            "dayofweek": Column(int, Check.in_range(0, 6)),
            "month": Column(int, Check.in_range(1, 12)),
            "week": Column(int, Check.in_range(0, 53)),
        }
    )
dates_data = schema.validate(dates_data)
schema.to_json('pandera_schemas/date-feature-pipeline-daily-schema.json')

project = hopsworks.login()
fs = project.get_feature_store()

date_fg = fs.get_or_create_feature_group(
    name="dates_data",
    version=1,
    primary_key=["date", "dayofyear", "dayofweek", "month", "week"],
    description="date features",
)
date_fg.insert(dates_data, write_options={"wait_for_job": False})
