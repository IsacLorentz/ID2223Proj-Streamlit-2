import datetime

import hopsworks
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
# from forex_python.converter import CurrencyRates
from utils import eur_sek_convert

import streamlit as st

st.markdown(
    """**Logging of predictions of daily average energy prices in Stockholm/SE3 for the upcoming 7 days**"""
)
progressBar = st.progress(0)
progressBar.progress(20)
# c = CurrencyRates()

project = hopsworks.login()
fs = project.get_feature_store()

price_pred_fg = fs.get_feature_group(name="price_predictions", version=1).read()
# print(price_pred_fg)
pric_actual_fg = fs.get_feature_group(name="price_data", version=1).read()
# print(pric_actual_fg)
price_combined = pric_actual_fg.merge(price_pred_fg, on="date")
price_combined["predicted_price"] = (
    eur_sek_convert(price_combined["predicted_price"]))
price_combined["entsoe_avg"] = (
    eur_sek_convert(price_combined["entsoe_avg"])
    )
progressBar.progress(60)
# print(price_combined)

total_mae_entsoe = (
    sum(abs(price_combined["entsoe_avg"] - price_combined["predicted_price"]))
    / price_combined.shape[0]
)

total_mae_elbruk = (
    sum(abs(price_combined["elbruk_dagspris"] - price_combined["predicted_price"]))
    / price_combined.shape[0]
)

days_ahead_mae_entsoe = [
    price_combined.query(f"days_ahead == {i}") for i in range(1, 8)
]
days_ahead_mae_entsoe = [df for df in days_ahead_mae_entsoe if not df.empty]
days_ahead_mae_entsoe = [
    sum(abs(df["entsoe_avg"] - df["predicted_price"])) / df.shape[0]
    for df in days_ahead_mae_entsoe
]

days_ahead_mae_elbruk = [
    price_combined.query(f"days_ahead == {i}") for i in range(1, 8)
]
days_ahead_mae_elbruk = [df for df in days_ahead_mae_elbruk if not df.empty]
days_ahead_mae_elbruk = [
    sum(abs(df["elbruk_dagspris"] - df["predicted_price"])) / df.shape[0]
    for df in days_ahead_mae_elbruk
]

progressBar.progress(70)
columns = ["total MAE"] + [f"d + {i} MAE" for i in range(1, 8)]
entsoe_mae_data = [total_mae_entsoe] + [mae for mae in days_ahead_mae_entsoe]
entsoe_mae_data += ["NaN"] * (8 - len(entsoe_mae_data))
entsoe_mae_data = [entsoe_mae_data]
# print(entsoe_mae_data)
# print(columns)
entsoe_mae = pd.DataFrame(data=entsoe_mae_data, columns=columns)
# print(entsoe_mae)
# mae_days_ahead = []

elbruk_mae_data = [total_mae_elbruk] + [mae for mae in days_ahead_mae_elbruk]
elbruk_mae_data += ["NaN"] * (8 - len(elbruk_mae_data))
elbruk_mae_data = [elbruk_mae_data]
# print(elbruk_mae_data)
# print(columns)
elbruk_mae = pd.DataFrame(data=elbruk_mae_data, columns=columns)

st.markdown("""**ENTSOE daily average price MAE (SEK ÖRE)**""")
st.dataframe(data=entsoe_mae.style.background_gradient(axis="columns", cmap="YlOrRd"))
st.markdown("""**elbruk.se dagspris MAE (SEK ÖRE)**""")
st.dataframe(data=elbruk_mae.style.background_gradient(axis="columns", cmap="YlOrRd"))
progressBar.progress(80)

# print(price_combined)
# print(price_combined.groupby('days_ahead').max())
# #print latest maes
# # print(price_combined)

# print(price_combined[price_combined['date'] == price_combined['date'].max()]['days_ahead'].min())

# it = price_combined[price_combined['date'] == price_combined['date'].max()]['days_ahead'].min()
# print(it)
# print(type(price_combined[price_combined['date'] == price_combined['date'].max()]['date'].values[0]))
latest_day = price_combined[price_combined["date"] == price_combined["date"].max()]
# print(latest_day[latest_day['days_ahead'] == latest_day['days_ahead'].min()])
latest_day_latest_pred = latest_day[
    latest_day["days_ahead"] == latest_day["days_ahead"].min()
]

# dates, predicted_price, entsoe, entsoe_error, elbruk, elbruk_error, days_ahead = [latest_day_latest_pred['date'].values[0]], [latest_day_latest_pred['predicted_price'].values[0]], [latest_day_latest_pred['entsoe_avg'].values[0]], [latest_day_latest_pred['predicted_price'].values[0] - latest_day_latest_pred['entsoe_avg'].values[0]], [latest_day_latest_pred['elbruk_dagspris'].values[0]], [latest_day_latest_pred['predicted_price'].values[0] - latest_day_latest_pred['elbruk_dagspris'].values[0]],[latest_day_latest_pred['days_ahead'].values[0]]
dates = [latest_day_latest_pred["date"].values[0]]
predicted_price = [latest_day_latest_pred["predicted_price"].values[0]]
entsoe = [latest_day_latest_pred["entsoe_avg"].values[0]]
entsoe_error = [
    latest_day_latest_pred["predicted_price"].values[0]
    - latest_day_latest_pred["entsoe_avg"].values[0]
]
elbruk = [latest_day_latest_pred["elbruk_dagspris"].values[0]]
elbruk_error = [
    latest_day_latest_pred["predicted_price"].values[0]
    - latest_day_latest_pred["elbruk_dagspris"].values[0]
]
days_ahead = [latest_day_latest_pred["days_ahead"].values[0]]

# print(dates[0])
last_day = datetime.datetime.strptime(dates[0], "%Y-%m-%d")
# print(last_day)
# print(price_combined)
for i in range(1, 10):
    prev_day = last_day - datetime.timedelta(days=i)
    if (price_combined["date"] == prev_day.strftime("%Y-%m-%d")).any():
        # print('succ')
        # print(prev_day.strftime("%Y-%m-%d"))
        prev_day_df = price_combined[
            price_combined["date"] == prev_day.strftime("%Y-%m-%d")
        ].sort_values(by=["days_ahead"])
        prev_day_df_latest_pred = prev_day_df[
            prev_day_df["days_ahead"] == prev_day_df["days_ahead"].min()
        ]
        # print(prev_day_df)
        for i, row in prev_day_df.iterrows():
            dates.append(row["date"])
            predicted_price.append(row["predicted_price"])
            entsoe.append(row["entsoe_avg"])
            entsoe_error.append(row["predicted_price"] - row["entsoe_avg"])
            elbruk.append(row["elbruk_dagspris"])
            elbruk_error.append(row["predicted_price"] - row["elbruk_dagspris"])
            days_ahead.append(row["days_ahead"])
        # dates.append(prev_day_df_latest_pred["date"].values[0])
        # predicted_price.append(prev_day_df_latest_pred["predicted_price"].values[0])
        # entsoe.append(prev_day_df_latest_pred["entsoe_avg"].values[0])
        # entsoe_error.append(
        #     prev_day_df_latest_pred["predicted_price"].values[0]
        #     - prev_day_df_latest_pred["entsoe_avg"].values[0]
        # )
        # elbruk.append(prev_day_df_latest_pred["elbruk_dagspris"].values[0])
        # elbruk_error.append(
        #     prev_day_df_latest_pred["predicted_price"].values[0]
        #     - prev_day_df_latest_pred["elbruk_dagspris"].values[0]
        # )
        # days_ahead.append(prev_day_df_latest_pred["days_ahead"].values[0])


data = [dates, days_ahead, predicted_price, entsoe, entsoe_error, elbruk, elbruk_error]
data = map(list, zip(*data))
progressBar.progress(90)
# print(data)
last_n_results = pd.DataFrame(
    data=data,
    columns=[
        "date",
        "days ahead",
        "predicted price",
        "entsoe price",
        "entsoe difference",
        "elbruk price",
        "elbruk difference",
    ],
)
# print(last_n_results)
st.markdown("""**Detailed logging for latest 10 days**""")
st.dataframe(
    data=last_n_results.style.background_gradient(
        subset=["entsoe difference", "elbruk difference"], cmap="YlOrRd"
    )
)
progressBar.progress(100)
