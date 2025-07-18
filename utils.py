import datetime
import os

import numpy as np
import pandas
import torch

from model.weather_dataset import WeatherDataset

MIN_TEMP = -25
MAX_TEMP = 50

MAX_PREC = 30

MAX_WIND = 75

MIN_LAT = 35
MAX_LAT = 50

MIN_LON = 5
MAX_LON = 20

MIN_ALT = 0
MAX_ALT = 1000

# w_codes_csv = pandas.read_csv("data/wmo_weather_codes.csv")

geo_csv = pandas.read_csv("data/capoluoghi_regione_italiani.csv")


def normalize_and_separate(geo, path):
    data_csv = pandas.read_csv(path)
    data_csv.pop("Unnamed: 0")

    date = data_csv["date"]
    y_list = []
    m_list = []
    d_list = []
    h_list = []

    for d in date:
        year = d.split("-")[0]
        y_list.append(year)

        month = d.split("-")[1]
        m_list.append(month)

        day = d.split("-")[2].split(" ")[0]
        d_list.append(day)

        hour = d.split("-")[2].split(" ")[1].split(":")[0]
        h_list.append(hour)

    data_csv.pop("date")
    data_csv.insert(0, "year", y_list)
    data_csv.insert(1, "month", m_list)
    data_csv.insert(2, "day", d_list)
    data_csv.insert(3, "hour", h_list)

    # normalizing
    data_list = []
    for index, row in data_csv.iterrows():
        feat_list = []

        # year
        feat_list.append((int(row["year"]) - 2000) / 25)

        # cyclic day (includes both month and day)
        day_of_year = datetime.datetime(int(row["year"]), int(row["month"]), int(row["day"]))
        day_of_year = day_of_year.timetuple().tm_yday
        feat_list.append(np.sin(2 * np.pi * day_of_year) / 12)
        feat_list.append(np.cos(2 * np.pi * day_of_year) / 12)

        # cyclic hour
        feat_list.append(np.sin(2 * np.pi * int(row["hour"])) / 12)
        feat_list.append(np.cos(2 * np.pi * int(row["hour"])) / 12)

        # weather normalization
        if MAX_TEMP >= float(row["temperature_2m"]) >= MIN_TEMP:
            temp = float(np.log1p(float(row["temperature_2m"])-MIN_TEMP))
            feat_list.append(temp / (np.log1p(MAX_TEMP - MIN_TEMP)))
        else:
            raise Exception(f'ERROR - TEMPERATURE OUT OF RANGE: {float(row["temperature_2m"])}')

        feat_list.append(float(row["relative_humidity_2m"]) / 100.0)

        if MAX_PREC >= float(row["precipitation"]) >= 0:
            prec = float(np.log1p(float(row["precipitation"])))
            feat_list.append(prec / np.log1p(MAX_PREC))
        else:
            raise Exception(f'ERROR - PRECIPITATIONS OUT OF RANGE: {float(row["precipitation"])}')

        # w_row = (w_codes_csv.loc[w_codes_csv["Code"] == row["weather_code"]]).index[0]
        # feat_list.append(w_row)

        feat_list.append(float(row["cloud_cover"]) / 100.0)

        if MAX_WIND >= float(row["wind_speed_10m"]) >= 0:
            wind = float(np.log1p(float(row["wind_speed_10m"])))
            feat_list.append(wind / np.log1p(MAX_WIND))
        else:
            raise Exception(f'ERROR - WIND OUT OF RANGE: {float(row["wind_speed_10m"])}')

        data_list.append(feat_list)

    if MAX_LAT >= float(geo["Latitude"]) >= MIN_LAT:
        geo["Latitude"] = (float(geo["Latitude"]) - MIN_LAT) / (MAX_LAT - MIN_LAT)
    else:
        raise Exception(f'ERROR - LATITUDE OUT OF RANGE: {float(geo["Latitude"])}')

    if MAX_LON >= float(geo["Longitude"]) >= MIN_LON:
        geo["Longitude"] = (float(geo["Longitude"]) - MIN_LON) / (MAX_LON - MIN_LON)
    else:
        raise Exception(f'ERROR - LONGITUDE OUT OF RANGE: {float(geo["Longitude"])}')

    if MAX_ALT >= float(geo["Altitude"]) >= MIN_ALT:
        geo["Altitude"] = (float(geo["Altitude"]) - MIN_ALT) / (MAX_ALT - MIN_ALT)
    else:
        raise Exception(f'ERROR - ALTITUDE OUT OF RANGE: {float(geo["Altitude"])}')

    w_dataset = WeatherDataset(geo=geo, raw_data=data_list)

    return w_dataset


def load_all(dates):
    dataset_list = []
    for p in os.listdir("./data"):
        if os.path.isdir("./data/" + p):
            g_row = geo_csv.loc[geo_csv["City"] == p]
            print(f"Loading {p} data...")
            for end_p in dates:
                geo = {
                    "Latitude": g_row.get("Latitude").item(),
                    "Longitude": g_row["Longitude"].item(),
                    "Altitude": g_row["Altitude"].item()
                }
                ds = normalize_and_separate(geo=geo, path=f"./data/{p}/{p}_Weather{end_p}.csv")
                dataset_list.append(ds)

    weather_dataset = torch.utils.data.ConcatDataset([c for c in dataset_list])
    return weather_dataset


def denormalize(t):
    denormalized = dict()

    # temperature
    denormalized.update({"TEMP": float(np.expm1(t[0].item() * np.log1p(MAX_TEMP-MIN_TEMP)) + MIN_TEMP)})

    # humidity
    denormalized.update({"HUM": float(t[1].item() * 100)})

    # precipitation
    denormalized.update({"PREC": float(np.expm1(t[2].item() * np.log1p(MAX_PREC)))})

    # cloud cover
    denormalized.update({"CLOUD": float(t[3].item() * 100)})

    # wind speed
    denormalized.update({"WIND": float(np.expm1(t[4].item() * np.log1p(MAX_WIND)))})

    return denormalized
