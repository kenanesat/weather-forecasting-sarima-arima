import time
from pathlib import Path

import pandas as pd
import requests

BASE_DIR = Path(__file__).resolve().parent

START_DATE = "2025-01-01"
END_DATE = "2025-12-31"
SLEEP_SECONDS = 3
TIMEOUT_SECONDS = 60

OUTPUT_TEMPERATURE = BASE_DIR / "temperature_2025.csv"
OUTPUT_WIND = BASE_DIR / "wind_2025.csv"
OUTPUT_HUMIDITY = BASE_DIR / "humidity_2025.csv"
OUTPUT_RAIN = BASE_DIR / "rain_2025.csv"

cities = {
    "Adana": (37.00, 35.32),
    "Adiyaman": (37.76, 38.27),
    "Afyonkarahisar": (38.76, 30.54),
    "Agri": (39.72, 43.05),
    "Amasya": (40.65, 35.83),
    "Ankara": (39.93, 32.85),
    "Antalya": (36.89, 30.70),
    "Artvin": (41.18, 41.82),
    "Aydin": (37.84, 27.84),
    "Balikesir": (39.65, 27.88),
    "Bilecik": (40.15, 29.98),
    "Bingol": (38.89, 40.49),
    "Bitlis": (38.40, 42.11),
    "Bolu": (40.73, 31.61),
    "Burdur": (37.72, 30.29),
    "Bursa": (40.19, 29.06),
    "Canakkale": (40.15, 26.41),
    "Cankiri": (40.60, 33.61),
    "Corum": (40.55, 34.95),
    "Denizli": (37.77, 29.08),
    "Diyarbakir": (37.91, 40.23),
    "Edirne": (41.67, 26.56),
    "Elazig": (38.67, 39.22),
    "Erzincan": (39.75, 39.49),
    "Erzurum": (39.90, 41.27),
    "Eskisehir": (39.78, 30.52),
    "Gaziantep": (37.06, 37.38),
    "Giresun": (40.91, 38.39),
    "Gumushane": (40.46, 39.48),
    "Hakkari": (37.57, 43.74),
    "Hatay": (36.20, 36.16),
    "Isparta": (37.76, 30.55),
    "Mersin": (36.80, 34.63),
    "Istanbul": (41.01, 28.97),
    "Izmir": (38.42, 27.14),
    "Kars": (40.60, 43.09),
    "Kastamonu": (41.39, 33.78),
    "Kayseri": (38.72, 35.48),
    "Kirklareli": (41.73, 27.22),
    "Kirsehir": (39.14, 34.17),
    "Kocaeli": (40.77, 29.92),
    "Konya": (37.87, 32.48),
    "Kutahya": (39.42, 29.99),
    "Malatya": (38.35, 38.31),
    "Manisa": (38.62, 27.43),
    "Kahramanmaras": (37.57, 36.93),
    "Mardin": (37.32, 40.72),
    "Mugla": (37.21, 28.36),
    "Mus": (38.74, 41.49),
    "Nevsehir": (38.62, 34.72),
    "Nigde": (37.97, 34.68),
    "Ordu": (40.98, 37.88),
    "Rize": (41.02, 40.52),
    "Sakarya": (40.78, 30.40),
    "Samsun": (41.29, 36.33),
    "Siirt": (37.93, 41.94),
    "Sinop": (42.03, 35.15),
    "Sivas": (39.75, 37.01),
    "Tekirdag": (40.98, 27.51),
    "Tokat": (40.32, 36.55),
    "Trabzon": (41.00, 39.72),
    "Tunceli": (39.11, 39.55),
    "Sanliurfa": (37.16, 38.79),
    "Usak": (38.68, 29.41),
    "Van": (38.50, 43.37),
    "Yozgat": (39.82, 34.81),
    "Zonguldak": (41.46, 31.80),
    "Aksaray": (38.37, 34.03),
    "Bayburt": (40.26, 40.22),
    "Karaman": (37.18, 33.22),
    "Kirikkale": (39.84, 33.52),
    "Batman": (37.88, 41.14),
    "Sirnak": (37.52, 42.45),
    "Bartin": (41.63, 32.34),
    "Ardahan": (41.11, 42.70),
    "Igdir": (39.92, 44.04),
    "Yalova": (40.65, 29.27),
    "Karabuk": (41.20, 32.63),
    "Kilis": (36.72, 37.12),
    "Osmaniye": (37.07, 36.25),
    "Duzce": (40.84, 31.16),
}


def fetch_city_weather(city: str, latitude: float, longitude: float) -> pd.DataFrame | None:
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "hourly": "temperature_2m,wind_speed_10m,relative_humidity_2m,precipitation",
        "timezone": "auto",
    }

    try:
        response = requests.get(url, params=params, timeout=TIMEOUT_SECONDS)
        response.raise_for_status()
        data = response.json()
    except Exception as error:
        print(f"Error: {city} -> request failed: {error}")
        return None

    if "hourly" not in data:
        print(f"Error: {city} -> hourly data not found")
        return None

    hourly = data["hourly"]
    required_keys = [
        "time",
        "temperature_2m",
        "wind_speed_10m",
        "relative_humidity_2m",
        "precipitation",
    ]

    missing_keys = [key for key in required_keys if key not in hourly]
    if missing_keys:
        print(f"Error: {city} -> missing keys: {missing_keys}")
        return None

    row_count = len(hourly["time"])
    if row_count == 0:
        print(f"Error: {city} -> empty hourly data")
        return None

    return pd.DataFrame({
        "datetime": hourly["time"],
        "city": city,
        "temperature": hourly["temperature_2m"],
        "wind": hourly["wind_speed_10m"],
        "humidity": hourly["relative_humidity_2m"],
        "rain": hourly["precipitation"],
    })


def main():
    all_data = []

    for city, (latitude, longitude) in cities.items():
        print("Downloading:", city)

        city_df = fetch_city_weather(city, latitude, longitude)
        if city_df is not None:
            all_data.append(city_df)
            print(f"OK: {city} -> {len(city_df)} rows")

        time.sleep(SLEEP_SECONDS)

    if not all_data:
        print("No data downloaded.")
        return

    weather = pd.concat(all_data, ignore_index=True)

    temperature = weather[["datetime", "city", "temperature"]].copy()
    wind = weather[["datetime", "city", "wind"]].copy()
    humidity = weather[["datetime", "city", "humidity"]].copy()
    rain = weather[["datetime", "city", "rain"]].copy()

    temperature.to_csv(OUTPUT_TEMPERATURE, index=False)
    wind.to_csv(OUTPUT_WIND, index=False)
    humidity.to_csv(OUTPUT_HUMIDITY, index=False)
    rain.to_csv(OUTPUT_RAIN, index=False)

    print("\nFinished!")
    print("temperature_2025.csv rows:", len(temperature))
    print("wind_2025.csv rows:", len(wind))
    print("humidity_2025.csv rows:", len(humidity))
    print("rain_2025.csv rows:", len(rain))
    print("City count:", weather["city"].nunique())


if __name__ == "__main__":
    main()
