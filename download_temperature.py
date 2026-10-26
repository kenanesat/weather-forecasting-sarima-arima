import requests
import pandas as pd
import time

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
    "Duzce": (40.84, 31.16)
}

all_data = []

for city, coords in cities.items():
    lat, lon = coords
    print("Downloading:", city)

    url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date=2015-01-01&end_date=2024-12-31&hourly=temperature_2m&timezone=auto"

    try:
        response = requests.get(url, timeout=60)
        data = response.json()

        if "hourly" not in data:
            print("Error:", city)
            continue

        df = pd.DataFrame({
            "datetime": data["hourly"]["time"],
            "city": city,
            "temperature": data["hourly"]["temperature_2m"]
        })

        all_data.append(df)

    except Exception as e:
        print("Error:", city, e)

    time.sleep(3)

temperature = pd.concat(all_data, ignore_index=True)
temperature.to_csv("temperature.csv", index=False)

print("Finished! temperature.csv created")
print("Rows:", len(temperature))
print(temperature.head())