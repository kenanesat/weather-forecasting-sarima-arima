# 🌤️ Time Series Weather Forecasting System (ARIMA & SARIMA)

This repository contains the graduation project developed for the **CPR 204** course at **Çankaya University (Computer Programming Department)**.

---

## 📌 Project Overview
The system fetches multi-year historical weather data via REST APIs, analyzes weather patterns (temperature, humidity, wind, rain), compares the performance of **SARIMA** and **ARIMA** time-series forecasting models, and generates high-accuracy weather predictions.

---

> 📌 **Note on Datasets & File Size Limits:**  
> The full historical weather dataset (`temperature.csv`, `rain.csv`, `wind.csv`, and `humidity.csv` exceeding 190–240 MB each) as well as the 2025 feature datasets (`temperature_2025.csv`, `wind_2025.csv`, `rain_2025.csv`, and `humidity_2025.csv` around 20 MB each) were successfully fetched via Open-Meteo REST API and used during model training and evaluation.  
> 
> Due to GitHub's **25 MB per-file upload limit via the web interface**, these raw full-history and large feature CSV files are omitted from this repository. Instead, the final combined actual dataset (`weather_2025_actual_6hour_combined.csv`), model forecasts (`final_forecast_1year_6hour_sarima_adf_min_d1_full_history_all.csv`), and all Python source scripts are provided for full reproduction and code review.

---

## 🛠️ Technologies & Tools Used
* **Language:** Python
* **Data Extraction:** Open-Meteo REST API
* **Data Processing & Analysis:** Pandas, NumPy
* **Time Series Modeling:** Statsmodels (ARIMA & SARIMA)

---

## 📂 Repository Content
* `download_2025_weather_actual.py`: Fetches actual weather data for 2025 using the Open-Meteo REST API.
* `sarima_hourly_adf_d1.py`: Time series modeling, stationarity tests (ADF), and forecasting logic.
* `download_*.py`: Modular scripts to extract specific weather metrics (humidity, rain, temperature, wind).
* `weather_2025_actual_6hour_combined.csv`: Processed 6-hour interval combined actual dataset.
* `final_forecast_1year_6hour_sarima_adf_min_d1_full_history_all.csv`: Generated 1-year SARIMA forecast outputs.

---

## 📑 Author & Credits
* **Developer:** Kenan Esat Şen
* **University:** Çankaya University (Computer Programming Department)
