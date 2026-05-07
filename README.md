# Time Series Prediction — Bombing Operations and Weather Analysis

Predicting mean temperature trends during World War II bombing 
operations using ARIMA time series modelling, combined with 
exploratory analysis of WWII aerial bombing mission data.

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Method](https://img.shields.io/badge/Method-ARIMA-orange)
![Task](https://img.shields.io/badge/Task-Time%20Series%20Forecasting-green)
![Domain](https://img.shields.io/badge/Domain-Historical%20Data%20Analysis-blue)

---

## Overview

Analyses World War II aerial bombing operations data alongside 
historical weather station records to predict mean temperature 
trends near active bombing zones. Focuses on the USA vs Burma 
(Katha city) conflict from 1942 to 1945, using the nearest 
weather station (Bindukuri) as the temperature source. Implements 
a full ARIMA time series pipeline including stationarity testing 
(Augmented Dickey-Fuller), moving average decomposition, 
differencing, ACF/PACF analysis, and ARIMA(1,0,1) forecasting.

---

## Problem Statement

Historical bombing operations were heavily influenced by weather 
conditions — temperature, visibility, and climate patterns 
determined when and where missions were feasible. This project 
uses ARIMA modelling to forecast mean temperatures in the 
Bindukuri area of Burma during the WWII period (1944–1945), 
demonstrating classical time series analysis techniques on a 
unique real-world historical dataset.

---

## Dataset

Three datasets are used in this project:

### 1. WWII Aerial Bombing Operations
- **Name:** WWII Aerial Bombing Operations Dataset
- **Source:** [Download from Kaggle](https://www.kaggle.com/datasets/usaf/world-war-ii)
- **File:** `operations.csv`
- **Key Columns used:** `Country`, `Target Country`, 
  `Target City`, `Mission Date`, `Aircraft Series`, 
  `Theater of Operations`, `Target Latitude`, 
  `Target Longitude`, `Takeoff Latitude`, `Takeoff Longitude`
- **Scope used:** USA bombing missions targeting Katha city, 
  Burma — filtered from 1943 onwards

### 2. Weather Station Locations
- **Name:** WWII Weather Station Locations
- **Source:** [Download from Kaggle](https://www.kaggle.com/datasets/usaf/world-war-ii)
- **File:** `Weather Station Locations.csv`
- **Key Columns used:** `WBAN`, `NAME`, 
  `STATE/COUNTRY ID`, `Latitude`, `Longitude`
- **Station used:** BINDUKURI (WBAN: 32907) — 
  the closest weather station to the Katha bombing zone

### 3. Weather Summary (Temperature Records)
- **Name:** WWII Summary of Weather
- **Source:** [Download from Kaggle](https://www.kaggle.com/datasets/usaf/world-war-ii)
- **File:** `Summary of Weather.csv`
- **Key Columns used:** `STA`, `Date`, `MeanTemp`
- **Scope used:** Daily mean temperature readings from 
  the Bindukuri station (STA: 32907)

> **Instructions:** Download all three CSV files from the 
> Kaggle link above and place them in the root folder of 
> this project before running the notebook.

---

## Approach

### Step 1 — Data Cleaning (Bombing Data)
- Dropped rows with NaN values in `Country`, 
  `Target Longitude`, and `Takeoff Longitude`
- Removed 28 unused feature columns (mission details, 
  weapon weights, aircraft counts) to keep only 
  geospatial and operational columns
- Filtered out erroneous coordinate values 
  (takeoff latitude `4248`, takeoff longitude `1355`)

### Step 2 — Exploratory Data Analysis
- Plotted bombing frequency by country using Seaborn 
  `countplot`
- Visualised top 10 target countries
- Plotted top 10 aircraft series using interactive 
  Plotly bar chart
- Analysed Theater of Operations distribution
- Focused analysis on USA → Burma (Katha) conflict

### Step 3 — Weather Time Series Construction
- Filtered Bindukuri weather station (WBAN: 32907)
- Parsed `Date` column to `datetime` using 
  `pd.to_datetime()`
- Plotted mean temperature trend over the full period
- Constructed time series with `Date` as the index

### Step 4 — Stationarity Testing
Applied two methods to check stationarity 
(required for ARIMA):

**Rolling Statistics:**
- Plotted 6-month rolling mean and rolling standard 
  deviation alongside the original series
- A stationary series has constant mean and variance 
  over time

**Augmented Dickey-Fuller (ADF) Test:**
- Used `statsmodels adfuller()` to formally test 
  for stationarity
- Compared test statistic against critical values 
  at 1%, 5%, and 10% significance levels

### Step 5 — Making the Series Stationary
Two transformation methods were applied and tested:

**Moving Average Subtraction:**
```python
moving_avg = ts.rolling(window=6).mean()
ts_moving_avg_diff = ts - moving_avg
```

**First-Order Differencing:**
```python
ts_diff = ts - ts.shift()
```
Both transformations were re-tested with ADF to 
confirm stationarity before fitting ARIMA.

### Step 6 — ACF and PACF Analysis
- Computed Autocorrelation Function (ACF) and Partial 
  Autocorrelation Function (PACF) up to 20 lags
- Plotted both with 95% confidence interval bounds 
  (`±1.96/√n`) to identify ARIMA p and q parameters
- ACF determines the MA order (q)
- PACF determines the AR order (p)

### Step 7 — ARIMA Modelling and Forecasting
```python
model = ARIMA(ts, order=(1, 0, 1))  # ARMA(1,1)
model_fit = model.fit()
```

| Parameter | Value | Meaning |
|-----------|-------|---------|
| p (AR) | 1 | 1 autoregressive lag |
| d (I) | 0 | No differencing (already stationary) |
| q (MA) | 1 | 1 moving average lag |

**Forecast period:** June 1944 — May 1945

- Visualised original vs predicted temperature on 
  the same line chart
- Computed Mean Squared Error (MSE) against the 
  full historical series

---

## Results

| Metric | Value |
|--------|-------|
| Model | ARIMA(1, 0, 1) |
| Forecast Period | June 1944 – May 1945 |
| Weather Station | Bindukuri, Burma (WBAN: 32907) |
| Target Conflict | USA bombing of Katha, Burma (1943–1945) |

> Run the notebook and replace `[add your value]` with 
> the MSE printed by the final `print("error:", error)` 
> cell.

---

## Visualisations

The notebook produces the following plots:

1. **Bombing frequency by country** — Seaborn countplot
2. **Top 10 target countries** — Seaborn countplot
3. **Top 10 aircraft series** — Interactive Plotly bar chart
4. **Theater of Operations distribution** — Seaborn countplot
5. **Bindukuri mean temperature over time** — line plot
6. **Rolling mean and rolling standard deviation** — 
   stationarity check
7. **Moving average decomposition** — original vs smoothed
8. **Differenced series** — stationarity transformation
9. **ACF and PACF plots** — side-by-side subplots with 
   confidence intervals
10. **ARIMA forecast** — original (red) vs predicted (blue) 
    temperature, saved as `graph.png`

---

## Technologies Used

- **Language:** Python 3
- **Time Series Modelling:** statsmodels 
  (ARIMA, adfuller, acf, pacf)
- **Data Handling:** Pandas, NumPy
- **Visualisation:** Matplotlib, Seaborn, 
  Plotly (chart_studio, plotly.graph_objs)
- **Evaluation:** Scikit-Learn (mean_squared_error)

---

## How to Run

```bash
# 1. Clone the repository
git clone https://github.com/OyelolaIbrahim/time-series-bombing-arima.git
cd time-series-bombing-arima

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download the three CSV files from Kaggle:
#    https://www.kaggle.com/datasets/usaf/world-war-ii
#    Place operations.csv, Weather Station Locations.csv,
#    and Summary of Weather.csv in the root folder

# 4. Run the notebook
jupyter notebook time_series_prediction.ipynb
```

