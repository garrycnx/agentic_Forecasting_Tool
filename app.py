import streamlit as st
import pandas as pd
import matplotlib.dates as mdates
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import xlsxwriter
import warnings
warnings.filterwarnings("ignore")

# Load and clean data
def load_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'], dayfirst=True)
    df = df.dropna(subset=['timestamp', 'volume'])
    df['month_num'] = range(len(df))
    df['month'] = df['timestamp'].dt.month
    df['year'] = df['timestamp'].dt.year
    df['quarter'] = df['timestamp'].dt.quarter
    return df

# Generate future features dynamically
def generate_future_dataframe(df, n_future=24):
    last_month_num = df['month_num'].iloc[-1]
    future_months = range(last_month_num + 1, last_month_num + 1 + n_future)
    future_dates = pd.date_range(start=df['timestamp'].max() + pd.DateOffset(months=1), periods=n_future, freq='MS')
    future_df = pd.DataFrame({
        'month_num': future_months,
        'month': future_dates.month,
        'year': future_dates.year,
        'quarter': future_dates.quarter
    })
    return future_df


# Accuracy metrics
def get_metrics(y_true, y_pred):
    min_len = min(len(y_true), len(y_pred))
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]
    return {
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'R2': r2_score(y_true, y_pred),
        'MAPE': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    }

# Forecasting models
def forecast_arima(df):
    model = ARIMA(df['volume'], order=(1,1,1)).fit()
    forecast = model.forecast(steps=24)
    future_dates = pd.date_range(start=df['timestamp'].max() + pd.DateOffset(months=1), periods=24, freq='MS')
    forecast_df = pd.DataFrame({'ds': future_dates, 'yhat': forecast})
    y_pred = model.predict(start=0, end=len(df)-1)
    return forecast_df, y_pred

def forecast_prophet(df):
    df_prophet = df[['timestamp', 'volume']].rename(columns={'timestamp': 'ds', 'volume': 'y'})
    model = Prophet()
    model.fit(df_prophet)
    future = model.make_future_dataframe(periods=24, freq='MS')
    forecast = model.predict(future)
    forecast_df = forecast[['ds', 'yhat']].tail(24)
    y_pred = model.predict(df_prophet)[['yhat']]
    return forecast_df, y_pred['yhat']

def forecast_rf(df):
    X = df[['month_num', 'month', 'year', 'quarter']]
    y = df['volume']
    model = RandomForestRegressor().fit(X, y)
    future = generate_future_dataframe(df)
    forecast = model.predict(future[['month_num', 'month', 'year', 'quarter']])
    forecast_df = pd.DataFrame({'ds': pd.date_range(df['timestamp'].max()+pd.DateOffset(months=1), periods=24, freq='MS'), 'yhat': forecast})
    y_pred = model.predict(X)
    return forecast_df, y_pred

def forecast_xgb(df):
    X = df[['month_num', 'month', 'year', 'quarter']]
    y = df['volume']
    model = xgb.XGBRegressor().fit(X, y)
    future = generate_future_dataframe(df)
    forecast = model.predict(future[['month_num', 'month', 'year', 'quarter']])
    forecast_df = pd.DataFrame({'ds': pd.date_range(df['timestamp'].max()+pd.DateOffset(months=1), periods=24, freq='MS'), 'yhat': forecast})
    y_pred = model.predict(X)
    return forecast_df, y_pred

def forecast_lgb(df):
    X = df[['month_num', 'month', 'year', 'quarter']]
    y = df['volume']
    model = lgb.LGBMRegressor().fit(X, y)
    future = generate_future_dataframe(df)
    forecast = model.predict(future[['month_num', 'month', 'year', 'quarter']])
    forecast_df = pd.DataFrame({'ds': pd.date_range(df['timestamp'].max()+pd.DateOffset(months=1), periods=24, freq='MS'), 'yhat': forecast})
    y_pred = model.predict(X)
    return forecast_df, y_pred

def forecast_lstm(df):
    data = df['volume'].values.reshape(-1, 1)
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    X, y = [], []
    for i in range(3, len(data_scaled)):
        X.append(data_scaled[i-3:i])
        y.append(data_scaled[i])
    X, y = np.array(X), np.array(y)
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(3,1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=50, verbose=0)
    last_input = data_scaled[-3:].reshape(1,3,1)
    forecast = []
    for _ in range(24):
        pred = model.predict(last_input)[0][0]
        forecast.append(pred)
        last_input = np.append(last_input[:,1:,:], [[[pred]]], axis=1)
    forecast = scaler.inverse_transform(np.array(forecast).reshape(-1,1)).flatten()
    forecast_df = pd.DataFrame({'ds': pd.date_range(df['timestamp'].max()+pd.DateOffset(months=1), periods=24, freq='MS'), 'yhat': forecast})
    y_pred = scaler.inverse_transform(model.predict(X)).flatten()
    return forecast_df, y_pred

# Export to Excel
def export_to_excel(results, output_path):
    workbook = xlsxwriter.Workbook(output_path)
    summary = workbook.add_worksheet('Summary')
    summary.write_row(0, 0, ['Model', 'Grade', 'RMSE', 'Explanation'])

    sorted_models = sorted(results.items(), key=lambda x: x[1]['metrics']['RMSE'])
    for rank, (name, data) in enumerate(sorted_models, start=1):
        grade = f"{rank}st" if rank == 1 else f"{rank}nd" if rank == 2 else f"{rank}rd" if rank == 3 else f"{rank}th"
        explanation = f"{name} ranked {grade} based on RMSE of {round(data['metrics']['RMSE'],2)}. It performs well for {data['category']}."
        summary.write_row(rank, 0, [name, grade, data['metrics']['RMSE'], explanation])

        sheet = workbook.add_worksheet(name[:31])
        sheet.write_row(0, 0, ['Date', 'Forecasted Volume'])
        for i, row in enumerate(data['forecast'].itertuples(), start=1):
            sheet.write(i, 0, str(row.ds))
            sheet.write(i, 1, row.yhat)

    workbook.close()

# Streamlit UI
st.title("📊 Agentic AI - Forecasting Tool By Data Quest")
uploaded_file = st.file_uploader("Upload CSV", type="csv")

if uploaded_file:
    df = load_data(uploaded_file)
    models = {
        'ARIMA': (forecast_arima, 'short-term, interpretable'),
        'Prophet': (forecast_prophet, 'multi-product, seasonal'),
        'Random Forest': (forecast_rf, 'short-term, interpretable'),
        'XGBoost': (forecast_xgb, 'high accuracy, nonlinear'),
        'LightGBM': (forecast_lgb, 'high accuracy, nonlinear'),
        'LSTM': (forecast_lstm, 'complex, long-term sequences')
    }

    results = {}
    for name, (func, category) in models.items():
        forecast_df, y_pred = func(df)
        metrics = get_metrics(df['volume'], y_pred)
        results[name] = {
            'forecast': forecast_df,
            'y_pred': y_pred,
            'metrics': metrics,
            'category': category
        }

    # Show actual vs forecast for each model
        # Show actual vs forecast for each model
    st.subheader("📉 Actual vs Forecast")
    for name in models.keys():
        st.write(f"**{name}**")
        y_pred = results[name]['y_pred']
        forecast_df = results[name]['forecast']

        # Actuals
        actual_df = df[['timestamp', 'volume']].copy()
        actual_df.columns = ['ds', 'y']
        actual_df['source'] = 'Actual'

        # Forecasts
        forecast_df = forecast_df.copy()
        forecast_df['source'] = 'Forecast'
        forecast_df['y'] = forecast_df['yhat']

        # Combine actuals and forecast
        combined_df = pd.concat([actual_df, forecast_df], ignore_index=True)

        # Plot
        plt.figure(figsize=(12, 4))
        for src, group in combined_df.groupby('source'):
            plt.plot(group['ds'], group['y'],
                     label=src,
                     linestyle='--' if src == 'Forecast' else '-',
                     marker='o' if src == 'Actual' else None)

        plt.title(f"{name} - Actual vs Forecast")
        plt.xlabel("Month-Year")
        plt.ylabel("Volume")
        plt.legend()
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
        plt.xticks(rotation=45)
        st.pyplot(plt)

    # Export to Excel
    export_to_excel(results, "forecast_summary.xlsx")
    with open("forecast_summary.xlsx", "rb") as f:
        st.download_button("📥 Download Forecast Excel", f, "forecast_summary.xlsx")

    st.success("✅ Forecast complete. Models ranked by RMSE in the summary sheet.")


