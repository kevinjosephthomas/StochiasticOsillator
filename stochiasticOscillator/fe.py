import streamlit as st
import yfinance as yf
import talib
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def preprocess_data(data):
    return data.dropna()

def get_stock_data(symbol, start_date, end_date):
    try:
        data = yf.download(symbol, start=start_date, end=end_date)
        return preprocess_data(data), None
    except Exception as e:
        return None, str(e)

def calculate_technical_indicators(data):
    data['%K'], data['%D'] = talib.STOCH(data['High'], data['Low'], data['Close'])
    data['ROC'] = talib.ROC(data['Close'], timeperiod=14)
    data['RSI'] = talib.RSI(data['Close'])
    macd, signal, hist = talib.MACD(data['Close'])
    data['MACD'] = macd
    data['MACD_Signal'] = signal
    data['MACD_Hist'] = hist
    return data

def analyze_trend(data):
    latest_data = data.iloc[-1]
    if latest_data['%K'] > latest_data['%D']:
        return "Bullish"
    else:
        return "Bearish"

def predict_stock_prices(data, prediction_date):
    features = data[['%K', '%D', 'ROC', 'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist']]
    target = data['Close']
    
    imputer = SimpleImputer(strategy='mean')
    features_imputed = imputer.fit_transform(features)
    
    X_train, X_test, y_train, y_test = train_test_split(features_imputed, target, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predict for the specified date
    prediction_date = pd.Timestamp(prediction_date)
    future_dates = pd.date_range(data.index[-1], prediction_date, freq='B')  # 'B' for business days
    
    if len(future_dates) == 0:
        return None  # Handle case where no future dates are generated
    
    last_close = data['Close'].iloc[-1]
    future_features = []
    
    for _ in future_dates:
        last_close_array = np.array([last_close])
        k, d = talib.STOCH(last_close_array, last_close_array, last_close_array, fastk_period=14, slowk_period=3, slowd_period=3)
        macd, signal, hist = talib.MACD(last_close_array)
        feature_row = [
            k[-1],
            d[-1],
            talib.ROC(last_close_array, timeperiod=14)[-1],
            talib.RSI(last_close_array)[-1],
            macd[-1],
            signal[-1],
            hist[-1]
        ]
        future_features.append(feature_row)
    
    future_features = np.array(future_features)
    future_features_imputed = imputer.transform(future_features)
    future_predictions = model.predict(future_features_imputed)
    
    prediction_result = future_predictions[-1] if len(future_predictions) > 0 else None
    return prediction_result

def generate_summary(data):
    summary = pd.DataFrame(columns=['Metric', 'Value', 'Date'])
    
    for metric in ['Open', 'High', 'Low', 'Close', 'Adj Close']:
        min_value = data[metric].min()
        min_date = data[data[metric] == min_value].index[0]
        summary = pd.concat([summary, pd.DataFrame([{'Metric': f'Min {metric}', 'Value': min_value, 'Date': min_date}])], ignore_index=True)

        max_value = data[metric].max()
        max_date = data[data[metric] == max_value].index[0]
        summary = pd.concat([summary, pd.DataFrame([{'Metric': f'Max {metric}', 'Value': max_value, 'Date': max_date}])], ignore_index=True)
    
    return summary

def interpret_indicators(data):
    interpretations = []
    buy_signal_count = 0
    sell_signal_count = 0

    # Current Trend Analysis
    trend = analyze_trend(data)
    if trend == 'Bullish':
        interpretations.append("Buy signal based on current trend[Bullish].")
        buy_signal_count += 1
    elif trend == 'Bearish':
        interpretations.append("Sell signal based on current trend[Bearish].")
        sell_signal_count += 1

    # Stochastic Oscillator Interpretation
    latest_k = data['%K'].iloc[-1]
    latest_d = data['%D'].iloc[-1]
    if latest_k > 80:
        interpretations.append("Stochastic Oscillator indicates the stock is overbought[Sell].")
        sell_signal_count += 1
    elif latest_k < 20:
        interpretations.append("Stochastic Oscillator indicates the stock is oversold[Buy].")
        buy_signal_count += 1
    if latest_k > latest_d:
        interpretations.append("Stochastic Oscillator shows a bullish signal (%K > %D)[Buy].")
        buy_signal_count += 1
    else:
        interpretations.append("Stochastic Oscillator shows a bearish signal (%K < %D)[Sell].")
        sell_signal_count += 1

    # ROC Interpretation
    latest_roc = data['ROC'].iloc[-1]
    if latest_roc > 0:
        interpretations.append("ROC indicates positive momentum[Buy].")
        buy_signal_count += 1
    else:
        interpretations.append("ROC indicates negative momentum[Sell].")
        sell_signal_count += 1

    # RSI Interpretation
    latest_rsi = data['RSI'].iloc[-1]
    if latest_rsi > 70:
        interpretations.append("RSI indicates the stock is overbought[Sell].")
        sell_signal_count += 1
    elif latest_rsi < 30:
        interpretations.append("RSI indicates the stock is oversold[Buy].")
        buy_signal_count += 1

    # MACD Interpretation
    latest_macd = data['MACD'].iloc[-1]
    latest_signal = data['MACD_Signal'].iloc[-1]
    if latest_macd > latest_signal:
        interpretations.append("MACD shows a bullish signal (MACD > Signal)[Buy].")
        buy_signal_count += 1
    else:
        interpretations.append("MACD shows a bearish signal (MACD < Signal)[Sell].")
        sell_signal_count += 1

    # Generate final buy and sell signals
    if buy_signal_count > sell_signal_count:
        interpretations.append("Final buy signal based on all indicators.")
    elif sell_signal_count > buy_signal_count:
        interpretations.append("Final sell signal based on all indicators.")
    else:
        interpretations.append("No clear buy or sell signal based on all indicators.")

    return interpretations

# Sidebar navigation
navigation = st.sidebar.radio("Navigation", ["Enter Data", "View Data", "Summary", "Stochastic Oscillator Plot", "ROC Plot", "RSI Plot", "MACD Plot", "Interpretation", "Prediction"])

if navigation == "Enter Data":
    st.markdown("<h3 style='color: #4285F4;'> 💹 Technical Analysis Using Stochastic Oscillator </h3>", unsafe_allow_html=True)
    symbol = st.text_input('Stock Symbol', 'NONE')
    start_date = st.date_input('Start Date')
    end_date = st.date_input('End Date')
    if st.button('Fetch Data'):
        data, error = get_stock_data(symbol, start_date, end_date)
        if data is not None:
            st.session_state.data = data
            st.success('Data fetched successfully!')
        else:
            st.error(f'Error fetching data: {error}')

elif navigation == "View Data":
    st.markdown("<h3 style='color: #4285F4;'> 👁️ View Data </h3>", unsafe_allow_html=True)
    if 'data' in st.session_state:
        st.dataframe(st.session_state.data)
    else:
        st.warning('No data available. Please fetch data first.')

elif navigation == "Summary":
    st.markdown("<h3 style='color: #4285F4;'> 🎈 Summary </h3>", unsafe_allow_html=True)
    if 'data' in st.session_state:
        summary = generate_summary(st.session_state.data)
        st.table(summary)
    else:
        st.warning('No data available. Please fetch data first.')

# Stochastic Oscillator Plot
elif navigation == "Stochastic Oscillator Plot":
    st.markdown("<h3 style='color: #4285F4;'> 👍 Stochastic Oscillator Plot </h3>", unsafe_allow_html=True)
    st.image(r"stoc.jpg", caption="Stocastic Oscillator", use_column_width=True)
    if 'data' in st.session_state:
        analyzed_data = calculate_technical_indicators(st.session_state.data)
        fig, ax1 = plt.subplots(figsize=(12, 8))

        ax1.plot(analyzed_data['%K'], label='%K', color='green')
        ax1.plot(analyzed_data['%D'], label='%D', color='red')
        ax1.set_title('Stochastic Oscillator and Close Price')
        ax1.legend(loc='upper left')

        ax2 = ax1.twinx()  # Create a second y-axis that shares the same x-axis
        ax2.plot(analyzed_data['Close'], label='Price', color='blue', alpha=0.5)
        ax2.set_ylabel('Close Price')
        ax2.legend(loc='upper right')

        # Overbought: %K > 80, Oversold: %K < 20
        overbought_date = analyzed_data[analyzed_data['%K'] > 80].index[-1]
        oversold_date = analyzed_data[analyzed_data['%K'] < 20].index[-1]
        overbought_price = analyzed_data['Close'][analyzed_data['%K'] > 80].iloc[-1]
        oversold_price = analyzed_data['Close'][analyzed_data['%K'] < 20].iloc[-1]

        st.write("Overbought: ${:.2f} at {}".format(overbought_price, overbought_date))
        st.write("Oversold: ${:.2f} at {}".format(oversold_price, oversold_date))
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.warning('No data available. Please fetch data first.')

# Rate of Change (ROC) Plot
elif navigation == "ROC Plot":
    st.markdown("<h3 style='color: #4285F4;'> 🕎 Rate Of Change </h3>", unsafe_allow_html=True)
    st.image(r"roc.jpg", caption="ROC", use_column_width=True)
    if 'data' in st.session_state:
        analyzed_data = calculate_technical_indicators(st.session_state.data)
        fig, ax1 = plt.subplots(figsize=(12, 8))

        ax1.plot(analyzed_data['ROC'], label='ROC', color='purple')
        ax1.set_title('Rate of Change (ROC) and Close Price')
        ax1.legend(loc='upper left')

        ax2 = ax1.twinx()
        ax2.plot(analyzed_data['Close'], label='Price', color='blue', alpha=0.5)
        ax2.set_ylabel('Close Price')
        ax2.legend(loc='upper right')

        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.warning('No data available. Please fetch data first.')

# Relative Strength Index (RSI) Plot
elif navigation == "RSI Plot":
    st.markdown("<h3 style='color: #4285F4;'> 😶‍🌫️ Relative Strength Index </h3>", unsafe_allow_html=True)
    st.image(r"rsi.jpg", caption="RSI", use_column_width=True)
    if 'data' in st.session_state:
        analyzed_data = calculate_technical_indicators(st.session_state.data)
        fig, ax1 = plt.subplots(figsize=(12, 8))

        ax1.plot(analyzed_data['RSI'], label='RSI', color='brown')
        ax1.set_title('Relative Strength Index (RSI) and Close Price')
        ax1.legend(loc='upper left')

        ax2 = ax1.twinx()
        ax2.plot(analyzed_data['Close'], label='Price', color='blue', alpha=0.5)
        ax2.set_ylabel('Close Price')
        ax2.legend(loc='upper right')
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.warning('No data available. Please fetch data first.')

# MACD Plot
elif navigation == "MACD Plot":
    st.markdown("<h3 style='color: #4285F4;'> 🎄 Moving Average Convergence Divergence </h3>", unsafe_allow_html=True)
    st.image(r"macd.jpg", caption="MACD", use_column_width=True)
    if 'data' in st.session_state:
        analyzed_data = calculate_technical_indicators(st.session_state.data)
        fig, ax1 = plt.subplots(figsize=(12, 8))

        ax1.plot(analyzed_data['MACD'], label='MACD', color='black')
        ax1.plot(analyzed_data['MACD_Signal'], label='MACD Signal', color='orange')
        ax1.bar(analyzed_data.index, analyzed_data['MACD_Hist'], label='MACD Hist', color='grey', alpha=0.5)
        ax1.set_title('Moving Average Convergence Divergence (MACD) and Close Price')
        ax1.legend(loc='upper left')
        ax2 = ax1.twinx()
        ax2.plot(analyzed_data['Close'], label='Price', color='blue', alpha=0.5)
        ax2.set_ylabel('Close Price')
        ax2.legend(loc='upper right')
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.warning('No data available. Please fetch data first.')
elif navigation == "Interpretation":
    st.markdown("<h3 style='color: #4285F4;'>🚗 Interpretation</h3>", unsafe_allow_html=True)
    st.image(r"analysis.jpg", caption="Conditions", use_column_width=True)
    if 'data' in st.session_state:
        analyzed_data = calculate_technical_indicators(st.session_state.data)
        interpretations = interpret_indicators(analyzed_data)
        for interpretation in interpretations:
            st.write(interpretation)
    else:
        st.warning('No data available. Please fetch data first.')

elif navigation == "Prediction":
    st.markdown("<h3 style='color: #4285F4;'> 🥽Prediction</h3>", unsafe_allow_html=True)
    st.image(r"note.jpg", caption="NOTE", use_column_width=True)
    if 'data' in st.session_state:
        prediction_date = st.date_input('Prediction Date')
        prediction_result = predict_stock_prices(st.session_state.data, prediction_date)
        if prediction_result is not None:
            st.write(f'The predicted stock price for {prediction_date} is {prediction_result}')
        else:
            st.warning('Unable to make prediction. Check the date and try again.')
    else:
        st.warning('No data available. Please fetch data first.')
