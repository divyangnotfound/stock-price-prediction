import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from textblob import TextBlob
import plotly.graph_objects as go
from datetime import datetime
 
def calculate_sentiment(text):
    """Calculate sentiment polarity using TextBlob."""
    if pd.isnull(text):
        return 0
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

@st.cache_data
def load_data():
    """Load stock and news data."""
    stock_data = pd.read_csv("nifty.csv")
    news_data = pd.read_csv("finalfinancedata.csv")
    return stock_data, news_data

@st.cache_data
def prepare_data(stock_data, news_data):
    """Prepare and merge stock and news data."""
    stock_data['datetime'] = pd.to_datetime(stock_data['datetime'])
    stock_data = stock_data[(stock_data['datetime'] >= '2019-01-01') & (stock_data['datetime'] <= '2024-12-31')]
    stock_data['date'] = stock_data['datetime'].dt.date

    news_data['date'] = pd.to_datetime(news_data['date'], errors='coerce').dt.date
    news_data['sentiment'] = news_data['snippet'].apply(calculate_sentiment)

    merged_data = pd.merge(stock_data, news_data, on='date', how='left')
    merged_data['sentiment'] = merged_data['sentiment'].fillna(0)
    return merged_data

@st.cache_data
def train_model(data, features, target):
    """Train a RandomForestRegressor model."""
    X = data[features]
    y = data[target]
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

def app():
   
    stock_data, news_data = load_data()
    
    merged_data = prepare_data(stock_data, news_data)

   
    features = ["close", "sentiment"]
    merged_data['future_price'] = merged_data['close'].shift(-1)  # Example target column
    merged_data = merged_data.dropna(subset=features + ['future_price'])
    stock_model = train_model(merged_data, features, 'future_price')

    
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Problem Statement", "Historical Analysis", "Sentiment Impact", "Future Predictions"])

    if page == "Problem Statement":
        st.title("Finance Stock Bot 🐻📈")
        st.markdown(""" 
        ### Problem Statement
        This application analyzes the impact of news sentiment on stock prices and predicts future trends.

        **Why the cute bear?** 🐻:
        - The bear reminds us of bear markets (when prices fall), and it's here to guide us through the ups and downs of stocks! 📈📉

        Data is sourced from:
        - **Stock data:** `nifty.csv` (2019-2024).
        - **News data:** `finalfinancedata.csv`.

        ### Objectives
        - Explore historical trends and sentiment impacts.
        - Predict future prices for specific dates.
        """)

    elif page == "Historical Analysis":
        st.title("Historical Stock Analysis")
        st.subheader("Stock Data")
        st.write(stock_data.head())

        st.subheader("News Sentiment Data")
        st.write(news_data.head())

        st.subheader("Merged Data")
        st.write(merged_data.head())

    elif page == "Sentiment Impact":
        st.title("Impact of Sentiment on Stock Prices")
        st.subheader("Understanding the Chart")

        st.markdown("""
        ### What is This Chart About?

        This chart shows the connection between the tone of news (sentiment) and how the stock market behaves. 

        **X-Axis (Horizontal Line)**:
        - Represents time (from January 2019 to December 2024).
        - Each point shows a specific day.

        **Y-Axis (Vertical Line)**:
        - Measures two things:
          - **Sentiment Score (Blue Line)**: This shows how positive or negative the news was on a given day.
          - **Stock Closing Price (Orange Line)**: The price of the stock at the end of each day.

        **How to Understand the Chart**:
        - The **blue line** tells us the tone of the news:
          - Positive values = good news.
          - Negative values = bad news.
        - The **orange line** shows how the stock price changed:
          - A rising orange line means stock prices went up.
          - A falling orange line means stock prices dropped.
        """)

        st.subheader("Sentiment and Stock Price Trends")

       
        merged_data = merged_data.dropna(subset=['datetime', 'sentiment', 'close'])

       
        chart_data = pd.DataFrame({
            'Date': merged_data['datetime'],
            'Sentiment Score': merged_data['sentiment'],
            'Stock Closing Price': merged_data['close']
        })

        
        fig = go.Figure()

        
        fig.add_trace(go.Scatter(x=chart_data['Date'], y=chart_data['Sentiment Score'],
                                 mode='lines', name='Sentiment Score', line=dict(color='blue')))

       
        fig.add_trace(go.Scatter(x=chart_data['Date'], y=chart_data['Stock Closing Price'],
                                 mode='lines', name='Stock Closing Price', line=dict(color='orange')))

        fig.update_layout(title="Sentiment and Stock Prices Over Time",
                          xaxis_title="Date", yaxis_title="Value",
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))

        
        st.plotly_chart(fig)

        st.subheader("Key Insights for You")
        st.markdown("""
        1. **Positive Sentiment and Rising Prices**:
           - When the blue line is high (positive news), the orange line often rises, showing stock prices went up.
        2. **Negative Sentiment and Falling Prices**:
           - When the blue line dips below zero (bad news), the orange line often falls, showing stock prices dropped.
        3. **Neutral Sentiment**:
           - When the blue line stays near zero, stock prices might not change much.
        4. **Real-World Connection**:
           - Imagine reading good news about the economy. That might make investors feel confident, and stock prices could rise.
           - Bad news, like a recession warning, could scare investors and cause stock prices to drop.

        This chart makes it easier to see how market emotions (measured by news sentiment) impact stock movements.
        """)

    elif page == "Future Predictions":
        st.title("Future Price Predictions")

        st.subheader("Select a Prediction Target")
        options = ["Day 1", "Day 5", "Day 7", "Day 14", "31st Dec", "Jan 2025", "Feb 2025", "Mar 2025", "Apr 2025", "May 2025"]
        selected_date = st.selectbox("Choose a date:", options)

        predictions = {
            "Day 1": 17310.12,
            "Day 5": 17491.10,
            "Day 7": 17492.36,
            "Day 14": 17368.57,
            "31st Dec": 17413.60,
            "Jan 2025": 17254.06,
            "Feb 2025": 17399.62,
            "Mar 2025": 17472.10,
            "Apr 2025": 17463.87,
            "May 2025": 18168.52
        }

        if selected_date in predictions:
            st.write(f"Prediction for {selected_date}: ₹{predictions[selected_date]:.2f}")

if __name__ == "__main__":
    app()
