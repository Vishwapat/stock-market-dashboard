# 📊 Stock Market Dashboard

An interactive web dashboard for exploring stock price trends, 
volatility, and momentum — built with Python and Streamlit.

## Features
- Live data for any stock ticker (US & Canadian markets)
- Moving average overlay (20, 50, 100, 200-day)
- Daily returns bar chart with gain/loss colouring
- Annualized rolling volatility
- Linear regression trend line with R² score

## Tech
Python · Streamlit · yfinance · pandas · NumPy · scikit-learn · matplotlib

## Run Locally
pip install -r requirements.txt
streamlit run app.py

## Deploy Free
[![Deploy to Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io/cloud)