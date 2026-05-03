# TA1 — Predicting Stock Prices Using RNN with LSTM

Predicts AAPL (S&P 500) daily close prices with a stacked LSTM.

## Setup
```powershell
pip install -r requirements.txt
jupyter notebook stock_price_lstm.ipynb
```

## Pipeline
1. **Data Preparation** — download AAPL 2019-2024 via `yfinance`; select `Close`; 80/20 chronological split; MinMax scaling (fit on train only); 60-day sliding-window sequences.
2. **Model Development** — Stacked LSTM (64 → 64) with Dropout(0.2) + Dense head.
3. **Training** — Adam + MSE loss, EarlyStopping (patience=8), validation split 10%.
4. **Prediction** — one-step predictions on test set + iterative 30-day future forecast.
5. **Evaluation** — MAE, RMSE, MAPE + line and candlestick visualizations.

## Files
- `stock_price_lstm.ipynb` — complete, section-labeled notebook (deliverable).
- `requirements.txt` — Python dependencies.
