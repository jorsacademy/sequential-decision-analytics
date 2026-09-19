# Project 1 — GRU vs Transformer for Multi-Horizon Demand Forecasting

## Industrial-engineering question

Given recent demand and exogenous operating signals, can we forecast the next several periods well enough to improve a simple replenishment decision?

This project compares two sequence-model families:

- **GRU**: a recurrent model with gated state updates;
- **Transformer encoder**: a self-attention model that can directly compare distant positions in the input window.

The goal is not to declare one architecture universally superior. The point is to learn when each representation is useful and to compare them under both predictive and decision-oriented metrics.

## What the script does

1. Generates synthetic daily demand with:
   - trend,
   - weekly seasonality,
   - promotions,
   - price effects,
   - random shocks.
2. Builds rolling input windows and seven-step forecasting targets.
3. Trains a GRU model.
4. Trains a Transformer encoder.
5. Reports:
   - MAE,
   - RMSE,
   - a simple asymmetric inventory-cost proxy.

## Why the cost proxy matters

A model with a slightly lower RMSE is not automatically more useful. Under-forecasting may create shortage costs while over-forecasting creates holding costs.

The project therefore evaluates

```text
cost = holding_cost * over_forecast + shortage_cost * under_forecast
```

with a higher penalty for under-forecasting.

## Extensions

- Replace synthetic data with SKU-level demand.
- Add known-future covariates such as calendar and promotions.
- Compare direct multi-horizon forecasting with recursive forecasting.
- Add probabilistic forecasts and optimize safety stock from quantiles.
- Couple forecasts to a MILP or stochastic inventory model.
