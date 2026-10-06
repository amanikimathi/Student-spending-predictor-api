# Student Spending Predictor — API

A machine learning project that trains a regression model to predict student discretionary monthly spending, then serves live predictions through a deployed REST API.

**Live API docs:** https://student-spending-predictor-api.onrender.com/docs
**Training notebook & full analysis:** https://github.com/amanithomas12/student-spending-predictor

## What this is

This is the second half of the Student Spending Predictor project. The [training notebook](https://github.com/amanithomas12/student-spending-predictor) covers the full ML workflow — data cleaning, feature engineering, catching and fixing data leakage, training and comparing Linear Regression and Random Forest models, and honestly evaluating the results.

This repo takes the trained model out of the notebook and serves it through a live FastAPI backend, so it can actually be used,not just described. A simple frontend form was also built locally to demonstrate the model being consumed by a client application, though it isn't currently deployed.

## Try it

Visit the [live API docs](https://student-spending-predictor-api.onrender.com/docs), open `POST /predict`, click "Try it out," and submit something like:

```json
{
  "age": 20,
  "monthly_allowance": 1000,
  "financial_aid": 500,
  "gender": "Female",
  "year_in_school": "Sophomore",
  "major": "Computer Science",
  "preferred_payment_method": "Credit/Debit Card"
}
```

## Honest limitation

The underlying model has weak predictive power (R² near zero on held-out test data — see the training notebook for the full evaluation). This API is a working demonstration of the **ML-to-production pipeline** — training a model, saving it, loading it in a live service, and serving predictions through a REST endpoint — not a financially accurate predictor. A stronger dataset with more real signal would be needed for that.

## Tech stack

Python, FastAPI, scikit-learn, joblib, pandas

## Running locally

```bash
git clone https://github.com/amanithomas12/student-spending-predictor-api.git
cd student-spending-predictor-api
python -m venv venv
venv\Scripts\activate      # Windows
pip install -r requirements.txt
uvicorn main:app --reload
```

Visit `http://127.0.0.1:8000/docs`.

## What I'd do with more time

- Retrain on a larger, richer dataset with features more likely to predict spending (e.g. location, lifestyle habits)
- Add input validation with clearer error messages for out-of-range values
- Deploy the accompanying frontend for a fully click-through demo
