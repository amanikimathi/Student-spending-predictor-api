from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model and the expected column structure, once, at startup
model = joblib.load("spending_model.pkl")
model_columns = joblib.load("model_columns.pkl")


class StudentInfo(BaseModel):
    age: int
    monthly_income: float
    financial_aid: float
    gender: str
    year_in_school: str
    major: str
    preferred_payment_method: str


@app.get("/")
def read_root():
    return {"message": "Student Spending Predictor API is running"}


@app.post("/predict")
def predict_spending(student: StudentInfo):
    # Build a single-row DataFrame from the submitted data
    input_data = pd.DataFrame([{
        "age": student.age,
        "monthly_income": student.monthly_income,
        "financial_aid": student.financial_aid,
        f"gender_{student.gender}": True,
        f"year_in_school_{student.year_in_school}": True,
        f"major_{student.major}": True,
        f"preferred_payment_method_{student.preferred_payment_method}": True,
    }])

    # Add any missing one-hot columns as False, and match the exact column order the model expects
    input_data = input_data.reindex(columns=model_columns, fill_value=False)

    prediction = model.predict(input_data)[0]

    return {"predicted_monthly_spending": round(float(prediction), 2)}