from fastapi import FastAPI, HTTPException, Query
import pandas as pd
import joblib

tags_metadata = [
    {
        "name": "Prediction",
        "description": "Generate salary predictions based on job-related inputs."
    }
]

app = FastAPI(
    title="Salary Prediction API",
    description="""
This API predicts the expected salary for data-related roles using a trained Decision Tree model.

### Inputs
- Experience level
- Employment type
- Job title
- Company size
- Employee residence
- Remote ratio
- Company location

### Output
- Predicted salary in USD
""",
    version="1.0.0",
    openapi_tags=tags_metadata,
    swagger_ui_parameters={
        "docExpansion": "none",
        "defaultModelsExpandDepth": -1,
        "syntaxHighlight.theme": "monokai"
    }
)

# Load model and training columns once when the app starts
model = joblib.load("model.pkl")
model_columns = joblib.load("model_columns.pkl")

# Allowed values based on your dataset
ALLOWED_EXPERIENCE_LEVELS = ["EN", "MI", "SE", "EX"]
ALLOWED_EMPLOYMENT_TYPES = ["PT", "FT", "CT", "FL"]
ALLOWED_COMPANY_SIZES = ["S", "M", "L"]
ALLOWED_REMOTE_RATIOS = [0, 50, 100]


@app.get("/predict", tags=["Prediction"], summary="Predict salary")
def predict(
    experience_level: str = Query(..., description="EN, MI, SE, EX"),
    employment_type: str = Query(..., description="PT, FT, CT, FL"),
    job_title: str = Query(..., min_length=1, description="Example: Data Scientist"),
    company_size: str = Query(..., description="S, M, L"),
    employee_residence: str = Query(..., min_length=1, description="Example: US"),
    remote_ratio: int = Query(..., description="Allowed values: 0, 50, 100"),
    company_location: str = Query(..., min_length=1, description="Example: US")
):
    try:
        experience_level = experience_level.strip()
        employment_type = employment_type.strip()
        job_title = job_title.strip()
        company_size = company_size.strip()
        employee_residence = employee_residence.strip()
        company_location = company_location.strip()

        if experience_level not in ALLOWED_EXPERIENCE_LEVELS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid experience_level. Allowed: {ALLOWED_EXPERIENCE_LEVELS}"
            )

        if employment_type not in ALLOWED_EMPLOYMENT_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid employment_type. Allowed: {ALLOWED_EMPLOYMENT_TYPES}"
            )

        if company_size not in ALLOWED_COMPANY_SIZES:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid company_size. Allowed: {ALLOWED_COMPANY_SIZES}"
            )

        if remote_ratio not in ALLOWED_REMOTE_RATIOS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid remote_ratio. Allowed: {ALLOWED_REMOTE_RATIOS}"
            )

        required_columns = [
            f"job_title_{job_title}",
            f"employee_residence_{employee_residence}",
            f"company_location_{company_location}"
        ]

        for col in required_columns:
            if col not in model_columns:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid value. '{col}' was not found in the training data."
                )

        input_data = pd.DataFrame([{
            "experience_level": experience_level,
            "employment_type": employment_type,
            "job_title": job_title,
            "company_size": company_size,
            "employee_residence": employee_residence,
            "remote_ratio": remote_ratio,
            "company_location": company_location
        }])

        input_data = pd.get_dummies(input_data)
        input_data = input_data.reindex(columns=model_columns, fill_value=0)
        input_data = input_data.astype(int)

        prediction = model.predict(input_data)[0]

        return {
            "experience_level": experience_level,
            "employment_type": employment_type,
            "job_title": job_title,
            "company_size": company_size,
            "employee_residence": employee_residence,
            "remote_ratio": remote_ratio,
            "company_location": company_location,
            "predicted_salary": float(prediction)
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")