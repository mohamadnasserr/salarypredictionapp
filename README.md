# Salary Prediction Application

## Overview

This project predicts data science salaries using a trained **Decision Tree Regressor**.  
The application accepts job-related inputs such as experience level, employment type, job title, company size, employee residence, remote ratio, and company location, then returns an estimated salary.

The project includes:
- data cleaning and preprocessing
- Decision Tree model training
- FastAPI prediction API
- Python pipeline to generate many predictions
- analysis text generation
- chart generation
- Supabase storage
- Streamlit dashboard

---

## Architecture

This project follows a pre-generation architecture:

**Local pipeline → Supabase → Streamlit dashboard**

It also includes a **standalone deployed FastAPI API** for salary prediction.

---

## Dataset

Dataset used: **Data Science Job Salaries** from Kaggle.

Target column:
- `salary_in_usd`

Main input features:
- `experience_level`
- `employment_type`
- `job_title`
- `company_size`
- `employee_residence`
- `remote_ratio`
- `company_location`

---

## Machine Learning Model

The salary prediction model is a **Decision Tree Regressor** trained on encoded job-related features.

Evaluation metrics used:
- **MAE**
- **R² Score**

---

## Project Files

- `clean_data.py` → cleans and encodes the dataset
- `train_model.py` → trains the Decision Tree model and saves model files
- `api.py` → FastAPI app for salary prediction
- `client.py` → single API request test
- `pipeline.py` → generates multiple valid salary predictions
- `chart.py` → chart by experience level
- `chart_job_title.py` → chart by job title
- `save_to_supabase.py` → uploads predictions to Supabase
- `streamlit_app.py` → dashboard app
- `model.pkl` → trained model
- `model_columns.pkl` → expected encoded input columns
- `predictions.csv` → generated predictions
- `salary_chart.png` → first visualization
- `salary_by_job_title.png` → second visualization

---

## How to Run Locally

### 1. Install dependencies

```bash
pip install -r requirements.txt