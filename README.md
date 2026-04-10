# Salary Prediction Application

## Overview

This project is an end-to-end machine learning application that predicts data science salaries using a trained **Decision Tree Regressor**.

The system takes in job-related details such as:
- experience level
- employment type
- job title
- company size
- employee residence
- remote ratio
- company location

It then predicts the expected salary in USD through a **FastAPI GET endpoint**.

The project also includes:
- dataset cleaning and preprocessing
- a prediction pipeline that generates multiple salary predictions
- narrative analysis text
- multiple visualizations
- Supabase storage
- a live Streamlit dashboard

---

## Project Architecture

This project follows a pre-generation architecture:

**Local pipeline → Supabase → Streamlit dashboard**

A separate deployed **FastAPI API** is also included as a standalone deliverable.

### Flow
1. Clean and preprocess the salary dataset
2. Train a Decision Tree model
3. Expose the model through a FastAPI API
4. Generate multiple prediction combinations through a Python pipeline
5. Create analysis text and charts
6. Store all results in Supabase
7. Display the results on a Streamlit dashboard

---

## Dataset

Dataset used: **Data Science Job Salaries** from Kaggle.

### Target Variable
- `salary_in_usd`

### Input Features
- `experience_level`
- `employment_type`
- `job_title`
- `company_size`
- `employee_residence`
- `remote_ratio`
- `company_location`

---

## Machine Learning Model

The machine learning model used is a **Decision Tree Regressor**.

The model was trained on encoded job-related features and evaluated using:

- **MAE (Mean Absolute Error)**
- **R² Score**

The trained model is saved as:
- `model.pkl`

The expected model input columns are saved as:
- `model_columns.pkl`

---

## Features Implemented

### 1. Data Cleaning
The dataset is cleaned by:
- removing duplicates
- selecting relevant columns
- handling missing values
- encoding categorical features

### 2. Decision Tree Training
A Decision Tree model is trained on the cleaned dataset and saved for inference.

### 3. GET API for Prediction
A FastAPI endpoint accepts validated input parameters and returns a predicted salary.

### 4. Python Prediction Pipeline
A pipeline script generates multiple valid combinations of inputs and calls the API automatically.

### 5. Analysis + Visualization
The project generates:
- a written salary analysis
- a chart by experience level
- a chart by job title

### 6. Supabase Persistence
Prediction records, analysis text, and chart references are stored in Supabase.

### 7. Streamlit Dashboard
The dashboard allows users to:
- explore stored results
- view salary metrics
- inspect prediction records
- interact with a live prediction panel
- view salary charts

---

## Project Files

- `clean_data.py` → cleans and encodes the dataset
- `train_model.py` → trains the model and saves artifacts
- `api.py` → FastAPI app for salary prediction
- `client.py` → single test request to the API
- `pipeline.py` → generates multiple valid prediction combinations
- `chart.py` → creates salary chart by experience level
- `chart_job_title.py` → creates salary chart by job title
- `save_to_supabase.py` → uploads predictions to Supabase
- `streamlit_app.py` → Streamlit dashboard
- `model.pkl` → trained Decision Tree model
- `model_columns.pkl` → expected encoded columns
- `predictions.csv` → generated predictions
- `salary_chart.png` → chart by experience level
- `salary_by_job_title.png` → chart by job title

---

## 🚀 Deployment Deliverables

### 📊 Interactive Dashboard
The frontend is built using **Streamlit**, allowing users to input data and receive real-time salary predictions through a user-friendly interface.
* **Live URL:** [Salary Prediction Dashboard](https://salarypredictionapp-u49nf7z52thhmqtazwytss.streamlit.app/)

### ⚡ backend API
The backend service is powered by **FastAPI**, providing high-performance endpoints for model inference.
* **Production API:** [https://salarypredictionapp.onrender.com](https://salarypredictionapp.onrender.com)

### 📖 API Documentation
The API is fully documented and can be tested directly via the interactive Swagger UI.
* **Swagger Docs:** [View API Documentation](https://salarypredictionapp.onrender.com/docs)

---

## 🛠️ Tech Stack
- **Frontend:** Streamlit
- **Backend:** FastAPI
- **Deployment:** Render / Streamlit Cloud

---

## API Example

### Endpoint
`/predict`

### Example Request

```text
/predict?experience_level=SE&employment_type=FT&job_title=Data%20Scientist&company_size=M&employee_residence=US&remote_ratio=100&company_location=US