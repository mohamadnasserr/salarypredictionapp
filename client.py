import requests

url = "http://127.0.0.1:8000/predict"

params = {
    "experience_level": "SE",
    "employment_type": "FT",
    "job_title": "Data Scientist",
    "company_size": "M",
    "employee_residence": "US",
    "remote_ratio": 100,
    "company_location": "US"
}

try:
    response = requests.get(url, params=params, timeout=15)

    if response.status_code == 200:
        data = response.json()
        print("Predicted Salary:", data["predicted_salary"])
        print("Full response:", data)
    else:
        print("Error:", response.status_code)
        print(response.text)

except requests.exceptions.RequestException as e:
    print("Request failed:", e)