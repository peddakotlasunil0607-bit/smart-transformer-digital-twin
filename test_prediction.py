
import requests

url = "http://127.0.0.1:5000/predict"

data = {
    "location": 1,
    "power": 500,
    "self_protection": 0,
    "avg_earth_ddt": 3.155602,
    "max_earth_ddt": 6.6,
    "burning_rate": 0,
    "criticality_ceramics": 0,
    "removable_connectors": 1,
    "num_users": 30,
    "eens_kwh": 1215,
    "installation_type": 0,
    "air_network": 1,
    "circuit_queue": 0,
    "network_km_lt": 0.12575,
    "year": 2019,
    "client_type_COMERCIAL": 0,
    "client_type_HOUSEHOLD": 1,
    "client_type_INDUSTRIAL": 0,
    "client_type_OFICIAL": 0
}

response = requests.post(url, json=data)

print("Status code:", response.status_code)
print("Prediction result:")
print(response.json())
