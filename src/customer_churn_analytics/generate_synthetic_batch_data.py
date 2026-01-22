import pandas as pd
import numpy as np

np.random.seed(42)

N = 30  # number of synthetic customers

def random_yes_no(p=0.5):
    return np.random.choice(["Yes", "No"], p=[p, 1 - p])

data = []

for _ in range(N):
    tenure = np.random.randint(0, 72)

    monthly_charges = np.round(
        np.random.uniform(20, 110), 2
    )

    total_charges = np.round(
        monthly_charges * tenure + np.random.uniform(-50, 50), 2
    )

    contract = np.random.choice(
        ["Month-to-month", "One year", "Two year"],
        p=[0.6, 0.25, 0.15]
    )

    internet = np.random.choice(
        ["DSL", "Fiber optic", "No"],
        p=[0.35, 0.5, 0.15]
    )

    row = {
        "gender": np.random.choice(["Male", "Female"]),
        "SeniorCitizen": np.random.choice([0, 1], p=[0.84, 0.16]),
        "Partner": random_yes_no(0.55),
        "Dependents": random_yes_no(0.45),
        "tenure": tenure,
        "PhoneService": "Yes",
        "MultipleLines": np.random.choice(
            ["Yes", "No", "No phone service"],
            p=[0.4, 0.5, 0.1]
        ),
        "InternetService": internet,
        "OnlineSecurity": random_yes_no(0.4 if internet != "No" else 0.0),
        "OnlineBackup": random_yes_no(0.5 if internet != "No" else 0.0),
        "DeviceProtection": random_yes_no(0.5 if internet != "No" else 0.0),
        "TechSupport": random_yes_no(0.3 if internet != "No" else 0.0),
        "StreamingTV": random_yes_no(0.6 if internet != "No" else 0.0),
        "StreamingMovies": random_yes_no(0.6 if internet != "No" else 0.0),
        "Contract": contract,
        "PaperlessBilling": random_yes_no(0.6),
        "PaymentMethod": np.random.choice(
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        ),
        "MonthlyCharges": monthly_charges,
        "TotalCharges": max(total_charges, 0)
    }

    data.append(row)

df = pd.DataFrame(data)
df.to_csv("synthetic_telco_batch_input_synthetic_data.csv", index=False)

print(" Synthetic batch data generated: synthetic_telco_batch_input.csv")
