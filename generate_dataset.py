"""
generate_dataset.py
--------------------
Gera um dataset sintético 'telco_churn.csv' com a MESMA estrutura do
famoso "Telco Customer Churn" (IBM/Kaggle), incluindo valores nulos
propositais em 'TotalCharges' (como no dataset original) para que o
critério de "tratamento de nulos" da competição seja testável.

Uso:
    python generate_dataset.py
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 2000  # número de clientes simulados

genders = np.random.choice(["Male", "Female"], N)
senior = np.random.choice([0, 1], N, p=[0.85, 0.15])
partner = np.random.choice(["Yes", "No"], N)
dependents = np.random.choice(["Yes", "No"], N)
tenure = np.random.randint(0, 73, N)  # meses de contrato
phone_service = np.random.choice(["Yes", "No"], N, p=[0.9, 0.1])
internet_service = np.random.choice(["DSL", "Fiber optic", "No"], N, p=[0.35, 0.45, 0.2])
contract = np.random.choice(["Month-to-month", "One year", "Two year"], N, p=[0.55, 0.25, 0.2])
payment_method = np.random.choice(
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], N
)
paperless_billing = np.random.choice(["Yes", "No"], N)
monthly_charges = np.round(np.random.uniform(18, 120, N), 2)

# TotalCharges correlacionado com tenure * monthly_charges, com ruído
total_charges = np.round(monthly_charges * tenure + np.random.normal(0, 50, N), 2)
total_charges = np.where(total_charges < 0, 0, total_charges)

# Regra latente de churn (para o dataset ter sinal aprendível):
# contratos mensais, fibra, tenure baixo e mensalidade alta -> mais churn
churn_score = (
    (contract == "Month-to-month") * 0.35
    + (internet_service == "Fiber optic") * 0.20
    + (tenure < 12) * 0.25
    + (monthly_charges > 80) * 0.15
    + np.random.normal(0, 0.15, N)
)
churn_prob = 1 / (1 + np.exp(-(churn_score - 0.5) * 5))  # sigmoide
churn = np.where(np.random.rand(N) < churn_prob, "Yes", "No")

df = pd.DataFrame({
    "customerID": [f"C{i:05d}" for i in range(N)],
    "gender": genders,
    "SeniorCitizen": senior,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "PhoneService": phone_service,
    "InternetService": internet_service,
    "Contract": contract,
    "PaperlessBilling": paperless_billing,
    "PaymentMethod": payment_method,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges.astype(str),
    "Churn": churn,
})

# Simula o "problema real" do dataset original: TotalCharges como string
# com espaços em branco em vez de nulo explícito, para clientes com tenure=0
mask_new_customers = df["tenure"] == 0
df.loc[mask_new_customers, "TotalCharges"] = " "  # valor "sujo" proposital

df.to_csv("telco_churn.csv", index=False)
print(f"Dataset gerado: telco_churn.csv ({len(df)} linhas)")
print(f"Valores 'sujos' em TotalCharges: {(df['TotalCharges'] == ' ').sum()}")
