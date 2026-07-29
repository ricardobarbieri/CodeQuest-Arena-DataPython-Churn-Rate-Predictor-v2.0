"""
churn_pipeline.py
==================
CodeQuest Arena | DataPython | Previsor de Churn Rate (v2.0 - gabarito 10/10)

O que este script faz, passo a passo:
  1. Carrega o dataset telco_churn.csv
  2. LIMPEZA: converte TotalCharges para número e imputa nulos pela mediana
  3. CODIFICAÇÃO: transforma colunas categóricas (texto) em números
  4. SPLIT: separa 80% treino / 20% teste, com seed fixa (reprodutibilidade)
  5. TREINO: treina Regressão Logística e Random Forest
  6. AVALIAÇÃO: calcula Acurácia, Precisão, Recall, F1, AUC-ROC e matriz
     de confusão para os dois modelos, e mostra qual venceu
  7. INTERPRETABILIDADE: mostra as 5 variáveis mais importantes segundo
     o Random Forest
  8. PERSISTÊNCIA: salva o melhor modelo em disco (best_churn_model.pkl)

Como executar:
    python generate_dataset.py   # (só na primeira vez, cria o CSV)
    python churn_pipeline.py

Dependências:
    pip install pandas scikit-learn joblib
"""

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

RANDOM_STATE = 42  # fixo em todo o script -> resultados reprodutíveis
CSV_PATH = "telco_churn.csv"


# ---------------------------------------------------------------------------
# ETAPA 1: CARGA E LIMPEZA DOS DADOS
# ---------------------------------------------------------------------------
def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # 'customerID' é só um identificador, não ajuda o modelo a aprender
    # padrões -> removemos para não "vazar" ruído inútil.
    df = df.drop(columns=["customerID"])

    # No dataset original (e no nosso simulado), TotalCharges vem como
    # texto e alguns clientes novos (tenure=0) têm espaço em branco " "
    # em vez de número. Passo a passo do tratamento:
    #   a) força conversão numérica; o que não for número vira NaN
    #   b) imputa os NaN com a MEDIANA da própria coluna
    #      (mediana é mais robusta a outliers que a média)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    mediana_total_charges = df["TotalCharges"].median()
    n_nulos = df["TotalCharges"].isna().sum()
    df["TotalCharges"] = df["TotalCharges"].fillna(mediana_total_charges)

    print(f"[LIMPEZA] {n_nulos} valores nulos encontrados em 'TotalCharges'")
    print(f"[LIMPEZA] Imputados com a mediana = {mediana_total_charges:.2f}")

    return df


# ---------------------------------------------------------------------------
# ETAPA 2: CODIFICAÇÃO DE VARIÁVEIS CATEGÓRICAS
# ---------------------------------------------------------------------------
def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Colunas de texto (object) precisam virar número para os modelos
    # de Scikit-Learn. Usamos LabelEncoder por simplicidade/didática;
    # em produção, colunas sem ordem natural (ex: PaymentMethod)
    # normalmente pediriam One-Hot Encoding, mas Label Encoding é
    # aceitável aqui e mantém o dataset compacto.
    colunas_categoricas = df.select_dtypes(include=["object"]).columns.tolist()

    encoders = {}
    for col in colunas_categoricas:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    print(f"[ENCODING] Colunas categóricas codificadas: {colunas_categoricas}")
    return df


# ---------------------------------------------------------------------------
# ETAPA 3: TREINO E COMPARAÇÃO DE MODELOS
# ---------------------------------------------------------------------------
def train_and_evaluate(df: pd.DataFrame):
    X = df.drop(columns=["Churn"])
    y = df["Churn"]  # já codificado: 0 = No, 1 = Yes

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    # stratify=y garante que a proporção de churners no treino e no
    # teste seja a mesma do dataset original -> avaliação mais justa
    # em problemas desbalanceados (que é o caso típico de churn).

    # A Regressão Logística é sensível à escala das variáveis (ex:
    # 'tenure' vai de 0-72 e 'TotalCharges' vai a milhares) -> usamos
    # um Pipeline com StandardScaler antes do modelo. O Random Forest
    # não precisa disso, pois é baseado em divisões (splits), não em
    # distância/gradiente.
    from sklearn.pipeline import Pipeline

    modelos = {
        "Regressão Logística": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
    }

    resultados = {}

    for nome, modelo in modelos.items():
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)
        y_proba = modelo.predict_proba(X_test)[:, 1]  # probabilidade da classe "Yes"

        metricas = {
            "acuracia": accuracy_score(y_test, y_pred),
            "precisao": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "auc_roc": roc_auc_score(y_test, y_proba),
            "matriz_confusao": confusion_matrix(y_test, y_pred),
        }
        resultados[nome] = {"modelo": modelo, "metricas": metricas}

        print(f"\n=== {nome} ===")
        print(f"Acurácia : {metricas['acuracia']:.4f}")
        print(f"Precisão : {metricas['precisao']:.4f}")
        print(f"Recall   : {metricas['recall']:.4f}")
        print(f"F1-Score : {metricas['f1']:.4f}")
        print(f"AUC-ROC  : {metricas['auc_roc']:.4f}")
        print("Matriz de Confusão:")
        print(metricas["matriz_confusao"])

    # Escolhe o melhor modelo pelo F1-Score (métrica mais equilibrada
    # que acurácia sozinha para problemas desbalanceados como churn)
    melhor_nome = max(resultados, key=lambda k: resultados[k]["metricas"]["f1"])
    melhor_modelo = resultados[melhor_nome]["modelo"]
    print(f"\n>>> Melhor modelo (por F1-Score): {melhor_nome}")

    # Importância das features (só o Random Forest expõe isso nativamente)
    rf_model = resultados["Random Forest"]["modelo"]
    importancias = pd.Series(rf_model.feature_importances_, index=X.columns)
    top5 = importancias.sort_values(ascending=False).head(5)
    print("\n[INTERPRETABILIDADE] Top 5 variáveis mais importantes (Random Forest):")
    print(top5.to_string())

    return melhor_modelo, melhor_nome


# ---------------------------------------------------------------------------
# ETAPA 4: PERSISTÊNCIA DO MODELO
# ---------------------------------------------------------------------------
def save_model(modelo, nome: str, path: str = "best_churn_model.pkl"):
    joblib.dump(modelo, path)
    print(f"\n[PERSISTÊNCIA] Modelo vencedor ('{nome}') salvo em: {path}")


# ---------------------------------------------------------------------------
# EXECUÇÃO
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df = load_and_clean(CSV_PATH)
    df = encode_categoricals(df)
    melhor_modelo, melhor_nome = train_and_evaluate(df)
    save_model(melhor_modelo, melhor_nome)