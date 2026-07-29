# CodeQuest Arena — DataPython — Previsor de Churn Rate (v2.0)

Gabarito de referência (10/10) para o desafio de previsão de Churn.

## 📁 Arquivos

| Arquivo | Função |
|---|---|
| `generate_dataset.py` | Gera o dataset sintético `telco_churn.csv` usado no desafio |
| `churn_pipeline.py` | Pipeline completo de Data Science (o gabarito) |
| `best_churn_model.pkl` | Modelo treinado salvo, gerado após rodar o pipeline |

## ▶️ Como executar

```bash
pip install pandas scikit-learn joblib
python generate_dataset.py   # cria telco_churn.csv (rodar 1x)
python churn_pipeline.py     # roda o pipeline completo
```

## 🧠 Como o código funciona (passo a passo)

### 1. `generate_dataset.py`
Cria um dataset de 2.000 clientes fictícios com as mesmas colunas do
famoso dataset "Telco Customer Churn". De propósito, insere valores
"sujos" (`" "`, espaço em branco) na coluna `TotalCharges` para os
clientes novos (`tenure = 0`) — reproduzindo um problema real e comum
em dados de produção, para testar se o participante trata isso
corretamente.

### 2. `churn_pipeline.py` — função por função

**`load_and_clean(path)`**
- Lê o CSV com `pd.read_csv`.
- Remove `customerID` (é só um identificador, não um padrão preditivo).
- Converte `TotalCharges` para número com `pd.to_numeric(..., errors="coerce")`
  — qualquer valor que não seja número (como `" "`) vira `NaN` automaticamente.
- Imputa os `NaN` com a **mediana** da coluna (`fillna`). Usamos mediana em
  vez de média porque é mais resistente a outliers (clientes com contas
  muito altas ou muito baixas não distorcem o valor de preenchimento).

**`encode_categoricals(df)`**
- Identifica automaticamente todas as colunas de texto (`select_dtypes`).
- Aplica `LabelEncoder` em cada uma, transformando categorias (ex: "Yes"/"No")
  em números (0/1) que os modelos conseguem processar.

**`train_and_evaluate(df)`**
- Separa `X` (variáveis preditoras) de `y` (o alvo: `Churn`).
- `train_test_split` com `test_size=0.2`, `random_state=42` (reprodutível) e
  `stratify=y` (mantém a proporção de clientes que cancelam igual nos dois
  conjuntos — importante porque churn costuma ser desbalanceado).
- Treina dois modelos dentro de um dicionário `modelos`:
  - **Regressão Logística**: dentro de um `Pipeline` com `StandardScaler`,
    pois esse modelo é sensível à escala das variáveis (ex: `tenure` vai de
    0 a 72, enquanto `TotalCharges` vai a milhares — sem padronizar, a
    variável de maior escala domina o modelo indevidamente).
  - **Random Forest**: não precisa de padronização, pois é baseado em
    divisões de árvore (splits), não em distância/gradiente.
- Para cada modelo, calcula 5 métricas (não só acurácia!):
  - **Acurácia**: % de acertos totais (mas engana em dados desbalanceados).
  - **Precisão**: dos que o modelo disse que iam cancelar, quantos realmente
    cancelaram (evita "alarme falso").
  - **Recall**: dos que realmente cancelaram, quantos o modelo pegou (evita
    "deixar passar" clientes em risco — geralmente a métrica mais importante
    em churn, pois o custo de perder um cliente é alto).
  - **F1-Score**: equilíbrio entre precisão e recall.
  - **AUC-ROC**: capacidade do modelo de separar as duas classes,
    independente do limiar de decisão.
- Mostra a **matriz de confusão** (acertos/erros detalhados por classe).
- Escolhe o **melhor modelo pelo F1-Score** (mais robusto que acurácia
  para esse tipo de problema).
- Mostra as **5 variáveis mais importantes** segundo o Random Forest
  (`feature_importances_`), dando interpretabilidade ao resultado.

**`save_model(modelo, nome, path)`**
- Salva o modelo vencedor em disco com `joblib.dump`, permitindo reutilizá-lo
  depois sem precisar retreinar (`joblib.load("best_churn_model.pkl")`).

## 📏 Critérios de pontuação sugeridos para os juízes

| Critério | Peso |
|---|---|
| Pipeline executa sem erro | 20% |
| Tratamento correto de nulos e categóricas | 20% |
| Métricas corretas (não só acurácia) | 25% |
| Comparação entre 2 modelos | 15% |
| Organização/legibilidade do código | 10% |
| Modelo salvo corretamente | 10% |

## 💡 Possíveis extensões (para bônus/desempate entre participantes)

- Balanceamento de classes com `class_weight="balanced"` ou SMOTE.
- Validação cruzada (`cross_val_score`) em vez de um único split.
- `GridSearchCV` para otimizar hiperparâmetros.
- One-Hot Encoding em vez de Label Encoding para colunas sem ordem natural.
