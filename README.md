# 🎯 CodeQuest Arena — Churn Rate Predictor v2.0

Pipeline de Data Science em Python (Pandas + Scikit-Learn) para prever cancelamento de assinaturas (**Churn**), com interface gráfica (GUI) para execução visual. Desenvolvido como solução de referência (gabarito 10/10) do desafio **DataPython** da competição **CodeQuest Arena**.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Cleaning-150458?logo=pandas)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?logo=scikit-learn)
![License](https://img.shields.io/badge/license-MIT-green)

---

## 📌 Sobre o projeto

Este projeto implementa um pipeline completo e reprodutível de Machine Learning para prever quais clientes têm maior probabilidade de cancelar um serviço (churn), a partir de um dataset no estilo do clássico *Telco Customer Churn*. Ele foi construído para ser ao mesmo tempo:

- **Funcional** — treina, avalia e salva um modelo real
- **Didático** — cada etapa é comentada explicando o *porquê*, não só o *como*
- **Visual** — inclui uma interface gráfica simples para acompanhar a execução sem precisar do terminal

## ✨ Funcionalidades

- 🧹 Limpeza de dados (tratamento de valores nulos com imputação por mediana)
- 🔤 Codificação automática de variáveis categóricas
- 🤖 Treino comparativo entre **Regressão Logística** e **Random Forest**
- 📊 Avaliação com múltiplas métricas: Acurácia, Precisão, Recall, F1-Score e AUC-ROC
- 🧩 Matriz de confusão e ranking das variáveis mais importantes
- 💾 Exportação automática do melhor modelo (`.pkl`)
- 🖥️ Interface gráfica (Tkinter) para rodar o pipeline com um clique

## 🖼️ Screenshot

> *(adicione aqui um print da GUI em execução, ex: `![GUI](docs/screenshot.png)`)*

## 📁 Estrutura do repositório

```
Churn Rate Predictor v2.0/
│
├── generate_dataset.py     # Gera o dataset sintético telco_churn.csv
├── churn_pipeline.py       # Pipeline de ML (limpeza, treino, avaliação, salvamento)
├── churn_gui.py            # Interface gráfica que executa o pipeline
├── telco_churn.csv         # Dataset gerado (após rodar generate_dataset.py)
├── best_churn_model.pkl    # Modelo treinado salvo (gerado após rodar o pipeline)
└── README.md
```

## ⚙️ Requisitos

- Python 3.9 ou superior
- Bibliotecas: `pandas`, `scikit-learn`, `joblib` (Tkinter já vem no Python padrão)

## 🚀 Instalação e uso

```bash
# 1. Clone o repositório
git clone https://github.com/seu-usuario/codequest-arena-churn.git
cd codequest-arena-churn

# 2. Instale as dependências
pip install pandas scikit-learn joblib

# 3. Gere o dataset (rodar apenas uma vez)
python generate_dataset.py

# 4. Execute o pipeline via terminal...
python churn_pipeline.py

# ...ou via interface gráfica
python churn_gui.py
```

### Saída esperada (terminal)

```
[LIMPEZA] 23 valores nulos encontrados em 'TotalCharges'
[LIMPEZA] Imputados com a mediana = 2078.85
[ENCODING] Colunas categóricas codificadas: [...]

=== Regressão Logística ===
Acurácia : 0.6775
Precisão : 0.5902
Recall   : 0.4768
F1-Score : 0.5275
AUC-ROC  : 0.7145
...

>>> Melhor modelo (por F1-Score): Regressão Logística

[INTERPRETABILIDADE] Top 5 variáveis mais importantes (Random Forest):
MonthlyCharges    0.203848
TotalCharges      0.201140
tenure            0.176385
...

[PERSISTÊNCIA] Modelo vencedor salvo em: best_churn_model.pkl
```

## 🧠 Metodologia

| Etapa | Técnica utilizada | Justificativa |
|---|---|---|
| Tratamento de nulos | Imputação pela **mediana** | Mais robusta a outliers que a média |
| Variáveis categóricas | `LabelEncoder` | Converte texto em número para os modelos |
| Escala das variáveis | `StandardScaler` (só na Regressão Logística) | Modelos baseados em distância/gradiente são sensíveis à escala; árvores não |
| Divisão treino/teste | 80/20 com `stratify=y` | Mantém a proporção de churners nos dois conjuntos (dados desbalanceados) |
| Seleção do melhor modelo | Maior **F1-Score** | Mais robusto que acurácia isolada em problemas desbalanceados |
| Reprodutibilidade | `random_state=42` fixo em todas as etapas | Garante resultados idênticos a cada execução |

## 📏 Critérios de avaliação (competição)

| Critério | Peso |
|---|---|
| Pipeline executa sem erro | 20% |
| Tratamento correto de nulos e categóricas | 20% |
| Métricas corretas (não só acurácia) | 25% |
| Comparação entre 2 modelos | 15% |
| Organização/legibilidade do código | 10% |
| Modelo salvo corretamente | 10% |

## 🔮 Possíveis extensões

- [ ] Balanceamento de classes (`class_weight="balanced"` ou SMOTE)
- [ ] Validação cruzada (`cross_val_score`)
- [ ] Otimização de hiperparâmetros (`GridSearchCV`)
- [ ] One-Hot Encoding para variáveis sem ordem natural
- [ ] Empacotar a GUI como executável (`.exe`) com PyInstaller

## 🐞 Solução de problemas

| Erro | Causa | Solução |
|---|---|---|
| `FileNotFoundError: telco_churn.csv` | Dataset não foi gerado | Rode `python generate_dataset.py` primeiro |
| `ModuleNotFoundError` | Dependência ausente | Rode `pip install pandas scikit-learn joblib` |
| `TypeError: numpy string dtypes...` | Versão de Pandas incompatível com `include=["object","str"]` | Use `include=["object"]` no `select_dtypes` |
| `python: command not found` | Python não está no PATH | Tente `python3` no lugar de `python` |

## 📄 Licença

Este projeto está sob a licença MIT — sinta-se livre para usar, estudar e adaptar.

## 👤 Autor

Desenvolvido para o desafio **CodeQuest Arena — DataPython**.
