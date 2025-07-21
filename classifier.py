import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score, balanced_accuracy_score
)
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")  # Para evitar warnings do GridSearch

# --- FIXAR SEMENTES ---
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# --- 1. CARREGAR O DATASET ---
file_path = 'datasets/clean/healthcare-dataset-stroke-data-PROCESSADO.csv'

try:
    df = pd.read_csv(file_path)
    print("Dataset processado carregado com sucesso.")
    print("-" * 50)
except FileNotFoundError:
    print(f"ERRO: O arquivo '{file_path}' não foi encontrado.")
    exit()

# --- 2. SEPARAR FEATURES E ALVO ---
X = df.drop('stroke', axis=1)
y = df['stroke']

print(f"Formato de X: {X.shape}")
print(f"Formato de y: {y.shape}")
print("-" * 50)

# --- 3. DIVISÃO EM TREINO/TESTE ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

print("Conjuntos de treino e teste separados.")
print(f"Tamanho do treino: {len(X_train)}")
print(f"Tamanho do teste: {len(X_test)}")
print("-" * 50)

# --- 4. ESCALONAMENTO ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 5. SMOTE APENAS NO TREINO ---
print(f"Antes do SMOTE:\n{y_train.value_counts()}")
smote = SMOTE(random_state=RANDOM_STATE)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
print(f"Após o SMOTE:\n{pd.Series(y_train_resampled).value_counts()}")
print("-" * 50)

# --- 6. VALIDAÇÃO CRUZADA + AJUSTE DE HIPERPARÂMETROS ---
param_grid = {
    'C': [0.01, 0.1, 1, 10],
    'penalty': ['l2'],
    'solver': ['liblinear']
}

model = LogisticRegression(random_state=RANDOM_STATE)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=cv,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)

print("Executando GridSearchCV com validação cruzada...")
grid_search.fit(X_train_resampled, y_train_resampled)

best_model = grid_search.best_estimator_
print(f"Melhores hiperparâmetros encontrados: {grid_search.best_params_}")
print("-" * 50)

# --- 7. AVALIAÇÃO FINAL NO CONJUNTO DE TESTE ---
print("Avaliando o modelo final no conjunto de teste...")
y_pred = best_model.predict(X_test_scaled)
y_proba = best_model.predict_proba(X_test_scaled)[:, 1]  # Probabilidades da classe 1

print(f"Acurácia: {accuracy_score(y_test, y_pred):.4f}")
print(f"Balanced Accuracy: {balanced_accuracy_score(y_test, y_pred):.4f}")
print(f"ROC AUC Score: {roc_auc_score(y_test, y_proba):.4f}")
print(f"Average Precision (PR AUC): {average_precision_score(y_test, y_proba):.4f}")
print("\nMatriz de Confusão:")
print(confusion_matrix(y_test, y_pred))
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))
print("-" * 50)

# --- 8. CURVAS DE AVALIAÇÃO VISUAL ---
# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label='ROC Curve')
plt.plot([0, 1], [0, 1], '--', color='gray')
plt.title("ROC Curve")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.legend()
plt.tight_layout()
plt.savefig("roc_curve.png")

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_proba)
plt.figure(figsize=(7, 5))
plt.plot(recall, precision, label='Precision-Recall Curve', color='green')
plt.title("Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.tight_layout()
plt.savefig("precision_recall_curve.png")

# --- 9. IMPORTÂNCIA DAS FEATURES ---
print("Importância das features (coeficientes):")
feature_importance = pd.Series(best_model.coef_[0], index=X.columns)
feature_importance = feature_importance.sort_values(key=abs, ascending=False)
print(feature_importance.to_string())
print("-" * 50)
