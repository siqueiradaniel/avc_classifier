import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score, balanced_accuracy_score
)
from imblearn.over_sampling import SMOTE
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
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# Grid para Regressão Logística
logreg_params = {
    'C': [0.01, 0.1, 1, 10],
    'penalty': ['l2'],
    'solver': ['liblinear']
}
logreg_model = LogisticRegression(random_state=RANDOM_STATE)
logreg_grid = GridSearchCV(
    estimator=logreg_model,
    param_grid=logreg_params,
    cv=cv,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)

print("Executando GridSearchCV (Logistic Regression)...")
logreg_grid.fit(X_train_resampled, y_train_resampled)
best_logreg = logreg_grid.best_estimator_
print(f"Melhores hiperparâmetros (LogReg): {logreg_grid.best_params_}")
print("-" * 50)

# Grid para KNN
knn_params = {
    'n_neighbors': [3, 5, 7],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}
knn_model = KNeighborsClassifier()
knn_grid = GridSearchCV(
    estimator=knn_model,
    param_grid=knn_params,
    cv=cv,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)

print("Executando GridSearchCV (KNN)...")
knn_grid.fit(X_train_resampled, y_train_resampled)
best_knn = knn_grid.best_estimator_
print(f"Melhores hiperparâmetros (KNN): {knn_grid.best_params_}")
print("-" * 50)

# Grid para Árvore de Decisão
dt_params = {
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
dt_model = DecisionTreeClassifier(random_state=RANDOM_STATE)
dt_grid = GridSearchCV(
    estimator=dt_model,
    param_grid=dt_params,
    cv=cv,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)

print("Executando GridSearchCV (Decision Tree)...")
dt_grid.fit(X_train_resampled, y_train_resampled)
best_dt = dt_grid.best_estimator_
print(f"Melhores hiperparâmetros (Decision Tree): {dt_grid.best_params_}")
print("-" * 50)
'''
# Grid para SVM (com probabilidade ativada para ROC)
svm_params = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}
svm_model = SVC(random_state=RANDOM_STATE, probability=True)
svm_grid = GridSearchCV(
    estimator=svm_model,
    param_grid=svm_params,
    cv=cv,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)

print("Executando GridSearchCV (SVM)...")
svm_grid.fit(X_train_resampled, y_train_resampled)
best_svm = svm_grid.best_estimator_
print(f"Melhores hiperparâmetros (SVM): {svm_grid.best_params_}")
print("-" * 50)
'''
# Naive Bayes (não precisa de ajuste de hiperparâmetros)
nb_model = GaussianNB()
nb_model.fit(X_train_resampled, y_train_resampled)

# --- 7. AVALIAÇÃO FINAL NO CONJUNTO DE TESTE ---
modelos = {
    "Logistic Regression": best_logreg,
    "KNN": best_knn,
    "Decision Tree": best_dt,
    #"SVM": best_svm,
    "Naive Bayes": nb_model
}

print("Comparando modelos no conjunto de teste...\n")

resultados = []

for nome, modelo in modelos.items():
    y_pred = modelo.predict(X_test_scaled)
    # Alguns modelos (NaiveBayes) podem não ter predict_proba, mas GaussianNB tem
    y_proba = modelo.predict_proba(X_test_scaled)[:, 1] if hasattr(modelo, "predict_proba") else None

    auc_roc = roc_auc_score(y_test, y_proba) if y_proba is not None else float('nan')
    avg_precision = average_precision_score(y_test, y_proba) if y_proba is not None else float('nan')

    print(f"--- {nome} ---")
    print(f"Acurácia: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Balanced Accuracy: {balanced_accuracy_score(y_test, y_pred):.4f}")
    if y_proba is not None:
        print(f"ROC AUC Score: {auc_roc:.4f}")
        print(f"Average Precision (PR AUC): {avg_precision:.4f}")
    print("\nMatriz de Confusão:")
    print(confusion_matrix(y_test, y_pred))
    print("\nRelatório de Classificação:")
    print(classification_report(y_test, y_pred))
    print("-" * 50)

# --- 8. CURVAS DE AVALIAÇÃO VISUAL para o melhor modelo (exemplo LogReg) ---
best_model_name = "Logistic Regression"
best_model = best_logreg

y_proba = best_model.predict_proba(X_test_scaled)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label='ROC Curve')
plt.plot([0, 1], [0, 1], '--', color='gray')
plt.title(f"ROC Curve - {best_model_name}")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.legend()
plt.tight_layout()
plt.savefig(f'./results/roc_curve_{best_model_name.lower().replace(" ", "_")}.png')

precision, recall, _ = precision_recall_curve(y_test, y_proba)
plt.figure(figsize=(7, 5))
plt.plot(recall, precision, label='Precision-Recall Curve', color='green')
plt.title(f"Precision-Recall Curve - {best_model_name}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.tight_layout()
plt.savefig(f'./results/precision_recall_curve_{best_model_name.lower().replace(" ", "_")}.png')

# --- 9. IMPORTÂNCIA DAS FEATURES para Logistic Regression ---
print("Importância das features (coeficientes) - Logistic Regression:")
feature_importance = pd.Series(best_logreg.coef_[0], index=X.columns)
feature_importance = feature_importance.sort_values(key=abs, ascending=False)
print(feature_importance.to_string())
print("-" * 50)
