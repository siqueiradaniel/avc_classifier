import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
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
from scipy.stats import shapiro
from imblearn.pipeline import Pipeline as ImbPipeline

from sklearn.base import BaseEstimator, TransformerMixin
from scipy.stats.mstats import mquantiles

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier

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


# --- Checagem de normalidade nas variáveis numéricas ---
variaveis_numericas = ['age', 'avg_glucose_level', 'bmi']

print("\nTeste de normalidade (Shapiro-Wilk):")
for col in variaveis_numericas:
    stat, p = shapiro(df[col].dropna())
    print(f"{col}: W = {stat:.4f}, p = {p:.4f} => {'Normal' if p > 0.05 else 'Não normal'}")
print("-" * 50)


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

# --- 6. VALIDAÇÃO CRUZADA + AJUSTE DE HIPERPARÂMETROS ---
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

class Winsorizer(BaseEstimator, TransformerMixin):
    def __init__(self, lower=0.01, upper=0.01, variables=None):
        self.lower = lower
        self.upper = upper
        self.variables = variables
        self.limits_ = {}

    def fit(self, X, y=None):
        self.limits_ = {
            col: mquantiles(X[col], prob=[self.lower, 1 - self.upper])
            for col in self.variables
        }
        return self

    def transform(self, X):
        X_copy = X.copy()
        for col in self.variables:
            low, high = self.limits_[col]
            X_copy[col] = np.clip(X_copy[col], low, high)
        return X_copy

from sklearn.model_selection import cross_validate

scoring_metrics = {
    'balanced_accuracy': 'balanced_accuracy',
    # 'precision': 'precision',
    # 'recall': 'recall',
    # 'f1': 'f1',
    # 'roc_auc': 'roc_auc',
    # 'average_precision': 'average_precision'
}

def avaliar_modelo_cv(modelo, X, y, nome_modelo):
    print(f"\nAvaliação final por cross-validation - {nome_modelo}:")
    resultados = cross_validate(
        modelo, X, y,
        cv=cv,
        scoring=scoring_metrics,
        return_train_score=False,
        n_jobs=-1
    )
    for metrica in scoring_metrics:
        valores = resultados[f'test_{metrica}']
        print(f"{metrica}: {np.mean(valores):.4f} ± {np.std(valores):.4f}")


# Grid para Regressão Logística
logreg_pipeline = ImbPipeline(steps=[
    ('winsor', Winsorizer(variables=variaveis_numericas)),
    ('scaler', RobustScaler()),
    ('smote', SMOTE(random_state=RANDOM_STATE)),
    ('clf', LogisticRegression(random_state=RANDOM_STATE))
])

logreg_grid = GridSearchCV(
    estimator=logreg_pipeline,
    param_grid={
        'clf__C': [0.01, 0.1, 1, 10, 100],
        'clf__penalty': ['l2'],
        'clf__solver': ['liblinear']
    },
    cv=cv,
    scoring='balanced_accuracy',
    n_jobs=-1,
    verbose=1
)

print("Executando GridSearchCV (Logistic Regression)...")
logreg_grid.fit(X_train, y_train)
best_logreg = logreg_grid.best_estimator_
print(f"Melhores hiperparâmetros (LogReg): {logreg_grid.best_params_}")
avaliar_modelo_cv(best_logreg, X_train, y_train, "Logistic Regression")
print("-" * 50)

# Grid para KNN
knn_pipeline = ImbPipeline(steps=[
    ('winsor', Winsorizer(variables=variaveis_numericas)),
    ('scaler', RobustScaler()),
    ('smote', SMOTE(random_state=RANDOM_STATE)),
    ('clf', KNeighborsClassifier())
])

knn_grid = GridSearchCV(
    estimator=knn_pipeline,
    param_grid={
        'clf__n_neighbors': range(3, 15, 2),
        'clf__weights': ['uniform', 'distance'],
        'clf__metric': ['euclidean', 'manhattan']
    },
    cv=cv,
    scoring='balanced_accuracy',
    n_jobs=-1,
    verbose=1
)

print("Executando GridSearchCV (KNN)...")
knn_grid.fit(X_train, y_train)
best_knn = knn_grid.best_estimator_
print(f"Melhores hiperparâmetros (KNN): {knn_grid.best_params_}")
avaliar_modelo_cv(best_knn, X_train, y_train, "KNN")
print("-" * 50)

# Grid para Árvore de Decisão
dt_pipeline = ImbPipeline(steps=[
    ('winsor', Winsorizer(variables=variaveis_numericas)),
    ('smote', SMOTE(random_state=RANDOM_STATE)),
    ('clf', DecisionTreeClassifier(random_state=RANDOM_STATE))
])

dt_grid = GridSearchCV(
    estimator=dt_pipeline,
    param_grid={
        'clf__max_depth': [3, 5, 10, 15, 20, None],
        'clf__min_samples_split': [2, 5, 10],
        'clf__min_samples_leaf': [1, 2, 4]
    },
    cv=cv,
    scoring='balanced_accuracy',
    n_jobs=-1,
    verbose=1
)

print("Executando GridSearchCV (Decision Tree)...")
dt_grid.fit(X_train, y_train)
best_dt = dt_grid.best_estimator_
print(f"Melhores hiperparâmetros (Decision Tree): {dt_grid.best_params_}")
avaliar_modelo_cv(best_dt, X_train, y_train, "Decision Tree")
print("-" * 50)


# Pipeline SVM com Winsorizer, SMOTE e Escalonamento
'''
svm_pipeline = ImbPipeline(steps=[
    ('winsor', Winsorizer(variables=variaveis_numericas)),
    ('smote', SMOTE(random_state=RANDOM_STATE)),
    ('scaler', RobustScaler()),
    ('clf', SVC(random_state=RANDOM_STATE, probability=True))
])

# Espaço de busca dos hiperparâmetros
svm_params = {
    'clf__C': [0.01, 0.1, 1, 10, 100],
    'clf__kernel': ['linear', 'rbf'],
    'clf__gamma': ['scale', 'auto']
}

# GridSearch com validação cruzada estratificada
svm_grid = GridSearchCV(
    estimator=svm_pipeline,
    param_grid=svm_params,
    cv=cv,
    scoring='balanced_accuracy',
    n_jobs=-1,
    verbose=1
)

# Treinamento e avaliação
print("Executando GridSearchCV (SVM)...")
svm_grid.fit(X_train, y_train)
best_svm = svm_grid.best_estimator_
print(f"Melhores hiperparâmetros (SVM): {svm_grid.best_params_}")
avaliar_modelo_cv(best_svm, X_train, y_train, "SVM")
print("-" * 50)
'''

# Naive Bayes (não precisa de ajuste de hiperparâmetros)
nb_pipeline = ImbPipeline(steps=[
    ('winsor', Winsorizer(variables=variaveis_numericas)),
    ('scaler', RobustScaler()),  # opcional, pode melhorar Naive Bayes
    ('smote', SMOTE(random_state=RANDOM_STATE)),
    ('clf', GaussianNB())
])

print("Avaliando Naive Bayes com cross-validation...")
avaliar_modelo_cv(nb_pipeline, X_train, y_train, "Naive Bayes")
print("-" * 50)

# Treina o pipeline completo no conjunto inteiro para avaliação final em teste
nb_pipeline.fit(X_train, y_train)


# Random Forest
rf_pipeline = ImbPipeline(steps=[
    ('winsor', Winsorizer(variables=variaveis_numericas)),
    ('scaler', RobustScaler()),
    ('smote', SMOTE(random_state=RANDOM_STATE)),
    ('clf', RandomForestClassifier(random_state=RANDOM_STATE))
])

rf_grid = GridSearchCV(
    estimator=rf_pipeline,
    param_grid={
        'clf__n_estimators': [100, 200],
        'clf__max_depth': [None, 10, 20],
        'clf__min_samples_split': [2, 5]
    },
    cv=cv,
    scoring='balanced_accuracy',
    n_jobs=-1,
    verbose=1
)

print("Executando GridSearchCV (Random Forest)...")
rf_grid.fit(X_train, y_train)
best_rf = rf_grid.best_estimator_
print(f"Melhores hiperparâmetros (Random Forest): {rf_grid.best_params_}")
avaliar_modelo_cv(best_rf, X_train, y_train, "Random Forest")
print("-" * 50)



# AdaBoost
ada_pipeline = ImbPipeline(steps=[
    ('winsor', Winsorizer(variables=variaveis_numericas)),
    ('scaler', RobustScaler()),
    ('smote', SMOTE(random_state=RANDOM_STATE)),
    ('clf', AdaBoostClassifier(random_state=RANDOM_STATE))
])

ada_grid = GridSearchCV(
    estimator=ada_pipeline,
    param_grid={
        'clf__n_estimators': [50, 100],
        'clf__learning_rate': [0.5, 1.0, 1.5]
    },
    cv=cv,
    scoring='balanced_accuracy',
    n_jobs=-1,
    verbose=1
)

print("Executando GridSearchCV (AdaBoost)...")
ada_grid.fit(X_train, y_train)
best_ada = ada_grid.best_estimator_
print(f"Melhores hiperparâmetros (AdaBoost): {ada_grid.best_params_}")
avaliar_modelo_cv(best_ada, X_train, y_train, "AdaBoost")
print("-" * 50)


# XGBoost
xgb_pipeline = ImbPipeline(steps=[
    ('winsor', Winsorizer(variables=variaveis_numericas)),
    ('scaler', RobustScaler()),
    ('smote', SMOTE(random_state=RANDOM_STATE)),
    ('clf', XGBClassifier(random_state=RANDOM_STATE, use_label_encoder=False, eval_metric='logloss'))
])

xgb_grid = GridSearchCV(
    estimator=xgb_pipeline,
    param_grid={
        'clf__n_estimators': [100, 200],
        'clf__max_depth': [3, 6],
        'clf__learning_rate': [0.01, 0.1, 0.2]
    },
    cv=cv,
    scoring='balanced_accuracy',
    n_jobs=-1,
    verbose=1
)

print("Executando GridSearchCV (XGBoost)...")
xgb_grid.fit(X_train, y_train)
best_xgb = xgb_grid.best_estimator_
print(f"Melhores hiperparâmetros (XGBoost): {xgb_grid.best_params_}")
avaliar_modelo_cv(best_xgb, X_train, y_train, "XGBoost")
print("-" * 50)


# --- 7. AVALIAÇÃO FINAL NO CONJUNTO DE TESTE ---
modelos = {
    "Logistic Regression": best_logreg,
    "KNN": best_knn,
    "Decision Tree": best_dt,
    #"SVM": best_svm,
    "Naive Bayes": nb_pipeline,
    "Random Florest": best_rf,
    "AdaBoost": best_ada,
    "XGBoost": best_xgb,
}

print("Comparando modelos no conjunto de teste...\n")

resultados = []

for nome, modelo in modelos.items():
    y_pred = modelo.predict(X_test)
    # Alguns modelos (NaiveBayes) podem não ter predict_proba, mas GaussianNB tem
    y_proba = modelo.predict_proba(X_test)[:, 1] if hasattr(modelo, "predict_proba") else None

    auc_roc = roc_auc_score(y_test, y_proba) if y_proba is not None else float('nan')
    avg_precision = average_precision_score(y_test, y_proba) if y_proba is not None else float('nan')

    print(f"--- {nome} ---")
    print(f"Acurácia: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Balanced Accuracy: {balanced_accuracy_score(y_test, y_pred):.4f}")
    # if y_proba is not None:
    #     print(f"ROC AUC Score: {auc_roc:.4f}")
    #     print(f"Average Precision (PR AUC): {avg_precision:.4f}")
    print("\nMatriz de Confusão:")
    print(confusion_matrix(y_test, y_pred))
    #print("\nRelatório de Classificação:")
    #print(classification_report(y_test, y_pred))
    print("-" * 50)

# --- 8. CURVAS DE AVALIAÇÃO VISUAL para o melhor modelo (exemplo LogReg) ---
best_model_name = "Logistic Regression"
best_model = best_logreg

y_proba = best_model.predict_proba(X_test)[:, 1]
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
# print("Importância das features (coeficientes) - Logistic Regression:")
# feature_importance = pd.Series(best_logreg.coef_[0], index=X.columns)
# feature_importance = feature_importance.sort_values(key=abs, ascending=False)
# print(feature_importance.to_string())
# print("-" * 50)
