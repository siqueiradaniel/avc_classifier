import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
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

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
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


def treinar_modelo_com_pipeline(
    nome_modelo,
    classificador,
    param_grid=None,
    usar_scaler=True,
    usar_winsor=True,
    usar_smote=True,
    ajustar=True,
    X_train=None,
    y_train=None,
    cv=None
):
    steps = []

    if usar_winsor:
        steps.append(('winsor', Winsorizer(variables=variaveis_numericas)))

    if usar_scaler:
        steps.append(('scaler', RobustScaler()))

    if usar_smote:
        steps.append(('smote', SMOTE(random_state=RANDOM_STATE)))

    steps.append(('clf', classificador))

    pipeline = ImbPipeline(steps=steps)

    if ajustar and param_grid is not None:
        print(f"Executando GridSearchCV ({nome_modelo})...")
        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=cv,
            scoring='balanced_accuracy',
            n_jobs=-1,
            verbose=1
        )
        grid.fit(X_train, y_train)
        melhor_modelo = grid.best_estimator_
        print(f"Melhores hiperparâmetros ({nome_modelo}): {grid.best_params_}")
    else:
        print(f"Avaliando {nome_modelo} sem grid search...")
        pipeline.fit(X_train, y_train)
        melhor_modelo = pipeline

    avaliar_modelo_cv(melhor_modelo, X_train, y_train, nome_modelo)
    print("-" * 50)
    return melhor_modelo

# Grid para Classificador Ingenuo
modelo_dummy = treinar_modelo_com_pipeline(
    nome_modelo="Classificador Ingenuo",
    classificador=DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE),
    usar_scaler=False,
    usar_winsor=False,
    usar_smote=False,
    ajustar=False,
    X_train=X_train,
    y_train=y_train,
    cv=cv
)

# Grid para Regressão Logística
logreg_params = {
    'clf__C': [0.01, 0.1, 1, 10, 100],
    'clf__penalty': ['l2'],
    'clf__solver': ['liblinear']
}

modelo_logreg = treinar_modelo_com_pipeline(
    nome_modelo="Logistic Regression",
    classificador=LogisticRegression(random_state=RANDOM_STATE),
    param_grid=logreg_params,
    usar_scaler=True,
    usar_winsor=True,
    usar_smote=True,
    ajustar=True,
    X_train=X_train,
    y_train=y_train,
    cv=cv
)

# Grid para KNN
knn_params = {
    'clf__n_neighbors': range(3, 15, 2),
    'clf__weights': ['uniform', 'distance'],
    'clf__metric': ['euclidean', 'manhattan']
}

modelo_knn = treinar_modelo_com_pipeline(
    nome_modelo="KNN",
    classificador=KNeighborsClassifier(),
    param_grid=knn_params,
    usar_scaler=True,
    usar_winsor=True,
    usar_smote=True,
    ajustar=True,
    X_train=X_train,
    y_train=y_train,
    cv=cv
)

# Grid para Árvore de Decisão
dt_params = {
    'clf__max_depth': [3, 5, 10, 15, 20, None],
    'clf__min_samples_split': [2, 5, 10],
    'clf__min_samples_leaf': [1, 2, 4]
}

modelo_dt = treinar_modelo_com_pipeline(
    nome_modelo="Decision Tree",
    classificador=DecisionTreeClassifier(random_state=RANDOM_STATE),
    param_grid=dt_params,
    usar_scaler=False,  # árvores não precisam de scaler
    usar_winsor=True,
    usar_smote=True,
    ajustar=True,
    X_train=X_train,
    y_train=y_train,
    cv=cv
)

# Pipeline SVM com Winsorizer, SMOTE e Escalonamento
svm_params = {
    'clf__C': [0.01, 0.1, 1, 10, 100],
    'clf__kernel': ['linear', 'rbf'],
    'clf__gamma': ['scale', 'auto']
}

modelo_svm = treinar_modelo_com_pipeline(
    nome_modelo="SVM",
    classificador=SVC(
        random_state=RANDOM_STATE, 
        # probability=True 
    ),
    param_grid=svm_params,
    usar_scaler=True,
    usar_winsor=True,
    usar_smote=True,
    ajustar=True,
    X_train=X_train,
    y_train=y_train,
    cv=cv
)

# Naive Bayes (não precisa de ajuste de hiperparâmetros)
modelo_nb = treinar_modelo_com_pipeline(
    nome_modelo="Naive Bayes",
    classificador=GaussianNB(),
    ajustar=False,
    usar_scaler=True,
    usar_winsor=True,
    usar_smote=True,
    X_train=X_train,
    y_train=y_train,
    cv=cv
)

# Random Forest (não é sensível a outliers nem a escalas)
rf_params = {
    'clf__n_estimators': [100, 200],
    'clf__max_depth': [None, 5, 10, 15, 20, 30],
    'clf__min_samples_split': [2, 5]
}

modelo_rf = treinar_modelo_com_pipeline(
    nome_modelo="Random Forest",
    classificador=RandomForestClassifier(
        random_state=RANDOM_STATE,
        class_weight='balanced'
    ),
    param_grid=rf_params,
    usar_scaler=False,     # não precisa para árvores
    usar_winsor=False,     # idem
    usar_smote=True,
    ajustar=True,
    X_train=X_train,
    y_train=y_train,
    cv=cv
)

# AdaBoost
ada_params = {
    'clf__n_estimators': [50, 100],
    'clf__learning_rate': [0.5, 1.0, 1.5]
}

modelo_ada = treinar_modelo_com_pipeline(
    nome_modelo="AdaBoost",
    classificador=AdaBoostClassifier(random_state=RANDOM_STATE),
    param_grid=ada_params,
    usar_scaler=True,
    usar_winsor=True,
    usar_smote=True,
    ajustar=True,
    X_train=X_train,
    y_train=y_train,
    cv=cv
)

# XGBoost
xgb_params = {
    'clf__n_estimators': [100, 200],
    'clf__max_depth': [3, 6],
    'clf__learning_rate': [0.01, 0.1, 0.2]
}

modelo_xgb = treinar_modelo_com_pipeline(
    nome_modelo="XGBoost",
    classificador=XGBClassifier(
        random_state=RANDOM_STATE,
        eval_metric='logloss'
    ),
    param_grid=xgb_params,
    usar_scaler=True,
    usar_winsor=True,
    usar_smote=True,
    ajustar=True,
    X_train=X_train,
    y_train=y_train,
    cv=cv
)

# --- 7. AVALIAÇÃO FINAL NO CONJUNTO DE TESTE ---
modelos = {
    "Classificador Ingenuo": modelo_dummy,
    "Logistic Regression": modelo_logreg,
    "KNN": modelo_knn,
    "Decision Tree": modelo_dt,
    "SVM": modelo_svm,
    "Naive Bayes": modelo_nb,
    "Random Forest": modelo_rf,
    "AdaBoost": modelo_ada,
    "XGBoost": modelo_xgb,
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
best_model = modelo_logreg

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
# feature_importance = pd.Series(modelo_logred.coef_[0], index=X.columns)
# feature_importance = feature_importance.sort_values(key=abs, ascending=False)
# print(feature_importance.to_string())
# print("-" * 50)
