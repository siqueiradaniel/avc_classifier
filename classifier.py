import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import wilcoxon
from itertools import combinations
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    average_precision_score,
    confusion_matrix,
)
from imblearn.over_sampling import SMOTE
from scipy.stats import shapiro
import scipy.stats as st
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
warnings.filterwarnings("ignore")

# Configuração da semente aleatória
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# --- 1. CARREGAR O DATASET ---
file_path = 'datasets/healthcare-dataset-stroke-data.csv'

try:
    df = pd.read_csv(file_path)
    print("Dataset bruto carregado com sucesso.")
    print("-" * 50)
except FileNotFoundError:
    print(f"ERRO: O arquivo '{file_path}' não foi encontrado.")
    exit()

# --- LIMPEZA INICIAL (OPERAÇÕES GLOBAIS) ---
df.drop(columns=['id'], inplace=True)
df = df[df['gender'] != 'Other']
print(f"Limpeza inicial concluída. Linhas restantes: {len(df)}")
print("-" * 50)


# --- Checagem de normalidade nas variáveis numéricas ---
variaveis_numericas = ['age', 'avg_glucose_level', 'bmi']

print("\nTeste de normalidade (Shapiro-Wilk):")
for col in variaveis_numericas:
    stat, p = shapiro(df[col].dropna())
    print(f"{col}: W = {stat:.4f}, p = {p:.4f} => {'Normal' if p > 0.05 else 'Não normal'}")
print("-" * 50)

# Separação de features e variável alvo
X = df.drop('stroke', axis=1)
y = df['stroke']

print(f"Formato de X: {X.shape}")
print(f"Formato de y: {y.shape}")
print("-" * 50)

# Divisão em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

print("Conjuntos de treino e teste separados.")
print(f"Tamanho do treino: {len(X_train)}")
print(f"Tamanho do teste: {len(X_test)}")
print("-" * 50)

# --- 4. PRÉ-PROCESSAMENTO PÓS-DIVISÃO (NOVO) ---
print("Iniciando pré-processamento pós-divisão (Padrão Ouro)...")

# --- 4.1 IMPUTAÇÃO DE 'bmi' ---
# A lógica é aprendida APENAS no X_train e aplicada em ambos

# Criar a coluna auxiliar de faixa etária
for df_temp in [X_train, X_test]:
    df_temp['age_group'] = pd.cut(df_temp['age'], bins=range(0, 105, 5), right=False)

# Aprender as medianas SOMENTE no treino
imputation_map = X_train.groupby(['gender', 'age_group'], observed=True)['bmi'].median()
global_bmi_median_train = X_train['bmi'].median()

# Aplicar em ambos os conjuntos
for df_temp in [X_train, X_test]:
    df_temp['bmi'] = df_temp.groupby(['gender', 'age_group'], observed=True)['bmi'].transform(lambda x: x.fillna(imputation_map.get(x.name, global_bmi_median_train)))
    df_temp['bmi'] = df_temp['bmi'].fillna(global_bmi_median_train)
    df_temp['bmi'] = df_temp['bmi'].round(1)
    df_temp.drop(columns=['age_group'], inplace=True)

print("Imputação de 'bmi' concluída.")

# --- 4.2 CODIFICAÇÃO DE VARIÁVEIS CATEGÓRICAS ---
# Mapeamento de binárias
for df_temp in [X_train, X_test]:
    df_temp['gender'] = df_temp['gender'].map({'Male': 0, 'Female': 1})
    df_temp['ever_married'] = df_temp['ever_married'].map({'No': 0, 'Yes': 1})
    df_temp['Residence_type'] = df_temp['Residence_type'].map({'Rural': 0, 'Urban': 1})

# One-Hot Encoding - Garantindo que ambos os conjuntos tenham as mesmas colunas
# pd.get_dummies é mais simples aqui que o OneHotEncoder e podemos garantir a consistência
# alinhando os dataframes após a codificação.
X_train = pd.get_dummies(X_train, columns=['work_type', 'smoking_status'], prefix=['work_type', 'smoking_status'], dtype=int)
X_test = pd.get_dummies(X_test, columns=['work_type', 'smoking_status'], prefix=['work_type', 'smoking_status'], dtype=int)

# Alinhar colunas para garantir que o teste tenha as mesmas colunas que o treino
train_cols = X_train.columns
test_cols = X_test.columns

missing_in_test = set(train_cols) - set(test_cols)
for c in missing_in_test:
    X_test[c] = 0

missing_in_train = set(test_cols) - set(train_cols)
for c in missing_in_train:
    X_train[c] = 0

X_test = X_test[train_cols] # Garante a mesma ordem de colunas

print("Codificação de variáveis categóricas concluída.")

# --- 4.3 CONVERSÃO FINAL DE TIPOS ---
for df_temp in [X_train, X_test]:
    df_temp['age'] = df_temp['age'].astype(int)
    df_temp['gender'] = df_temp['gender'].astype(int)

print("Conversão de tipos concluída.")
print("-" * 50)


# --- 6. VALIDAÇÃO CRUZADA + AJUSTE DE HIPERPARÂMETROS ---
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

# Configuração das métricas de avaliação
scoring_metrics = {
    'balanced_accuracy': 'balanced_accuracy',
    'average_precision': 'average_precision',
}

metricas_sklearn = [
    balanced_accuracy_score,
    average_precision_score
]

# Função para construir pipelines de pré-processamento
def construir_pipeline(
    classificador,
    usar_scaler=True,
    usar_winsor=True,
    usar_smote=True
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
    return pipeline

# Função principal para avaliação com Nested Cross-Validation
def avaliar_modelo_cv_nested(
    nome_modelo,
    classificador,
    param_grid=None,
    X_train=None, y_train=None,
    X_test=None, y_test=None,
    usar_scaler=True,
    usar_winsor=True,
    usar_smote=True,
    ajustar=True
):
    print(f"\nAvaliação com Nested Cross-Validation - {nome_modelo}:")

    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    if ajustar and param_grid:
        inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

        pipeline = construir_pipeline(
            classificador,
            usar_scaler=usar_scaler,
            usar_winsor=usar_winsor,
            usar_smote=usar_smote
        )

        modelo_cv = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=inner_cv,
            scoring='balanced_accuracy',
            n_jobs=-1
        )
    else:
        modelo_cv = construir_pipeline(
            classificador,
            usar_scaler=usar_scaler,
            usar_winsor=usar_winsor,
            usar_smote=usar_smote
        )

    resultados = cross_validate(
        modelo_cv, X_train, y_train,
        cv=outer_cv,
        scoring=scoring_metrics,
        return_train_score=False,
        n_jobs=-1
    )

    print("\nMÉTRICAS DE VALIDAÇÃO (Nested CV):")
    for metrica in scoring_metrics:
        valores = resultados[f'test_{metrica}']
        media = np.mean(valores)
        desvio = np.std(valores, ddof=1)
        n = len(valores)
        z = st.t.ppf(0.975, df=n-1)
        erro = z * (desvio / np.sqrt(n))
        print(f"{metrica}: {media:.4f} ± {erro:.4f} (IC 95%)")

    if ajustar and param_grid:
        modelo_cv.fit(X_train, y_train)
        modelo_final = modelo_cv.best_estimator_
    else:
        modelo_final = modelo_cv
        modelo_final.fit(X_train, y_train)

    y_pred = modelo_final.predict(X_test)
    print("\nMÉTRICAS NO CONJUNTO DE TESTE:")
    for nome, func in zip(scoring_metrics, metricas_sklearn):
        valor = func(y_test, y_pred)
        print(f"{nome}: {valor:.4f}")

    print("\nMatriz de Confusão:")
    print(confusion_matrix(y_test, y_pred))

    print("-" * 50)

    return modelo_final, resultados

# Avaliação dos modelos de classificação

# Classificador Ingênuo (baseline)
dummy_model, dummy_results = avaliar_modelo_cv_nested(
    nome_modelo="Classificador Ingênuo",
    classificador=DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE),
    param_grid=None,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    ajustar=False,
    usar_scaler=False,
    usar_winsor=False,
    usar_smote=False,
)

# Regressão Logística
logreg_params = {
    'clf__C': [0.01, 0.1, 1, 10, 100],
    'clf__penalty': ['l2'],
    'clf__solver': ['liblinear']
}

logreg_model, logreg_results = avaliar_modelo_cv_nested(
    nome_modelo="Logistic Regression",
    classificador=LogisticRegression(random_state=RANDOM_STATE),
    param_grid=logreg_params,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
)

# K-Nearest Neighbors
knn_params = {
    'clf__n_neighbors': range(3, 15, 2),
    'clf__weights': ['uniform', 'distance'],
    'clf__metric': ['euclidean', 'manhattan']
}

knn_model, knn_results = avaliar_modelo_cv_nested(
    nome_modelo="KNN",
    classificador=KNeighborsClassifier(),
    param_grid=knn_params,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
)

# Árvore de Decisão
dt_params = {
    'clf__max_depth': [3, 5, 10, 15, 20, None],
    'clf__min_samples_split': [2, 5, 10],
    'clf__min_samples_leaf': [1, 2, 4]
}

dt_model, dt_results = avaliar_modelo_cv_nested(
    nome_modelo="Decision Tree",
    classificador=DecisionTreeClassifier(random_state=RANDOM_STATE),
    param_grid=dt_params,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    usar_scaler=False,
)

# Support Vector Machine
svm_params = {
    'clf__C': [0.01, 0.1, 1, 10, 100],
    'clf__kernel': ['linear', 'rbf'],
    'clf__gamma': ['scale', 'auto']
}

svm_model, svm_results = avaliar_modelo_cv_nested(
    nome_modelo="SVM",
    classificador=SVC(
        random_state=RANDOM_STATE,
        # probability=True 
    ),
    param_grid=svm_params,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
)

# Naive Bayes
nb_model, nb_results = avaliar_modelo_cv_nested(
    nome_modelo="Naive Bayes",
    classificador=GaussianNB(),
    ajustar=False,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test
)

# Random Forest
rf_params = {
    'clf__n_estimators': [100, 200],
    'clf__max_depth': [None, 5, 10, 15, 20, 30],
    'clf__min_samples_split': [2, 5]
}

rf_model, rf_results = avaliar_modelo_cv_nested(
    nome_modelo="Random Forest",
    classificador=RandomForestClassifier(
        random_state=RANDOM_STATE,
        class_weight='balanced'
    ),
    param_grid=rf_params,
    usar_scaler=False,
    usar_winsor=False,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test
)

# AdaBoost
ada_params = {
    'clf__n_estimators': [50, 100],
    'clf__learning_rate': [0.5, 1.0, 1.5]
}

ada_model, ada_results = avaliar_modelo_cv_nested(
    nome_modelo="AdaBoost",
    classificador=AdaBoostClassifier(random_state=RANDOM_STATE),
    param_grid=ada_params,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test
)

# XGBoost
xgb_params = {
    'clf__n_estimators': [100, 200],
    'clf__max_depth': [3, 6],
    'clf__learning_rate': [0.01, 0.1, 0.2]
}

xgb_model, xgb_results = avaliar_modelo_cv_nested(
    nome_modelo="XGBoost",
    classificador=XGBClassifier(
        random_state=RANDOM_STATE,
        eval_metric='logloss'
    ),
    param_grid=xgb_params,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test
)

# Análise comparativa dos resultados

# Criação do DataFrame com resultados de acurácia balanceada
df_plot = pd.DataFrame({
    'Dummy': dummy_results['test_balanced_accuracy'],
    'LogReg': logreg_results['test_balanced_accuracy'],
    'KNN': knn_results['test_balanced_accuracy'],
    'DecisionTree': dt_results['test_balanced_accuracy'],
    'SVM': svm_results['test_balanced_accuracy'],
    'NaiveBayes': nb_results['test_balanced_accuracy'],
    'RandomForest': rf_results['test_balanced_accuracy'],
    'AdaBoost': ada_results['test_balanced_accuracy'],
    'XGBoost': xgb_results['test_balanced_accuracy']
})

# Transformação para formato longo
df_long = df_plot.melt(var_name='Modelo', value_name='Balanced Accuracy')

# Geração do boxplot comparativo
plt.figure(figsize=(12, 6))
sns.boxplot(data=df_long, x='Modelo', y='Balanced Accuracy')
plt.xticks(rotation=45)
plt.title('Boxplot de Acurácia Balanceada por Modelo (Nested CV)')
plt.tight_layout()
plt.savefig('./results/boxplot_todos_modelos.png')
plt.close()

# Teste estatístico de Wilcoxon para comparação entre modelos
model_names = df_plot.columns.tolist()
pvals_matrix = pd.DataFrame(np.ones((len(model_names), len(model_names))), index=model_names, columns=model_names)

for model1, model2 in combinations(model_names, 2):
    stat, p = wilcoxon(df_plot[model1], df_plot[model2])
    pvals_matrix.loc[model1, model2] = p
    pvals_matrix.loc[model2, model1] = p

print("Matriz de p-values do teste Wilcoxon entre modelos:")
print(pvals_matrix)