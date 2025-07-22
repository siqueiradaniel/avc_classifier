import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE

# --- Passos 1 a 4 permanecem inalterados ---
# (Código de carregamento, limpeza, separação X/y e divisão treino-teste)
# --- 1. CARREGAR DADOS BRUTOS ---
input_file = 'datasets/healthcare-dataset-stroke-data.csv'
print("Iniciando o pipeline 'Padrão Ouro' de ponta a ponta...")
print("-" * 50)
try:
    df = pd.read_csv(input_file)
    print(f"Dataset bruto '{input_file}' carregado. Linhas: {len(df)}")
except FileNotFoundError:
    print(f"❌ ERRO: O arquivo de entrada '{input_file}' não foi encontrado.")
    exit()
# --- 2. LIMPEZA INICIAL ---
df.drop(columns=['id'], inplace=True)
df = df[df['gender'] != 'Other']
print(f"Limpeza inicial concluída. Linhas restantes: {len(df)}")
print("-" * 50)
# --- 3. SEPARAR FEATURES E ALVO ---
X = df.drop('stroke', axis=1)
y = df['stroke']
# --- 4. DIVISÃO TREINO-TESTE ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("Dados divididos em treino e teste.")
print("-" * 50)
# --- 5. IMPUTAÇÃO DE 'bmi' ---
print("Iniciando imputação de 'bmi'...")
for df_temp in [X_train, X_test]:
    df_temp['age_group'] = pd.cut(df_temp['age'], bins=range(0, 105, 5), right=False)
imputation_map = X_train.groupby(['gender', 'age_group'], observed=True)['bmi'].median()
global_bmi_median_train = X_train['bmi'].median()
for df_temp in [X_train, X_test]:
    df_temp['bmi'] = df_temp.groupby(['gender', 'age_group'], observed=True)['bmi'].transform(lambda x: x.fillna(imputation_map.get(x.name, global_bmi_median_train)))
    df_temp['bmi'] = df_temp['bmi'].fillna(global_bmi_median_train)
    df_temp['bmi'] = df_temp['bmi'].round(1)
    df_temp.drop(columns=['age_group'], inplace=True)
print("Imputação de 'bmi' concluída.")
print("-" * 50)

# --- PASSO 5.5: TRATAMENTO DE OUTLIERS (NOVO) ---
print("Iniciando tratamento de outliers (Padrão Ouro)...")
cols_to_check = ['avg_glucose_level', 'bmi']

for col in cols_to_check:
    # APRENDER os limites APENAS no treino
    Q1 = X_train[col].quantile(0.25)
    Q3 = X_train[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    print(f"Coluna '{col}': Limite Inferior={lower_bound:.2f}, Limite Superior={upper_bound:.2f}")

    # APLICAR os limites em ambos os conjuntos de dados (treino e teste)
    X_train[col] = np.clip(X_train[col], lower_bound, upper_bound)
    X_test[col] = np.clip(X_test[col], lower_bound, upper_bound)

print("Tratamento de outliers concluído sem vazamento de dados.")
print("-" * 50)

# --- Passos 6 a 11 (Codificação, Escalonamento, SMOTE, Otimização e Avaliação) ---
# O restante do código continua exatamente o mesmo.
# --- 6. CODIFICAÇÃO DE VARIÁVEIS CATEGÓRICAS ---
print("Iniciando codificação de variáveis categóricas...")
for df_temp in [X_train, X_test]:
    df_temp['gender'] = df_temp['gender'].map({'Male': 0, 'Female': 1})
    df_temp['ever_married'] = df_temp['ever_married'].map({'No': 0, 'Yes': 1})
    df_temp['Residence_type'] = df_temp['Residence_type'].map({'Rural': 0, 'Urban': 1})
categorical_cols = ['work_type', 'smoking_status']
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False, dtype=int)
encoder.fit(X_train[categorical_cols])
X_train_encoded = pd.DataFrame(encoder.transform(X_train[categorical_cols]), columns=encoder.get_feature_names_out(), index=X_train.index)
X_test_encoded = pd.DataFrame(encoder.transform(X_test[categorical_cols]), columns=encoder.get_feature_names_out(), index=X_test.index)
X_train = X_train.drop(columns=categorical_cols).join(X_train_encoded)
X_test = X_test.drop(columns=categorical_cols).join(X_test_encoded)
print("Codificação concluída.")
print("-" * 50)
# --- 7. ESCALONAMENTO DE FEATURES ---
print("Iniciando escalonamento de features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Features escalonadas.")
print("-" * 50)
# --- 8. BALANCEAMENTO DE CLASSES COM SMOTE ---
print(f"Contagem de classes no treino antes do SMOTE:\n{y_train.value_counts()}")
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
print(f"\nContagem de classes no treino depois do SMOTE:\n{pd.Series(y_train_resampled).value_counts()}")
print("-" * 50)
# --- 9. OTIMIZAÇÃO DE HIPERPARÂMETROS ---
print("Iniciando otimização de hiperparâmetros para Logistic Regression...")
model = LogisticRegression(random_state=42, max_iter=1000)
param_grid = {'penalty': ['l1', 'l2'], 'C': [0.01, 0.1, 1, 10, 100], 'solver': ['liblinear', 'saga'], 'class_weight': [None, 'balanced']}
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='f1_weighted', cv=5, n_jobs=-1)
grid_search.fit(X_train_resampled, y_train_resampled)
best_model = grid_search.best_estimator_
print("Otimização concluída.")
print(f"Melhores Parâmetros Encontrados: {grid_search.best_params_}")
print("-" * 50)
# --- 10. AVALIAÇÃO DO MELHOR MODELO (COM LIMIAR PADRÃO 0.5) ---
print("Avaliando o MELHOR modelo no conjunto de teste...")
y_pred = best_model.predict(X_test_scaled)
print("--- Resultados com Limiar Padrão de 0.5 ---")
print(classification_report(y_test, y_pred, digits=4))
print("-" * 50)
# --- 11. AJUSTE FINO COM LIMIAR DE DECISÃO ---
print("Iniciando ajuste do Limiar de Decisão...")
y_probs = best_model.predict_proba(X_test_scaled)[:, 1]
# ... (o resto do passo 11 para testar e plotar limiares)
# (O código do passo 11, se você quiser usá-lo, vem aqui)