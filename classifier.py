import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE

# --- 1. CARREGAR O DATASET PROCESSADO ---
# Usamos o arquivo final gerado pelo script de limpeza.
file_path = 'datasets/clean/healthcare-dataset-stroke-data-PROCESSADO.csv'

try:
    df = pd.read_csv(file_path)
    print("Dataset processado carregado com sucesso.")
    print("-" * 50)
except FileNotFoundError:
    print(f"ERRO: O arquivo '{file_path}' não foi encontrado. Execute o script de limpeza primeiro.")
    exit()

# --- 2. SEPARAR FEATURES (X) E ALVO (y) ---
X = df.drop('stroke', axis=1)
y = df['stroke']
print("Features (X) e Alvo (y) separados.")
print(f"Formato de X: {X.shape}")
print(f"Formato de y: {y.shape}")
print("-" * 50)

# --- 3. DIVIDIR EM CONJUNTOS DE TREINO E TESTE ---
# A divisão é o PRIMEIRO passo antes de qualquer pré-processamento que aprenda com os dados.
# stratify=y garante que a proporção de casos de AVC seja a mesma nos conjuntos de treino e teste.
# random_state=42 garante que a divisão seja a mesma toda vez que rodarmos o código.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,    # 20% dos dados para teste
    random_state=42,  # Para reprodutibilidade
    stratify=y        # Essencial para dados desbalanceados
)
print("Dados divididos em treino e teste.")
print(f"Tamanho do treino: {len(X_train)} amostras")
print(f"Tamanho do teste: {len(X_test)} amostras")
print("-" * 50)

# --- 4. ESCALONAMENTO DE FEATURES ---
# O scaler é "treinado" (fit) APENAS com os dados de treino para evitar data leakage.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test) # Apenas transformamos os dados de teste.
print("Features escalonadas.")
print("-" * 50)

# --- 5. BALANCEAMENTO DE CLASSES COM SMOTE ---
# O SMOTE é aplicado APENAS nos dados de treino. O teste deve refletir a realidade.
print(f"Contagem de classes antes do SMOTE:\n{y_train.value_counts()}")
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
print(f"\nContagem de classes depois do SMOTE:\n{pd.Series(y_train_resampled).value_counts()}")
print("-" * 50)

# --- 6. TREINAMENTO DO MODELO DE CLASSIFICAÇÃO ---
# Usaremos a Regressão Logística como um primeiro modelo de base.
print("Treinando o modelo de Regressão Logística...")
model = LogisticRegression(random_state=42)
model.fit(X_train_resampled, y_train_resampled)
print("Modelo treinado.")
print("-" * 50)

# --- 7. AVALIAÇÃO DO MODELO ---
print("Avaliando o modelo no conjunto de teste...")
# Fazendo previsões nos dados de teste (que nunca foram vistos ou usados no treino)
y_pred = model.predict(X_test_scaled)

# Métricas de avaliação
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Acurácia no Teste: {accuracy:.4f}\n")
print("Matriz de Confusão:")
print(conf_matrix)
print("\nRelatório de Classificação:")
print(class_report)
print("-" * 50)
