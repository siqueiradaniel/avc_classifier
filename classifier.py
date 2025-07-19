import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# 1. Ler o dataset
df = pd.read_csv("healthcare-dataset-stroke-data.csv")

# 2. Visualizar primeiras linhas
print("🔍 Primeiras linhas:")
print(df.head())

# 3. Informações básicas
print("\n📄 Info:")
print(df.info())

# 4. Verificar valores ausentes
print("\n🧹 Valores ausentes:")
print(df.isnull().sum())

# 5. Remover 'id' (não é informativo)
df.drop(columns=['id'], inplace=True)
