import pandas as pd
import numpy as np

# --- 1. CONFIGURAÇÃO DOS ARQUIVOS ---
# O único arquivo de entrada é o original, sem modificações.
input_file = 'datasets/healthcare-dataset-stroke-data.csv'
# O arquivo de saída conterá o dataset final e totalmente processado.
output_file = 'datasets/clean/healthcare-dataset-stroke-data-PROCESSADO.csv'

print("Iniciando o pipeline de processamento de dados...")
print("-" * 50)

try:
    # --- 2. CARREGAMENTO E LIMPEZA INICIAL ---
    df = pd.read_csv(input_file)
    print(f"Dataset '{input_file}' carregado. Linhas: {len(df)}")

    # Remover a coluna 'id' que não será usada no modelo.
    df.drop(columns=['id'], inplace=True)

    # A coluna 'gender' possui um único registro com valor 'Other'.
    # Vamos removê-lo para trabalhar apenas com 'Male' e 'Female'.
    df = df[df['gender'] != 'Other']
    print(f"Removida 1 linha com gênero 'Other'. Linhas restantes: {len(df)}")
    print("-" * 50)

    # --- 3. IMPUTAÇÃO DE VALORES AUSENTES EM 'bmi' ---
    print("Iniciando imputação de valores ausentes em 'bmi'...")
    # Criar uma coluna temporária de faixa etária para agrupar
    df['age_group'] = pd.cut(df['age'], bins=range(0, 105, 5), right=False)

    # Usar groupby e transform para preencher NaNs com a mediana do grupo.
    # É uma forma mais eficiente que o loop 'for'.
    df['bmi'] = df.groupby(['gender', 'age_group'])['bmi'].transform(lambda x: x.fillna(x.median()))

    # Se ainda houver algum NaN (caso um grupo inteiro não tenha valor de bmi),
    # preenchemos com a mediana global.
    df['bmi'].fillna(df['bmi'].median(), inplace=True)
    
    df['bmi'] = df['bmi'].round(1)
    
    # Remover a coluna de faixa etária que era apenas auxiliar
    df.drop(columns=['age_group'], inplace=True)
    print("Imputação de 'bmi' concluída.")
    print("-" * 50)

    # --- 4. CODIFICAÇÃO DE VARIÁVEIS CATEGÓRICAS ---
    print("Iniciando codificação de variáveis categóricas...")

    # Mapeamento de variáveis binárias para 0 e 1
    df['gender'] = df['gender'].map({'Male': 0, 'Female': 1})
    df['ever_married'] = df['ever_married'].map({'No': 0, 'Yes': 1})
    df['Residence_type'] = df['Residence_type'].map({'Rural': 0, 'Urban': 1})

    # One-Hot Encoding para variáveis com múltiplas categorias em uma única chamada
    df = pd.get_dummies(df, columns=['work_type', 'smoking_status'], prefix=['work_type', 'smoking_status'], dtype=int)
    
    print("Codificação concluída.")
    print("-" * 50)

    # --- 5. CONVERSÃO FINAL DE TIPOS DE DADOS ---
    print("Convertendo 'age' e 'gender' para o tipo inteiro...")
    # Agora que todas as manipulações foram feitas, convertemos para int
    df['age'] = df['age'].astype(int)
    df['gender'] = df['gender'].astype(int)
    print("Conversão de tipos concluída.")
    print("-" * 50)
    
    # --- 6. VERIFICAÇÃO FINAL ---
    print("Verificação final do DataFrame processado:")
    print("Primeiras 5 linhas:")
    print(df.head())
    print("\nInformações e tipos de dados:")
    df.info()
    print("-" * 50)

    # --- 7. SALVAR O DATASET FINAL ---
    df.to_csv(output_file, index=False)
    print(f"✅ SUCESSO! O pipeline foi concluído.")
    print(f"O dataset final foi salvo como '{output_file}'")

except FileNotFoundError:
    print(f"❌ ERRO: O arquivo de entrada '{input_file}' não foi encontrado.")
except Exception as e:
    print(f"❌ ERRO: Ocorreu um erro inesperado durante o processamento: {e}")