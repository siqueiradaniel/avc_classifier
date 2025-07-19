import pandas as pd
import numpy as np

# Carregar o dataset
df = pd.read_csv("healthcare-dataset-stroke-data.csv")

# Substituir "N/A" por np.nan para que o pandas reconheça como NaN
df['bmi'] = pd.to_numeric(df['bmi'], errors='coerce')

# Criar uma coluna de faixa etária em intervalos de 5 anos
# Ex: [0-5), [5-10), ..., [95-100)
df['age_group'] = pd.cut(df['age'], bins=range(0, 105, 5), right=False)

# Função para preencher valores ausentes de bmi com a mediana por grupo de gênero e faixa etária
def fill_bmi_by_group(dataframe):
    df_copy = dataframe.copy()
    for gender in df_copy['gender'].unique():
        gender_df = df_copy[df_copy['gender'] == gender]
        for age_bin in df_copy['age_group'].unique():
            mask = (df_copy['gender'] == gender) & (df_copy['age_group'] == age_bin)
            median_bmi = df_copy.loc[mask, 'bmi'].median()
            df_copy.loc[mask & df_copy['bmi'].isna(), 'bmi'] = median_bmi
    return df_copy

# Aplicar a função
df = fill_bmi_by_group(df)

# Verificar se todos os valores NaN foram preenchidos
print("Valores ausentes em bmi após preenchimento:", df['bmi'].isna().sum())

# Arredondar bmi para uma casa decimal
df['bmi'] = df['bmi'].round(1)

# Remover 'age_group' e 'id'
df.drop(columns=['age_group'], inplace=True)
df.drop(columns=['id'], inplace=True)

# (Opcional) Salvar o novo dataset em outro CSV
df.to_csv("healthcare-dataset-stroke-data-preenchido.csv", index=False)
