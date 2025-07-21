import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Ler o dataset
df = pd.read_csv("./datasets/healthcare-dataset-stroke-data.csv")

# Remover a coluna 'id' que não será usada no modelo.
df.drop(columns=['id'], inplace=True)

# Calcular percentual geral de AVC
total_avc = df['stroke'].sum()
total_geral = len(df)
percentual_geral = total_avc / total_geral * 100

# Função padrão para gráficos
def plot_percentual_avc(df, col, title, order=None, bins=None, labels=None, filename=None):
    if bins:
        df[col + '_group'] = pd.cut(df[col], bins=bins, labels=labels, right=False)
        grupo = col + '_group'
    else:
        grupo = col

    avc_por_grupo = df[df['stroke'] == 1][grupo].value_counts().sort_index()
    total_por_grupo = df[grupo].value_counts().sort_index()
    percentuais = (avc_por_grupo / total_por_grupo * 100).fillna(0)

    df_percentual = percentuais.reset_index()
    df_percentual.columns = [grupo, 'stroke_percent']

    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_percentual, x=grupo, y='stroke_percent', order=order, hue=grupo, palette="Set2", legend=False)
    plt.axhline(y=percentual_geral, color='red', linestyle='--', label=f'Média geral ({percentual_geral:.2f}%)')
    plt.title(title)
    plt.xlabel(grupo.replace('_', ' ').capitalize())
    plt.ylabel("Percentual de AVCs (%)")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    if filename:
        plt.savefig(filename)

# Gráficos solicitados

# 2. Gênero
plot_percentual_avc(df, 'gender', "Percentual de AVCs por gênero", order=['Female', 'Male'], filename='gender')

# 3. Idade (faixas de 10 anos)
bins_idade = list(range(0, 91, 10))
labels_idade = [f"{i}-{i+9}" for i in range(0, 90, 10)]
plot_percentual_avc(df, 'age', "Percentual de AVCs por faixa etária (10 anos)", bins=bins_idade, labels=labels_idade, filename='age')

# 4. Hipertensão
plot_percentual_avc(df, 'hypertension', "Percentual de AVCs por hipertensão", order=[0, 1], filename='hypertension')

# 5. Doença cardíaca
plot_percentual_avc(df, 'heart_disease', "Percentual de AVCs por doença cardíaca", order=[0, 1], filename='heart_disease')

# 6. Estado civil
plot_percentual_avc(df, 'ever_married', "Percentual de AVCs por estado civil", order=['No', 'Yes'], filename='ever_married')

# 7. Tipo de trabalho
plot_percentual_avc(df, 'work_type', "Percentual de AVCs por tipo de trabalho",
                    order=['children', 'Govt_job', 'Never_worked', 'Private', 'Self-employed'], filename='work_type')

# 8. Tipo de residência
plot_percentual_avc(df, 'Residence_type', "Percentual de AVCs por tipo de residência", order=['Rural', 'Urban'], filename='residence_type')

# 9. Glicose média (faixas de 20)
max_glucose = int(df['avg_glucose_level'].max()) + 20
bins_glicose = list(range(0, max_glucose, 20))
labels_glicose = [f"{i}-{i+19}" for i in bins_glicose[:-1]]
plot_percentual_avc(df, 'avg_glucose_level', "Percentual de AVCs por glicose média (faixas de 20)",
                    bins=bins_glicose, labels=labels_glicose, filename='glucose_level')

# 10. BMI (6 faixas)
bins_bmi = [0, 18.5, 25, 30, 35, 40, df['bmi'].max() + 1]
labels_bmi = ['Abaixo do peso', 'Normal', 'Sobrepeso', 'Obesidade I', 'Obesidade II', 'Obesidade III']
plot_percentual_avc(df, 'bmi', "Percentual de AVCs por faixa de IMC", bins=bins_bmi, labels=labels_bmi, filename='bmi')

# 11. Fumante
plot_percentual_avc(df, 'smoking_status', "Percentual de AVCs por status de tabagismo",
                    order=['never smoked', 'formerly smoked', 'smokes', 'Unknown'], filename='smoking')


def mostrar_amostras_por_grupo(df, nome_grupo, coluna_original=None):
    """
    Mostra e retorna a contagem de amostras por grupo (como faixa de BMI, glicose, idade agrupada, etc).

    Parâmetros:
    - df: DataFrame original
    - nome_grupo: coluna com os grupos (ex: 'bmi_group', 'avg_glucose_group')
    - coluna_original: opcional, nome da coluna original usada para gerar os grupos (só para visualização)

    Retorna:
    - Um DataFrame com contagens por grupo
    """
    print(f"\n Contagem de amostras por grupo: {nome_grupo}")
    if coluna_original:
        print(f"(Agrupado a partir de: {coluna_original})")
    
    contagem = df[nome_grupo].value_counts().sort_index()
    
    # Exibir de forma bonita
    for grupo, qtd in contagem.items():
        print(f" {grupo}: {qtd} amostras")

    return contagem.reset_index().rename(columns={'index': nome_grupo, nome_grupo: 'quantidade'})


# Supondo que você já tenha criado a coluna 'bmi_group'
mostrar_amostras_por_grupo(df, 'bmi_group', 'bmi')

# Para glicose agrupada
# Agrupar avg_glucose_level de 0 até o valor máximo em intervalos de 20
glucose_bins = list(range(0, int(df['avg_glucose_level'].max()) + 20, 20))
glucose_labels = [f"{glucose_bins[i]}–{glucose_bins[i+1]-1}" for i in range(len(glucose_bins)-1)]

df['avg_glucose_group'] = pd.cut(df['avg_glucose_level'], bins=glucose_bins, labels=glucose_labels, right=False)

mostrar_amostras_por_grupo(df, 'avg_glucose_group', 'avg_glucose_level')

# Para faixa etária
mostrar_amostras_por_grupo(df, 'age_group', 'age')


# Correlação de Pearson (linear) entre variáveis numéricas
correlation_matrix = df.corr(numeric_only=True)

# Visualizar com heatmap
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Matriz de Correlação")
plt.tight_layout()
plt.savefig('correlacao_atributos')


plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x='bmi', color='skyblue')
plt.title("Distribuição de BMI com outliers")
plt.xlabel("BMI")
plt.tight_layout()
plt.savefig("boxplot_bmi_outliers.png")

plt.figure(figsize=(10, 6))


# Scatter plot bmi x age 
plt.figure(figsize=(10, 6))

# Stroke = 0 (sem AVC) - bolinha vazia
sns.scatterplot(data=df[df['stroke'] == 0], x='age', y='bmi',
                edgecolor='black', facecolor='none', label='Sem AVC (0)', alpha=0.5)

# Stroke = 1 (com AVC) - bolinha preenchida
sns.scatterplot(data=df[df['stroke'] == 1], x='age', y='bmi',
                color='red', label='Com AVC (1)', alpha=0.8)

plt.title("Relação entre Idade, BMI e Ocorrência de AVC")
plt.xlabel("Idade")
plt.ylabel("IMC (BMI)")
plt.legend()
plt.tight_layout()
plt.savefig("scatter_age_bmi_stroke.png")
