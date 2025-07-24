import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import chi2_contingency, mannwhitneyu, spearmanr
import warnings
warnings.filterwarnings("ignore")

# Configuração para gráficos
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Carregamento e preparação dos dados
df = pd.read_csv("./datasets/healthcare-dataset-stroke-data.csv")
df.drop(columns=['id'], inplace=True)

print("=== ANÁLISE EXPLORATÓRIA COMPLETA - STROKE DATASET ===\n")

# 1. ESTATÍSTICAS DESCRITIVAS BÁSICAS
print("1. ESTATÍSTICAS DESCRITIVAS BÁSICAS")
print("-" * 50)
print(f"Total de amostras: {len(df)}")
print(f"Total de casos de AVC: {df['stroke'].sum()} ({df['stroke'].mean()*100:.2f}%)")
print(f"Proporção de casos sem AVC: {(1-df['stroke'].mean())*100:.2f}%")
print(f"Razão de desequilíbrio: {(1-df['stroke'].mean())/df['stroke'].mean():.1f}:1")
print()

# 2. ANÁLISE DE VALORES AUSENTES
print("2. ANÁLISE DE VALORES AUSENTES")
print("-" * 50)
missing_data = df.isnull().sum()
missing_percent = (missing_data / len(df)) * 100
missing_df = pd.DataFrame({
    'Valores_Ausentes': missing_data,
    'Percentual': missing_percent
}).sort_values('Percentual', ascending=False)

print(missing_df[missing_df['Valores_Ausentes'] > 0])
print()

# 3. ANÁLISE ESTATÍSTICA BIVARIADA
print("3. ANÁLISE ESTATÍSTICA BIVARIADA (Stroke vs Features)")
print("-" * 50)

# Variáveis categóricas - Teste Chi-quadrado
categorical_vars = ['gender', 'hypertension', 'heart_disease', 'ever_married', 
                   'work_type', 'Residence_type', 'smoking_status']

chi2_results = []
for var in categorical_vars:
    if df[var].dtype == 'object' or var in ['hypertension', 'heart_disease']:
        contingency_table = pd.crosstab(df[var], df['stroke'])
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        
        # Calcular Cramér's V (força da associação)
        n = contingency_table.sum().sum()
        cramers_v = np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1)))
        
        chi2_results.append({
            'Variável': var,
            'Chi2': chi2,
            'p-valor': p_value,
            'Cramers_V': cramers_v,
            'Significativo': 'Sim' if p_value < 0.05 else 'Não'
        })

chi2_df = pd.DataFrame(chi2_results)
print("Teste Chi-quadrado (Variáveis Categóricas):")
print(chi2_df.round(4))
print()

# Variáveis numéricas - Teste Mann-Whitney U
numerical_vars = ['age', 'avg_glucose_level', 'bmi']
mannwhitney_results = []

for var in numerical_vars:
    # Remover valores NaN
    clean_data = df[df[var].notna()]
    group_stroke = clean_data[clean_data['stroke'] == 1][var]
    group_no_stroke = clean_data[clean_data['stroke'] == 0][var]
    
    statistic, p_value = mannwhitneyu(group_stroke, group_no_stroke, alternative='two-sided')
    
    # Calcular effect size (r = Z/sqrt(N))
    z_score = stats.norm.ppf(p_value/2)
    effect_size = abs(z_score) / np.sqrt(len(clean_data))
    
    # Medianas para interpretação
    median_stroke = group_stroke.median()
    median_no_stroke = group_no_stroke.median()
    
    mannwhitney_results.append({
        'Variável': var,
        'U-statistic': statistic,
        'p-valor': p_value,
        'Effect_Size': effect_size,
        'Mediana_AVC': median_stroke,
        'Mediana_Sem_AVC': median_no_stroke,
        'Significativo': 'Sim' if p_value < 0.05 else 'Não'
    })

mannwhitney_df = pd.DataFrame(mannwhitney_results)
print("Teste Mann-Whitney U (Variáveis Numéricas):")
print(mannwhitney_df.round(4))
print()

# 4. ANÁLISE DE CORRELAÇÃO ENTRE VARIÁVEIS NUMÉRICAS
print("4. ANÁLISE DE CORRELAÇÃO (Spearman)")
print("-" * 50)
numerical_data = df[numerical_vars + ['stroke']].dropna()
correlation_matrix = numerical_data.corr(method='spearman')

print("Correlação de Spearman com AVC:")
stroke_correlations = correlation_matrix['stroke'].drop('stroke').sort_values(key=abs, ascending=False)
for var, corr in stroke_correlations.items():
    print(f"{var}: {corr:.4f}")
print()

# 5. ANÁLISE DETALHADA POR FAIXAS ETÁRIAS
print("5. ANÁLISE DETALHADA POR FAIXAS ETÁRIAS")
print("-" * 50)

# Criar faixas etárias mais específicas
age_bins = [0, 30, 45, 60, 75, 100]
age_labels = ['<30', '30-44', '45-59', '60-74', '75+']
df['age_group_detailed'] = pd.cut(df['age'], bins=age_bins, labels=age_labels, right=False)

age_analysis = df.groupby('age_group_detailed').agg({
    'stroke': ['count', 'sum', 'mean'],
    'age': 'mean',
    'bmi': 'mean',
    'avg_glucose_level': 'mean',
    'hypertension': 'mean',
    'heart_disease': 'mean'
}).round(3)

age_analysis.columns = ['Total_Casos', 'Casos_AVC', 'Taxa_AVC', 'Idade_Media', 
                       'BMI_Medio', 'Glicose_Media', 'Taxa_Hipertensao', 'Taxa_Doenca_Cardiaca']

print("Análise por faixa etária:")
print(age_analysis)
print()

# 6. ANÁLISE DO BMI - Por que não é tão relevante?
print("6. ANÁLISE CRÍTICA DO BMI")
print("-" * 50)

# BMI por faixa etária em casos de AVC
bmi_age_analysis = df[df['stroke'] == 1].groupby('age_group_detailed')['bmi'].agg(['count', 'mean', 'std']).round(2)
print("BMI em casos de AVC por faixa etária:")
print(bmi_age_analysis)
print()

# Análise de outliers no BMI
q1 = df['bmi'].quantile(0.25)
q3 = df['bmi'].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

outliers_count = len(df[(df['bmi'] < lower_bound) | (df['bmi'] > upper_bound)])
print(f"Outliers no BMI: {outliers_count} ({outliers_count/len(df)*100:.1f}%)")
print(f"Faixa normal (Q1-Q3): {q1:.1f} - {q3:.1f}")
print(f"Mediana BMI (AVC): {df[df['stroke']==1]['bmi'].median():.1f}")
print(f"Mediana BMI (Sem AVC): {df[df['stroke']==0]['bmi'].median():.1f}")
print()

# 7. FATORES DE RISCO COMBINADOS
print("7. ANÁLISE DE FATORES DE RISCO COMBINADOS")
print("-" * 50)

# Criar score de risco
df['risk_score'] = (
    df['hypertension'] * 1 +
    df['heart_disease'] * 2 +
    (df['age'] > 65).astype(int) * 2 +
    (df['avg_glucose_level'] > 140).astype(int) * 1
)

risk_analysis = df.groupby('risk_score').agg({
    'stroke': ['count', 'sum', 'mean']
}).round(3)
risk_analysis.columns = ['Total_Casos', 'Casos_AVC', 'Taxa_AVC']

print("Taxa de AVC por score de risco combinado:")
print(risk_analysis)
print()

# 8. GRÁFICOS INFORMATIVOS
print("8. GERANDO GRÁFICOS INFORMATIVOS")
print("-" * 50)

# Gráfico 1: Distribuição de idade por status de AVC
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
df[df['stroke'] == 0]['age'].hist(bins=30, alpha=0.7, label='Sem AVC', density=True)
df[df['stroke'] == 1]['age'].hist(bins=30, alpha=0.7, label='Com AVC', density=True)
plt.xlabel('Idade')
plt.ylabel('Densidade')
plt.title('Distribuição de Idade por Status de AVC')
plt.legend()

# Gráfico 2: Taxa de AVC por faixa etária
plt.subplot(2, 2, 2)
stroke_rate_by_age = df.groupby('age_group_detailed')['stroke'].mean() * 100
stroke_rate_by_age.plot(kind='bar', color='coral')
plt.xlabel('Faixa Etária')
plt.ylabel('Taxa de AVC (%)')
plt.title('Taxa de AVC por Faixa Etária')
plt.xticks(rotation=45)

# Gráfico 3: Comparação BMI entre grupos
plt.subplot(2, 2, 3)
sns.boxplot(data=df, x='stroke', y='bmi')
plt.xlabel('Status AVC (0=Não, 1=Sim)')
plt.ylabel('BMI')
plt.title('Distribuição de BMI por Status de AVC')

# Gráfico 4: Heatmap de correlação
plt.subplot(2, 2, 4)
correlation_matrix_plot = df[['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi', 'stroke']].corr()
sns.heatmap(correlation_matrix_plot, annot=True, cmap='coolwarm', center=0)
plt.title('Matriz de Correlação')

plt.tight_layout()
plt.savefig('./analisis/analise_exploratoria_completa.png', dpi=300, bbox_inches='tight')
plt.show()

# Gráfico específico: Por que BMI não é discriminativo
plt.figure(figsize=(14, 5))

plt.subplot(1, 3, 1)
# Scatter plot idade vs BMI colorido por AVC
scatter = plt.scatter(df['age'], df['bmi'], c=df['stroke'], alpha=0.6, cmap='RdYlBu')
plt.xlabel('Idade')
plt.ylabel('BMI')
plt.title('Idade vs BMI (colorido por AVC)')
plt.colorbar(scatter, label='AVC')

plt.subplot(1, 3, 2)
# BMI distribution by age groups for stroke cases
df_stroke = df[df['stroke'] == 1]
for age_group in ['<30', '30-44', '45-59', '60-74', '75+']:
    if age_group in df_stroke['age_group_detailed'].values:
        data = df_stroke[df_stroke['age_group_detailed'] == age_group]['bmi'].dropna()
        if len(data) > 0:
            plt.hist(data, alpha=0.5, label=age_group, bins=15)
plt.xlabel('BMI')
plt.ylabel('Frequência')
plt.title('Distribuição de BMI em casos de AVC por idade')
plt.legend()

plt.subplot(1, 3, 3)
# Taxa de AVC por score de risco
risk_rates = df.groupby('risk_score')['stroke'].mean() * 100
risk_rates.plot(kind='bar', color='steelblue')
plt.xlabel('Score de Risco Combinado')
plt.ylabel('Taxa de AVC (%)')
plt.title('Taxa de AVC por Score de Risco')
plt.xticks(rotation=0)

plt.tight_layout()
plt.savefig('./analisis/analise_bmi_detalhada.png', dpi=300, bbox_inches='tight')
plt.show()

print("Análise exploratória completa finalizada!")
print("Arquivos salvos em: ./analisis/")




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
plot_percentual_avc(df, 'gender', "Percentual de AVCs por gênero", order=['Female', 'Male'], filename='./analisis/gender')

# 3. Idade (faixas de 10 anos)
bins_idade = list(range(0, 91, 10))
labels_idade = [f"{i}-{i+9}" for i in range(0, 90, 10)]
plot_percentual_avc(df, 'age', "Percentual de AVCs por faixa etária (10 anos)", bins=bins_idade, labels=labels_idade, filename='./analisis/age')

# 4. Hipertensão
plot_percentual_avc(df, 'hypertension', "Percentual de AVCs por hipertensão", order=[0, 1], filename='./analisis/hypertension')

# 5. Doença cardíaca
plot_percentual_avc(df, 'heart_disease', "Percentual de AVCs por doença cardíaca", order=[0, 1], filename='./analisis/heart_disease')

# 6. Estado civil
plot_percentual_avc(df, 'ever_married', "Percentual de AVCs por estado civil", order=['No', 'Yes'], filename='./analisis/ever_married')

# 7. Tipo de trabalho
plot_percentual_avc(df, 'work_type', "Percentual de AVCs por tipo de trabalho",
                    order=['children', 'Govt_job', 'Never_worked', 'Private', 'Self-employed'], filename='./analisis/work_type')

# 8. Tipo de residência
plot_percentual_avc(df, 'Residence_type', "Percentual de AVCs por tipo de residência", order=['Rural', 'Urban'], filename='./analisis/residence_type')

# 9. Glicose média (faixas de 20)
max_glucose = int(df['avg_glucose_level'].max()) + 20
bins_glicose = list(range(0, max_glucose, 20))
labels_glicose = [f"{i}-{i+19}" for i in bins_glicose[:-1]]
plot_percentual_avc(df, 'avg_glucose_level', "Percentual de AVCs por glicose média (faixas de 20)",
                    bins=bins_glicose, labels=labels_glicose, filename='./analisis/glucose_level')

# 10. BMI (6 faixas)
bins_bmi = [0, 18.5, 25, 30, 35, 40, df['bmi'].max() + 1]
labels_bmi = ['Abaixo do peso', 'Normal', 'Sobrepeso', 'Obesidade I', 'Obesidade II', 'Obesidade III']
plot_percentual_avc(df, 'bmi', "Percentual de AVCs por faixa de IMC", bins=bins_bmi, labels=labels_bmi, filename='./analisis/bmi')

# 11. Fumante
plot_percentual_avc(df, 'smoking_status', "Percentual de AVCs por status de tabagismo",
                    order=['never smoked', 'formerly smoked', 'smokes', 'Unknown'], filename='./analisis/smoking')

# Correlação de Pearson (linear) entre variáveis numéricas
correlation_matrix = df.corr(numeric_only=True)

# Visualizar com heatmap
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Matriz de Correlação")
plt.tight_layout()
plt.savefig('./analisis/correlacao_atributos')


plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x='bmi', color='skyblue')
plt.title("Distribuição de BMI com outliers")
plt.xlabel("BMI")
plt.tight_layout()
plt.savefig("./analisis/boxplot_bmi_outliers.png")

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
plt.savefig("./analisis/scatter_age_bmi_stroke.png")


# --- Gráfico combinado: % AVC por faixa etária + média de BMI ---

# Garantir que 'age_group' foi criada
bins_idade = list(range(0, 91, 10))
labels_idade = [f"{i}-{i+9}" for i in range(0, 90, 10)]
df['age_group'] = pd.cut(df['age'], bins=bins_idade, labels=labels_idade, right=False)

# Calcular percentual de AVC por faixa etária
total_por_faixa = df['age_group'].value_counts().sort_index()
avc_por_faixa = df[df['stroke'] == 1]['age_group'].value_counts().sort_index()
percentual_avc = (avc_por_faixa / total_por_faixa * 100).fillna(0)

# Calcular média de BMI por faixa etária
media_bmi = df.groupby('age_group', observed=False)['bmi'].mean()

# --- Plotando gráfico combinado ---
fig, ax1 = plt.subplots(figsize=(10, 6))

# Eixo da esquerda: barras vermelhas com % de AVC
ax1.bar(percentual_avc.index, percentual_avc.values, color='crimson', alpha=0.7, label='% de AVC')
ax1.set_ylabel('% de AVC', color='crimson')
ax1.tick_params(axis='y', labelcolor='crimson')
ax1.set_xlabel('Faixa Etária')
ax1.set_title('% de AVC e Média de BMI por Faixa Etária')

# Eixo da direita: linha azul com média de BMI
ax2 = ax1.twinx()
ax2.plot(media_bmi.index, media_bmi.values, color='steelblue', marker='o', linewidth=2, label='Média de BMI')
ax2.set_ylabel('Média de BMI', color='steelblue')
ax2.tick_params(axis='y', labelcolor='steelblue')

# Legenda combinada
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

plt.tight_layout()
plt.savefig("./analisis/grafico_combinado_avc_bmi_por_idade.png")


# Filtrar pessoas entre 30 e 60 anos
df_jovens = df[(df['age'] >= 30) & (df['age'] <= 60)].copy()

# Definir faixas de BMI
bins_bmi = [0, 18.5, 25, 30, 35, 40, df['bmi'].max() + 1]
labels_bmi = ['Abaixo do peso', 'Normal', 'Sobrepeso', 'Obesidade I', 'Obesidade II', 'Obesidade III']

# Gerar gráfico apenas com os jovens (30 a 60 anos)
plot_percentual_avc(
    df_jovens,
    col='bmi',
    title="Percentual de AVCs por faixa de IMC (30 a 60 anos)",
    bins=bins_bmi,
    labels=labels_bmi,
    filename='./analisis/bmi_jovens_30a60'
)
