# Análise Comparativa de Modelos para Previsão de AVC 🧠

Este projeto contém uma implementação completa em Python de um fluxo de trabalho de Machine Learning para **comparar rigorosamente o desempenho de vários algoritmos de classificação** na tarefa de previsão de Acidente Vascular Cerebral (AVC).

A análise utiliza uma metodologia robusta, incluindo:

  * **Validação Cruzada Aninhada (Nested Cross-Validation)**: Para uma estimativa de desempenho não enviesada e ajuste de hiperparâmetros.
  * **Pipeline de Pré-processamento**: Garante que o tratamento dos dados (imputação, normalização, etc.) seja aprendido apenas no conjunto de treino para evitar vazamento de dados (*data leakage*).
  * **Tratamento de Desbalanceamento**: Utiliza a técnica **SMOTE** para lidar com a natureza desbalanceada do conjunto de dados.
  * **Análise Estatística**: Emprega o **Teste de Wilcoxon** para determinar se as diferenças de desempenho entre os modelos são estatisticamente significativas.

-----

## Como Usar este Projeto

Primeiramente, você deve instalar as dependências:

```bash
pip install -r requirements.txt
```

O projeto está organizado da seguinte forma:

  * `datasets/`: esta pasta deve conter o arquivo `healthcare-dataset-stroke-data.csv`. 📁
  * `results/`: esta pasta é criada automaticamente para salvar o gráfico comparativo dos modelos. 📊
  * `classifier.py`: o arquivo principal que contém toda a lógica de pré-processamento, modelagem e avaliação. 🐍

Para executar a análise completa, basta rodar o script principal no seu terminal:

```bash
python classifier.py
```

O script irá imprimir os resultados da validação e do teste para cada modelo no console e salvará um boxplot comparativo na pasta `results/`.

-----

## Metodologia

Todas as etapas de pré-processamento e modelagem seguem as melhores práticas para garantir uma avaliação justa e robusta dos modelos.

1.  **Pré-processamento**: Inclui a limpeza inicial, imputação de valores ausentes (`bmi`) baseada em grupos de gênero e idade, codificação de variáveis categóricas e tratamento de outliers (Winsorizing). Todas as etapas de transformação são ajustadas **apenas** nos dados de treino a cada *fold* da validação cruzada.
2.  **Modelagem**: Para cada algoritmo, uma **Validação Cruzada Aninhada** é executada:
      * **Loop Externo (5-folds)**: Divide os dados para estimar o desempenho do modelo em dados não vistos.
      * **Loop Interno (3-folds com `GridSearchCV`)**: Realiza a busca pelos melhores hiperparâmetros dentro de cada *fold* de treino do loop externo.
3.  **Avaliação**: Os modelos são avaliados usando as métricas **Acurácia Balanceada** e **Average Precision**, adequadas para problemas desbalanceados. Os resultados da validação cruzada são apresentados com média e Intervalo de Confiança de 95%.
4.  **Comparação Estatística**: Ao final, um **Teste de Wilcoxon** é aplicado aos resultados da acurácia balanceada de todos os modelos para verificar se as diferenças de desempenho são estatisticamente significantes.

-----

## Algoritmos Avaliados

Este script compara o desempenho dos seguintes algoritmos de classificação:

  * `DummyClassifier` (Baseline)
  * `Logistic Regression`
  * `K-Nearest Neighbors` (KNN)
  * `Decision Tree`
  * `Support Vector Machine` (SVM)
  * `Gaussian Naive Bayes`
  * `Random Forest`
  * `AdaBoost`
  * `XGBoost`

-----

## Dependências Principais 📚

Se preferir instalar manualmente, as principais bibliotecas utilizadas são:

  * `pandas`
  * `numpy`
  * `scikit-learn`
  * `imbalanced-learn`
  * `xgboost`
  * `matplotlib`
  * `seaborn`
  * `scipy`

-----

# Análise Exploratória de Dados (EDA) - Previsão de AVC 📊

Este repositório contém scripts Python para uma Análise Exploratória de Dados (EDA) detalhada e multifacetada do conjunto de dados sobre Acidente Vascular Cerebral (AVC). O objetivo é investigar as relações entre as variáveis, identificar os principais fatores de risco e extrair insights que possam guiar a etapa de modelagem preditiva.

A análise é dividida em duas abordagens complementares:

  * **Análise Estatística Profunda:** Focada em testes de hipóteses e quantificação de associações.
  * **Visualização Bivariada Detalhada:** Focada na criação de gráficos individuais para cada variável, comparando a incidência de AVC contra uma linha de base geral.

## Principais Características e Análises

O projeto realiza uma investigação completa, cobrindo os seguintes pontos:

  * **Estatísticas Descritivas:** Análise inicial de desbalanceamento de classes e valores ausentes.
  * **Testes Estatísticos Bivariados:**
      * **Teste Qui-quadrado (χ²):** Para avaliar a associação entre variáveis categóricas (gênero, hipertensão, etc.) e a ocorrência de AVC. A força da associação é medida com o **V de Cramér**.
      * **Teste Mann-Whitney U:** Para comparar as distribuições de variáveis numéricas (idade, glicose, IMC) entre os grupos com e sem AVC, ideal para dados não-normais.
  * **Análise de Correlação:** Utiliza a correlação de **Spearman** para medir a força e a direção da relação monotônica entre as variáveis numéricas.
  * **Análise de Risco Combinado:** Cria um `risk_score` para demonstrar como a combinação de fatores de risco (idade, hipertensão, doença cardíaca) impacta a probabilidade de AVC.
  * **Visualização Padronizada:** Gera gráficos de barras para cada variável, mostrando o percentual de casos de AVC por categoria e uma linha de base (média geral de AVC), facilitando a identificação de grupos de risco.

## Como Usar

Tendo em vista que você já instalou as dependências anteriormente, então execute o comando abaixo para gerar a análise dos dados:

```bash
# Para a análise estatística e gráficos de resumo
python analisis.py
```

Os resultados dos testes estatísticos serão impressos no console, e todos os gráficos gerados serão salvos na pasta `analisis/`.

## Principais Insights e Visualizações 💡

A execução dos scripts gera uma série de visualizações e relatórios estatísticos. Os principais insights incluem:

  * **Fatores de Risco Dominantes:** Idade, hipertensão e doença cardíaca são os fatores mais fortemente associados a um maior risco de AVC.
  * **O Enigma do IMC (BMI):** Embora estatisticamente significativo, o IMC por si só não é um forte discriminador. A análise mostra que a média de IMC não aumenta drasticamente com a idade, ao contrário da taxa de AVC, indicando que seu impacto é mais complexo.
  * **Efeito de Combinação:** O `risk_score` demonstra claramente que a acumulação de fatores de risco eleva exponencialmente a probabilidade de um AVC.