
## 1. Objetivo
 
Prever o consumo de combustível (`mpg`, em milhas por galão) de veículos a partir de suas características técnicas, comparando três abordagens:
 
1. Regressão linear simples (um preditor numérico).
2. Regressão linear múltipla (dois ou mais preditores, incluindo o da simples).
3. Regressão polinomial (graus 2, 3 e 5 sobre o preditor da simples).
O foco é interpretar coeficientes, métricas e resíduos, e distinguir ajuste aos dados de treino de capacidade de generalização.
 
## 2. Dataset
 
| Campo | Informação |
|---|---|
| **Título** | Auto-mpg dataset |
| **URL no Kaggle** | https://www.kaggle.com/datasets/uciml/autompg-dataset |
| **Publicador no Kaggle** | uciml _(confirmar na página)_ |
| **Fonte original** | UCI Machine Learning Repository, *Auto MPG* (Quinlan, 1993), revisado a partir da biblioteca StatLib (Carnegie Mellon) |
| **URL da fonte original** | https://archive.ics.uci.edu/dataset/9/auto+mpg |
| **DOI** | https://doi.org/10.24432/C5859H |
| **Licença na UCI** | CC BY 4.0 |
| **Data de acesso** | 05/10/2026 |
| **Arquivo utilizado** | `auto-mpg.csv` |
| **Natureza dos dados** | Reais (não sintéticos) |
 
**Unidade de observação:** cada linha representa um modelo de veículo.
**Alvo:** `mpg` (milhas por galão, contínuo).
**Preditores candidatos:** `displacement`, `horsepower`, `weight`, `acceleration` _(a escolha final será justificada no relatório, com base apenas no treino)_.
 
> O arquivo CSV **não** é versionado neste repositório por padrão. Veja a seção 5 para obter os dados.
 
## 3. Estrutura do repositório
 
```
.
├── README.md
├── notebooks/
│   └── Nome_RA_Unidade2_Regressao.ipynb
├── relatorio/
│   └── Nome_RA_Unidade2_Relatorio.pdf
├── data/
│   └── README.md            # instruções para obter o CSV
├── figuras/                 # gráficos exportados do notebook
├── docs/
│   └── Atividade_Pratica_Regressao_Unidade2.pdf   # enunciado
├── requirements.txt
└── .gitignore
```
 
## 4. Protocolo experimental (resumo)
 
- **Bibliotecas:** pandas, NumPy, Matplotlib, scikit-learn (Seaborn opcional). Versões registradas no notebook e no `requirements.txt`.
- **Separação:** 60% treino / 20% validação / 20% teste. Primeiro reserva-se 20% para teste; depois 25% dos 80% restantes para validação. `train_test_split` com `random_state=42` nas duas etapas. Os mesmos índices são usados nos três cenários.
- **Sem vazamento:** exploração, imputação e padronização usam apenas o treino. Etapas ajustadas (`SimpleImputer`, `StandardScaler`) ficam dentro de `Pipeline`.
- **Baseline:** `DummyRegressor(strategy="mean")`.
- **Métricas:** MAE, RMSE e R², em treino e validação, para baseline, simples, múltipla e graus 2, 3 e 5.
- **Escolha:** grau polinomial escolhido pelo RMSE de validação. O teste só é usado uma vez, ao final, após reajustar os quatro modelos em treino + validação.
## 5. Como reproduzir
 
1. Abra o notebook no Google Colab: _LINK DO COLAB (permissão de leitura para o professor)_.
2. Baixe `auto-mpg.csv` na página do Kaggle indicada acima.
3. Envie o arquivo para a sessão do Colab (ou monte o Google Drive) e ajuste o caminho na célula de carregamento, conforme explicado no notebook.
4. Use **Ambiente de execução → Executar tudo** em uma sessão nova.
Para rodar localmente:
 
```bash
pip install -r requirements.txt
jupyter notebook notebooks/
```
 
## 6. Andamento do projeto
 
- [x] Escolha do dataset
- [ ] Pesquisa de aplicação de regressão e leitura da documentação do scikit-learn
- [ ] Carregamento, inspeção inicial e regras de limpeza
- [ ] Separação treino / validação / teste
- [ ] Exploração do treino e hipóteses
- [ ] Baseline
- [ ] Regressão linear simples
- [ ] Regressão linear múltipla
- [ ] Regressão polinomial (graus 2, 3 e 5)
- [ ] Escolha do grau por validação
- [ ] Reajuste final e avaliação no teste
- [ ] Diagnóstico de resíduos e limitações
- [ ] Relatório em PDF
- [ ] Conferência final (sessão nova, link do Colab, números do relatório)
## 7. Resultados
 
| Modelo | MAE (val) | RMSE (val) | R² (val) | RMSE (teste) |
|---|---|---|---|---|
| Baseline (média) | | | | |
| Linear simples | | | | |
| Linear múltipla | | | | |
| Polinomial (grau escolhido) | | | | |
 
## 8. Referências
 
- Quinlan, R. (1993). *Auto MPG* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5859H
- Quinlan, R. (1993). Combining Instance-Based and Model-Based Learning. *Proceedings of the Tenth International Conference on Machine Learning*, 236-243.
- scikit-learn: [LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
- scikit-learn: [PolynomialFeatures](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html)
- scikit-learn: [Métricas de regressão](https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics)
- scikit-learn: [Armadilhas comuns e vazamento de dados](https://scikit-learn.org/stable/common_pitfalls.html)
- scikit-learn: [DummyRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyRegressor.html)
- _Aplicação de regressão pesquisada no tema (a adicionar)_
