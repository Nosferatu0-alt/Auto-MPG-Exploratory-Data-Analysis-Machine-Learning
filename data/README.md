# Dados

O arquivo de dados **não é versionado** neste repositório (ver `.gitignore`). Obtenha-o por uma das opções abaixo.

## Origem

| Campo | Informação |
|---|---|
| Título | Auto-mpg dataset |
| Kaggle | https://www.kaggle.com/datasets/uciml/autompg-dataset |
| Publicador no Kaggle | uciml (UCI Machine Learning) |
| Fonte original | UCI Machine Learning Repository, *Auto MPG* (Quinlan, 1993), revisado a partir da biblioteca StatLib (Carnegie Mellon University) |
| URL da fonte | https://archive.ics.uci.edu/dataset/9/auto+mpg |
| DOI | https://doi.org/10.24432/C5859H |
| Licença | CC BY 4.0 |
| Data de acesso | 05/10/2026 |
| Arquivo usado | `auto-mpg.csv` |
| Natureza | Dados reais (não sintéticos) |

## Como obter

1. **Automático:** o notebook tenta baixar com `kagglehub` (`uciml/autompg-dataset`), sem token, por ser dataset público.
2. **Manual:** baixe o CSV na página do Kaggle e salve como `data/auto-mpg.csv` (local) ou envie ao Colab quando solicitado.

## Esquema (398 linhas, 9 colunas)

| Coluna | Descrição | Unidade |
|---|---|---|
| `mpg` | **Alvo.** Consumo | milhas por galão |
| `cylinders` | Nº de cilindros | — |
| `displacement` | Cilindrada | polegadas cúbicas |
| `horsepower` | Potência (6 valores ausentes, marcados com `?`) | hp |
| `weight` | Peso | lb |
| `acceleration` | Tempo de 0 a 60 mph | s |
| `model year` | Ano do modelo (70 = 1970) | ano |
| `origin` | Região (1 = EUA, 2 = Europa, 3 = Japão) | código |
| `car name` | Nome do carro (identificador; não usado como preditor) | texto |
