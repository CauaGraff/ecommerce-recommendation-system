# Sistema de Recomendação para E-commerce

Sistema de recomendação de produtos baseado no comportamento de navegação
dos usuários, com um modelo central em rede neural (PyTorch), pipeline
containerizado (Docker), dados versionados (DVC) e experimentos
rastreados (MLflow).

Dataset utilizado: [MovieLens 100k](https://files.grouplens.org/datasets/movielens/ml-32m.zip).


## Estrutura do projeto

```
.
├── src/
│   ├── models/               # arquiteturas de rede neural + baselines Scikit-Learn
│   ├── factories/            # Factory Method para criação de modelos
│   ├── preprocessing/        # Strategy Pattern para pré-processamento
│   ├── training/             # Template Method (Trainer) + Early Stopping
│   ├── evaluation/           # métricas de avaliação (mse, rmse, mae, r2)
│   ├── data/                 # loader do dataset + encoders de IDs
│   └── config/               # Pydantic Settings
├── tests/                    # testes unitários (pytest)
├── docs/model_card.md        # Model Card (performance, limitações, vieses)
├── configs/                  # arquivos de configuração (YAML)
├── scripts/                  # CLIs finas: cada stage do pipeline DVC
├── models/                   # modelos treinados serializados
├── data/                     # dados versionados via DVC
│   ├── raw/
│   └── processed/
├── artifacts/                # métricas e artefatos de experimentos
├── Dockerfile                # multi-stage: builder + runtime
├── docker-compose.yml        # serviços: mlflow + train
├── dvc.yaml                     
├── pyproject.toml
├── .env.example
├── .pre-commit-config.yaml
└── .gitignore
```

## Como rodar o projeto a partir de um clone

Siga os passos abaixo para preparar o ambiente em uma máquina nova e executar o fluxo completo do projeto.

### 1. Clonar o repositório

```bash
git clone https://github.com/CauaGraff/ecommerce-recommendation-system.git
cd ecommerce-recommendation-system
```

### 2. Criar o ambiente virtual e instalar dependências

Este projeto usa o gerenciador de pacotes `uv`.

```bash
uv sync
cp .env.example .env
uv run python scripts/validate_env.py
```

Se a validação passar, o ambiente está pronto para uso. Caso contrário, revise as dependências e o arquivo `.env`.

### 3. Rodar os testes

```bash
uv run pytest -v
```

### 4. Subir o MLflow para acompanhar experimentos

```bash
docker compose up -d mlflow
```

O painel do MLflow ficará disponível em `http://localhost:5000`.

### 5. Executar o pipeline completo

```bash
dvc pull
dvc repro
```

O comando acima executa as etapas de preprocessamento, engenharia de features, treino do modelo neural, treino dos baselines e avaliação. Ao final, as métricas são salvas em `artifacts/` e o modelo pode aparecer no MLflow com ciclo de vida `Staging`/`Production`.

### 6. Visualizar métricas e experimentos

```bash
dvc metrics show
```

Também é possível acompanhar o processo em `http://localhost:5000` na aba `Models`.

### 7. Rodar o treino dentro do Docker

```bash
docker compose up --build train
```

Essa opção é útil quando se quer rodar o treino em um ambiente mais isolado e semelhante ao usado para entrega do projeto.

## Sobre o dataset

O projeto usa o dataset MovieLens 100K como substituto para um ambiente real de e-commerce. Ele contém interações de usuários com filmes, o que permite simular um problema clássico de recomendação com feedback implícito/explicito.

### Arquivos principais

- `data/raw/ratings.csv`: avaliações de usuários para filmes, com colunas como `userId`, `movieId`, `rating` e `timestamp`.
- `data/raw/movies.csv`: informações dos filmes, como identificador e título.
- `data/raw/tags.csv`: tags atribuídas pelos usuários, usadas como contexto adicional.
- `data/raw/links.csv`: links para bases externas como IMDb/TMDB.

### Como o projeto usa esses dados

1. Os IDs de usuários e filmes são transformados em representações numéricas para o modelo.
2. O dataset é dividido em treino, validação e teste.
3. O modelo aprende a prever a nota que um usuário provavelmente daria a um item.
4. As métricas de erro (MSE, RMSE, MAE e $R^2$) são usadas para comparar o modelo neural com os baselines.

Essa abordagem é uma simplificação de um cenário real de e-commerce, mas preserva os desafios centrais de recomendação: sparse interactions, cold start e necessidade de generalização.

## Pipeline de dados, treino e avaliação

```bash
# 1. Preparar ambiente
dvc pull

# 2. Subir o MLflow server (via Docker Compose)
docker compose up -d mlflow

# 3. Rodar o pipeline completo:
#    preprocess -> feature_engineering -> train (com early stopping,
#    registra no Model Registry como "Staging") -> train_baselines
#    (Scikit-Learn) -> evaluate (4 métricas + promove a "Production" se
#    RMSE <= limiar definido em scripts/evaluate.py)
dvc repro

# 4. Comparar resultados
dvc metrics show
# ou acesse http://localhost:5000 -> aba "Models" para ver o ciclo de
# vida do modelo (Staging -> Production) no Model Registry

# 5. Rodar o treino dentro do Docker
docker compose up --build train
```

## Model Card

Ver [`docs/model_card.md`](docs/model_card.md) para arquitetura, dados de
treino, performance comparada aos baselines, limitações e vieses conhecidos.
Preencha a tabela de performance com os números reais do seu `dvc repro`
antes da entrega.
