# Model Card — Embedding Recommender (E-commerce/MovieLens)

## Visão geral

- **Tipo de modelo:** rede neural com embeddings de usuário e item (`EmbeddingRecommender`), treinada em PyTorch.
- **Tarefa:** prever o rating (1–5) que um usuário daria a um item, a partir de pares `(userId, movieId)`.
- **Dataset:** [MovieLens](https://grouplens.org/datasets/movielens/100k/) — 100.000 avaliações, usado como substituto do dataset de interações de e-commerce sugerido no desafio.
- **Framework:** PyTorch (modelo), Scikit-Learn (baselines e métricas), MLflow (tracking e registry), DVC (versionamento de dados e pipeline).

## Arquitetura

- Camadas de embedding (`nn.Embedding`) para usuário e item, dimensão configurável (padrão: 32).
- Score de afinidade calculado via produto escalar entre os dois vetores de embedding.
- Treinado com `MSELoss` e otimizador Adam, com early stopping (paciência configurável) monitorando a perda de validação.

## Dados de treino

- Split: 80% treino / 20% teste (seed fixada via configuração), com 10% do treino reservado para validação durante o treino.
- Pré-processamento: remoção de duplicatas e nulos; `userId`/`movieId` convertidos em índices contíguos via `LabelEncoder` (ajustado em treino+teste combinados, para evitar IDs desconhecidos).

## Performance

> Preencha esta seção com os valores reais reportados por `uv run python scripts/evaluate.py` e `uv run python scripts/train_baselines.py` no seu ambiente, e depois copie os números do MLflow aqui antes da entrega final.

| Modelo | MSE | RMSE | MAE | R² |
|---|---|---|---|---|
| Baseline — Média Global | 1.1000666029381543 | 1.0488405993944716 | 0.8315907291881814 | -2.3425006417499006e-05 |
| Baseline — Regressão Linear | 1.0973639025187965 | 1.0475513841901964 | 0.8355545678425147 | 0.002433484168821276 |
| **Embedding Recommender (PyTorch)** | 2.391535997390747 | 1.5464591806416188 | 1.128282070159912 | -1.1740427017211914 |

Critério de promoção a `Production` no MLflow Model Registry: RMSE de teste ≤ `1.2` (ajustável em `scripts/evaluate.py`, constante `PRODUCTION_RMSE_THRESHOLD`).

## Limitações

- **Cold start:** o modelo não sabe prever ratings para usuários ou filmes que não estavam no conjunto de treino/teste original — os encoders (`user_encoder.joblib`, `item_encoder.joblib`) não têm índice para IDs desconhecidos.
- **Sem contexto temporal:** o modelo não usa a coluna `timestamp` — não captura mudanças de preferência do usuário ao longo do tempo, nem tendências sazonais de produtos.
- **Sem features de conteúdo:** não usa metadados de item (categoria, gênero, preço) nem de usuário (idade, localização) — aprende apenas por co-ocorrência de interações.
- **Dataset substituto:** o MovieLens é um proxy razoável para "interações usuário-item", mas o domínio (filmes) difere de um catálogo real de e-commerce (maior turnover de produtos, dados de navegação implícitos como cliques/carrinho em vez de ratings explícitos 1–5).

## Vieses conhecidos

- **Popularidade:** filmes com muitas avaliações no treino tendem a ter embeddings mais bem ajustados; filmes raros (poucas interações) têm previsões menos confiáveis — um viés comum de popularidade em sistemas de recomendação baseados em embedding.
- **Viés de exposição:** o dataset reflete o que os usuários efetivamente avaliaram, não uma amostra aleatória de todos os pares usuário-item possíveis — o modelo aprende a partir de decisões de exposição/seleção que já existiam no sistema original de coleta dos dados.
- **Sem auditoria de subgrupos:** este projeto não segmentou a performance por grupos de usuários (ex.: usuários com poucas avaliações vs. muitas) — recomenda-se essa análise antes de um uso real em produção.

## Uso recomendado

- Adequado como recomendador colaborativo baseline para prototipagem e para o objetivo do Tech Challenge (demonstrar o pipeline MLOps completo).
- **Não recomendado** para produção real sem: tratamento de cold start, features de conteúdo, e reavaliação periódica com dados novos (drift).
