# Nutritional Score Engine — MVP

Engine de score nutricional personalizado: dado um **indivíduo** e um **alimento**, o sistema retorna um score de **0 a 10** indicando compatibilidade nutricional.

## Stack

| Camada | Tecnologia |
|---|---|
| Dados de alimentos | Open Food Facts (API pública) |
| Dados de indivíduos | Geração sintética (distribuições realistas) |
| Modelo | `scikit-learn` MLPRegressor |
| API | FastAPI + Uvicorn |

## Estrutura

```
chall-ia/
├── data/
│   ├── raw/              # JSON bruto do Open Food Facts
│   ├── processed/        # CSVs de treino
│   └── models/           # model.pkl + preprocessor.pkl
├── src/
│   ├── data/             # fetch_foods · generate_individuals · generate_pairs
│   ├── features/         # preprocessing (ColumnTransformer)
│   ├── model/            # train · evaluate
│   └── api/              # FastAPI (main.py)
├── docs/
│   └── approach.md       # documentação completa
└── run_pipeline.py       # orquestrador
```

## Instalação

```bash
pip install -r requirements.txt
```

## Uso

### 1. Executar o pipeline completo

```bash
python run_pipeline.py
```

Se os alimentos já foram baixados anteriormente:

```bash
python run_pipeline.py --skip-fetch
```

### 2. Iniciar a API

```bash
uvicorn src.api.main:app --reload
```

Acesse a documentação interativa: **http://127.0.0.1:8000/docs**

## Endpoints principais

| Método | URL | Descrição |
|---|---|---|
| `GET` | `/health` | Status da API e modelo |
| `GET` | `/foods` | Lista alimentos disponíveis |
| `POST` | `/individuals` | Cadastra um indivíduo |
| `GET` | `/individuals/{id}` | Retorna dados do indivíduo |
| `GET` | `/individuals/{id}/top-foods?limit=10` | Melhores alimentos para o indivíduo |
| `POST` | `/score` | Score para um par individual × alimento |

### Exemplo rápido

```bash
# Cadastrar indivíduo
curl -X POST http://localhost:8000/individuals \
  -H "Content-Type: application/json" \
  -d '{
    "name": "João Silva",
    "age": 35,
    "diet_type": "vegetarian",
    "allergies": ["lactose"],
    "total_cholesterol": 210,
    "weight_kg": 70,
    "height_cm": 175,
    "restrictions": ["low_sugar"],
    "goal": "weight_loss",
    "activity_level": "moderately_active",
    "glycemic_condition": "pre_diabetic",
    "hypertension": "none"
  }'

# Usar o individual_id retornado para ver os melhores alimentos
curl "http://localhost:8000/individuals/<id>/top-foods?limit=5"
```

## Documentação

Ver [docs/approach.md](docs/approach.md) para descrição completa da abordagem, modelo e resultados.
