# Abordagem Técnica — Nutritional Score Engine

## 1. Visão Geral

O objetivo deste projeto é atribuir um **score de compatibilidade nutricional (0–10)** para cada par *(indivíduo, alimento)*. Um score alto indica que aquele alimento é bem adequado às características, restrições e objetivos daquele indivíduo; um score baixo indica incompatibilidade.

A abordagem segue três pilares:

1. **Dados de alimentos reais** — coletados via API pública (Open Food Facts)
2. **Dados de indivíduos sintéticos** — gerados com distribuições estatisticamente plausíveis
3. **Modelo MLP supervisionado** — treinado sobre scores calculados por uma função heurística clínica

---

## 2. Dados de Indivíduos

### 2.1 Features (11 campos)

| Campo | Tipo | Distribuição / Valores |
|---|---|---|
| `age` | int | Normal(μ=40, σ=15) clipped [18, 80] |
| `diet_type` | categórico | omnivore 50%, vegetarian 20%, vegan 10%, keto 10%, pescatarian 8%, paleo 2% |
| `total_cholesterol` | int (mg/dL) | correlacionado com idade e IMC; Normal(base, σ=20) clipped [120, 320] |
| `weight_kg` | float | Normal(75, 15) clipped [45, 150] |
| `height_cm` | float | Normal(170, 10) clipped [150, 200] |
| `bmi` | float | derivado: weight / (height/100)² |
| `allergies` | multi-hot | Bernoulli por alérgeno: gluten 2%, lactose 15%, nuts 3%, shellfish 4%, eggs 1%, soy 2% |
| `restrictions` | multi-hot | correlacionadas com condições de saúde (ex: low_sodium ↔ hypertension) |
| `goal` | categórico | weight_loss 35%, maintenance 25%, muscle_gain 20%, health_improvement 15%, energy_boost 5% |
| `activity_level` | categórico | lightly_active 35%, moderately_active 30%, sedentary 20%, very_active 15% |
| `glycemic_condition` | categórico | none 84%, pre_diabetic 10%, type_2 5%, type_1 1% — piorado por IMC alto |
| `hypertension` | categórico | none 70%, controlled 20%, uncontrolled 10% — piorado por colesterol alto e idade |

### 2.2 Correlações realistas

A geração não é totalmente independente. Exemplos de correlações intencionais:

- Indivíduos com **IMC > 30** têm maior probabilidade de `glycemic_condition != none`
- Indivíduos com **colesterol > 240** ou **idade > 55** têm maior probabilidade de `hypertension != none`
- `restriction_low_sodium` é fortemente associada a `hypertension`
- `restriction_low_sugar` é fortemente associada a `glycemic_condition`
- `restriction_high_protein` é fortemente associada a `goal == muscle_gain`

### 2.3 Arquivo gerado

`data/processed/individuals.csv` — 1 000 linhas, 23 colunas

---

## 3. Dados de Alimentos

### 3.1 Fonte

**Open Food Facts** ([world.openfoodfacts.org](https://world.openfoodfacts.org))
- API REST pública, sem chave de acesso
- Cobertura: > 3 milhões de produtos globalmente
- Endpoint utilizado: `GET /cgi/search.pl` com `action=process&json=1`

### 3.2 Estratégia de coleta

Definimos 12 categorias de alimentos (`grain`, `legume`, `vegetable`, `fruit`, `dairy`, `meat`, `fish`, `egg`, `nut`, `processed`, `beverage`, `snack`) com 6–9 termos de busca por categoria. Para cada termo, coletamos até 8 produtos, filtramos os que têm valor calórico disponível, e limitamos a 25 por categoria.

### 3.3 Features extraídas

| Campo | Descrição | Unidade |
|---|---|---|
| `energy_kcal_100g` | calorias | kcal/100g |
| `proteins_100g` | proteínas | g/100g |
| `carbohydrates_100g` | carboidratos totais | g/100g |
| `fat_100g` | gordura total | g/100g |
| `saturated_fat_100g` | gordura saturada | g/100g |
| `fiber_100g` | fibra alimentar | g/100g |
| `sodium_mg_100g` | sódio | mg/100g |
| `sugar_100g` | açúcares | g/100g |
| `contains_{alérgeno}` | presença de alérgeno | booleano |
| `is_animal_product` | produto animal | booleano |
| `is_meat` | carne | booleano |
| `is_fish` | peixe/frutos do mar | booleano |

### 3.4 Tratamento de valores ausentes

Valores nutricionais ausentes são imputados pela **mediana do grupo** (ex: mediana de `proteins_100g` dentro de `grain`). Caso toda a categoria tenha valor ausente, usa-se a mediana global. Produtos sem valor calórico são descartados.

### 3.5 Arquivo gerado

`data/processed/foods.csv` — ~250 linhas, 20 colunas

---

## 4. Função de Score Heurística

A função heurística gera os **labels de treino** (ground truth) para o modelo supervisionado. Ela simula o raciocínio de um nutricionista de forma parametrizada.

### 4.1 Lógica (em pseudocódigo)

```
score = 10.0

# 1. Alérgenos (hard constraint)
se alimento contém alérgeno do indivíduo:
    retornar 0.0

# 2. Compatibilidade com dieta
se diet == "vegan" e alimento animal:        score -= 4.0
se diet == "vegetarian" e carne/peixe:       score -= 3.0
se diet == "pescatarian" e carne:            score -= 2.0
se diet == "keto" e carbs > 10g:            score -= min(3.0, (carbs-10) * 0.1)
se diet == "paleo" e (laticínio|grão|legume|processado): score -= 2.0

# 3. Alinhamento com objetivo
se goal == "weight_loss":
    se energy > 400 kcal:       score -= 1.5
    se energy < 200 kcal:       score += 0.5
    se fiber > 5g:              score += 0.5
    se protein > 15g:           score += 0.5
    se sugar > 15g:             score -= 0.5
se goal == "muscle_gain":
    se protein >= 20g:          score += 2.0
    se protein >= 10g:          score += 0.8
    se energy > 250 kcal:       score += 0.5
se goal == "health_improvement":
    se fiber > 5g:              score += 1.0
    se sugar < 5g:              score += 0.5
    se grupo é vegetal/fruta/legume: score += 0.5
se goal == "energy_boost":
    se 200 ≤ energy ≤ 350 kcal: score += 1.0

# 4. Condições de saúde
se hypertension == "uncontrolled":
    se sodium > 400mg:          score -= 3.0
    se sodium > 200mg:          score -= 1.5
se hypertension == "controlled":
    se sodium > 400mg:          score -= 2.0
se glycemic_condition != "none":
    se sugar > 15g:             score -= 2.0
    se sugar > 8g:              score -= 1.0
se total_cholesterol > 240:
    se saturated_fat > 7g:      score -= 1.5

# 5. Restrições explícitas
se restriction_low_sodium e sodium > 200mg:      score -= 1.0
se restriction_low_sugar e sugar > 5g:           score -= 1.0
se restriction_low_fat e fat > 15g:              score -= 1.0
se restriction_high_protein e protein < 10g:     score -= 1.0
se restriction_low_carb e carbs > 20g:           score -= 1.5

score += N(0, 0.3)   # ruído realista
score = clip(score, 0, 10)
```

### 4.2 Distribuição dos scores gerados

Com 1000 indivíduos × ~250 alimentos (~250 000 pares):
- A distribuição é **multimodal**: pico em ~0 (pares incompatíveis) e pico em ~7-8 (pares compatíveis)
- Scores = 0 ocorrem por alérgenos ou incompatibilidade de dieta
- A média situa-se tipicamente em ~5–6

---

## 5. Pipeline de Features

O pipeline é implementado com `sklearn.compose.ColumnTransformer`:

```
ColumnTransformer
├── ind_num   → StandardScaler     → [age, total_cholesterol, weight_kg, height_cm, bmi]
├── food_num  → StandardScaler     → [energy_kcal_100g, proteins_100g, carbohydrates_100g,
│                                      fat_100g, saturated_fat_100g, fiber_100g,
│                                      sodium_mg_100g, sugar_100g]
├── ind_cat   → OneHotEncoder      → [diet_type, goal, activity_level,
│                                      glycemic_condition, hypertension]
├── food_cat  → OneHotEncoder      → [food_group]
└── bin       → passthrough        → [allergy_*, restriction_*, contains_*,
                                       is_animal_product, is_meat, is_fish]
```

**Dimensão de entrada ao MLP**: ~55–60 features (varia com número de categorias distintas encontradas)

O preprocessor é ajustado (`fit`) apenas nos dados de treino para evitar data leakage.

---

## 6. Arquitetura do MLP

### 6.1 Configuração

```python
MLPRegressor(
    hidden_layer_sizes = (128, 64, 32),
    activation         = "relu",
    solver             = "adam",
    alpha              = 0.01,        # regularização L2
    batch_size         = 256,
    learning_rate      = "adaptive",
    learning_rate_init = 1e-3,
    max_iter           = 500,
    random_state       = 42,
)
```

### 6.2 Justificativas

| Escolha | Justificativa |
|---|---|
| 3 camadas (128→64→32) | Complexidade suficiente para aprender interações não-lineares entre saúde e nutrição sem overfit |
| ReLU | Gradientes estáveis, sem saturação em redes rasas |
| Adam | Adaptativo, converge bem em features heterogêneas (booleanos + contínuos) |
| alpha=0.01 | Regularização moderada; o ruído σ=0.3 já suaviza o target |
| Early stopping manual | Controle preciso sobre o val set (split por indivíduo) |

### 6.3 Split de dados

O split é feito **por indivíduo**, não por par aleatório:

- **80%** dos indivíduos → treino
- **10%** dos indivíduos → validação (early stopping)
- **10%** dos indivíduos → teste (avaliação final)

Isso garante que a avaliação mede **generalização para novos indivíduos**, não apenas interpolação.

---

## 7. Resultados

| Métrica | Valor |
|---|---|
| MAE (test) | 0.1995 |
| RMSE (test) | 0.4084 |
| R² (test) | **0.9691** |

*Modelo treinado com 291 alimentos reais do Open Food Facts (291 000 pares).*

### 7.1 Interpretação esperada

Como o target é uma função quase-determinística (heurística + σ=0.3), espera-se:
- **R² > 0.85**: o MLP aprende a função heurística com boa precisão
- **MAE < 0.8**: erro médio de menos de 1 ponto na escala 0–10
- Erros maiores em regiões de transição (ex: pares borderline entre score 0 e score 3)

### 7.2 Gráficos gerados

Salvos em `docs/`:
- `evaluation_plots.png` — curva de loss, predicted vs actual, distribuição de resíduos
- `score_by_food_group.png` — boxplot de scores por grupo alimentar
- `score_by_goal.png` — boxplot de scores por objetivo do indivíduo

---

## 8. Como Executar

### Pré-requisitos

```bash
pip install -r requirements.txt
```

### Pipeline completo

```bash
python run_pipeline.py
```

Se a busca no Open Food Facts foi feita anteriormente:

```bash
python run_pipeline.py --skip-fetch
```

### API

```bash
uvicorn src.api.main:app --reload
# Swagger UI: http://127.0.0.1:8000/docs
```

### Exemplo de uso da API

```bash
# 1. Cadastrar indivíduo
RESPONSE=$(curl -s -X POST http://localhost:8000/individuals \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Maria Fernanda",
    "age": 42,
    "diet_type": "vegan",
    "allergies": ["gluten"],
    "total_cholesterol": 185,
    "weight_kg": 65,
    "height_cm": 168,
    "restrictions": ["low_sugar", "low_fat"],
    "goal": "health_improvement",
    "activity_level": "moderately_active",
    "glycemic_condition": "none",
    "hypertension": "none"
  }')
ID=$(echo $RESPONSE | python -c "import sys,json; print(json.load(sys.stdin)['individual_id'])")

# 2. Ver top 10 alimentos
curl "http://localhost:8000/individuals/$ID/top-foods?limit=10"

# 3. Score pontual
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d "{\"individual_id\": \"$ID\", \"food_id\": \"<food_id>\"}"
```

---

## 9. Ferramentas Utilizadas

| Ferramenta | Versão mínima | Papel |
|---|---|---|
| `scikit-learn` | 1.4 | MLPRegressor, ColumnTransformer, StandardScaler, OneHotEncoder |
| `pandas` | 2.0 | Manipulação de dados |
| `numpy` | 1.26 | Operações numéricas e geração de dados sintéticos |
| `fastapi` | 0.111 | Framework da API REST |
| `uvicorn` | 0.29 | ASGI server |
| `pydantic` | 2.0 | Validação de esquemas da API |
| `requests` | 2.31 | Cliente HTTP para Open Food Facts |
| `matplotlib` + `seaborn` | 3.8 / 0.13 | Visualizações de avaliação |
| `joblib` | 1.4 | Serialização de artefatos do modelo |

---

## 10. Limitações e Próximos Passos

### Limitações do MVP

| Limitação | Impacto |
|---|---|
| Scores gerados por heurística | O modelo aprende uma função sintética, não dados clínicos reais |
| Dados de alimentos podem ter campos ausentes | Imputação por mediana introduz viés em nutrientes raros |
| Armazenamento de indivíduos em memória | Dados perdidos ao reiniciar a API |
| Ausência de índice glicêmico | Relevante para controle glicêmico, não disponível no OFF |
| Score único (0–10) | Não captura trade-offs multi-dimensionais |

### Possíveis melhorias

- **Dados clínicos reais**: usar datasets como NHANES (CDC) para treino mais fidedigno
- **Índice glicêmico**: integrar base de dados dedicada (ex: Glycemic Index Foundation)
- **Score multi-dimensional**: separar score por eixo (calórico, alérgico, macro, micro)
- **Persistência**: substituir dict em memória por banco de dados (PostgreSQL, SQLite)
- **Explicabilidade**: SHAP values para detalhar contribuição de cada feature
- **Feedback de usuário**: coletar avaliações reais para fine-tuning do modelo
- **Porções reais**: considerar quantidade consumida, não apenas por 100g
