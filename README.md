# House Price Prediction

Projeto de machine learning para estimar preco de imoveis com base em caracteristicas como area util, numero de quartos, numero de banheiros, idade da casa e distancia ao centro urbano.

O repositorio inclui:
- pipeline de treinamento com geracao de dados sinteticos, EDA e avaliacao de modelos
- API Flask para previsao
- CLI interativa para inferencia local
- frontend estatico com visualizacao de resultado, metricas e importancias

## Estrutura

- `train.py`: gera dados, treina os modelos, avalia e salva artefatos
- `predict.py`: executa previsao via terminal
- `app.py`: expoe `POST /predict` e `GET /model_info`
- `frontend/`: interface web do dashboard
- `dataset.csv`: dataset sintetico gerado pelo treinamento
- `house_price_model.pkl`: modelo treinado
- `model_metadata.json`: metricas e feature importances
- `plots/`: graficos gerados no treinamento
- `tests/`: testes automatizados da API

## Features

- `sqft_living`
- `bedrooms`
- `bathrooms`
- `age_years`
- `distance_to_center_km`

## Como rodar

### 1. Clonar o repositorio

```bash
git clone https://github.com/JoaoVictorAAbreu-Dev/house-price-prediction-model.git
cd house-price-prediction-model
```

### 2. Criar ambiente virtual e instalar dependencias

Use Python 3.12 ou mais recente (validado aqui com Python 3.14). O arquivo
`requirements.txt` fixa as versoes das bibliotecas usadas para gerar o modelo
incluido no repositorio.

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Windows (PowerShell):

```powershell
py -3 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### 3. Treinar o modelo

```bash
python train.py
```

Esse comando gera:
- `dataset.csv`
- `house_price_model.pkl`
- `model_metadata.json`
- arquivos em `plots/`

O modelo incluido no GitHub pode ser usado diretamente apos instalar as
dependencias fixadas. Rode `train.py` para regenerar os artefatos com o seu
ambiente. A API e a CLI recusam um modelo treinado com versoes diferentes das
bibliotecas registradas em `model_metadata.json`, evitando resultados sem
compatibilidade garantida.

O Random Forest e o modelo selecionado para a demonstracao interativa e para o
grafico de importancia das variaveis. A regressao linear tambem e avaliada e
obtem R2 maior neste conjunto sintetico; o Random Forest nao e apresentado
como o modelo mais preciso.

### 4. Rodar a API e o frontend

```bash
python app.py
```

Abra:
- http://127.0.0.1:5000

### 5. Rodar previsao pela CLI

Modo interativo:

```bash
python predict.py
```

Modo com argumentos:

```bash
python predict.py --sqft 2200 --bedrooms 3 --bathrooms 2.5 --age 8 --distance 4.2
```

### 6. Rodar os testes

```bash
pytest
```

## API

### `POST /predict`

Payload:

```json
{
  "sqft_living": 2200,
  "bedrooms": 3,
  "bathrooms": 2.5,
  "age_years": 8,
  "distance_to_center_km": 4.2
}
```

Resposta:

```json
{
  "price": 445267.36,
  "breakdown": {
    "base_value": 100000,
    "size_contribution": 264000,
    "rooms_contribution": 107500,
    "age_depreciation": -6400,
    "location_effect": -12600
  }
}
```

O campo `breakdown` reproduz termos da formula usada para gerar os dados
sinteticos. Nao e uma explicacao da previsao do Random Forest e seus valores
nao precisam somar o campo `price`.

### `GET /model_info`

Retorna:
- melhores hiperparametros do Random Forest
- metricas dos modelos
- feature importances
- modelo selecionado e versoes das dependencias usadas no treinamento

## Configuracao

Variaveis opcionais definidas em `.env.example`:

- `HOST`
- `PORT`
- `FLASK_DEBUG`

## Tecnologias

- Python 3.12+
- Flask
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- pytest

## Observacao

Os dados sao sinteticos e o melhor desempenho neste conjunto tende a ser do modelo linear, porque a regra de geracao e majoritariamente aditiva.
