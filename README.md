# Testes de transmissão LSL

Três etapas em Python: gerar/transmitir, receber/gravar e analisar os sinais pelo tempo.

| Arquivo | Responsabilidade |
| --- | --- |
| `gerador.py` | Simula EEG, EMG, IMU e marcadores; transmite e salva os tempos de origem |
| `receptor.py` | Recebe, registra tempos de leitura e salva os dados; inclui alinhamento online |
| `analise.py` | Reconstrói pelos vetores temporais e mostra atraso, perdas e drift |
| `main.py` | Executa as três etapas e organiza varreduras de parâmetros |
| `config.py` | Lê os JSONs e reúne validação e pequenos utilitários compartilhados |

## Começar no VS Code

Abra a pasta do projeto. Com Python 3.12 instalado, no terminal:

```bash
python -m venv .venv
```

Selecione `.venv` em **Python: Select Interpreter** e abra um novo terminal:

```bash
python -m pip install -r requirements.txt
python main.py
```

Isso roda EEG por 10 s, grava e gera as figuras automaticamente.
Também pode selecionar **LSL: EEG** em **Run and Debug** e usar **Ctrl+F5**.
Use F5 para depurar; o depurador altera as medições de tempo.

## Outros testes

Edite taxas, canais, duração e buffers nos JSONs de `configs/`:

```bash
python main.py --config configs/multi.json
python main.py --config configs/extreme.json
python main.py --config configs/drift.json
python main.py sweep --rates 100 2000 --channels 8 32 --chunks 1 128 --repeats 2
```

`chunk` é o agrupamento em amostras; `inlet_buffer` é a retenção em segundos.
`drift_ppm` e `drift_slope_ppm_s` controlam desvio e crescimento da taxa simulada.

Para executar manualmente, use dois terminais com a mesma configuração:

```bash
# Terminal 1: gerador (espera a conexão do receptor)
python gerador.py
# Terminal 2: receptor
python receptor.py
# Quando ambos terminarem: análise
python analise.py runs/rx01 --tx-dir runs/tx01
```

Os padrões são `configs/single.json`, ID `teste` e pastas `runs/tx01`/`runs/rx01`.
Nas próximas execuções, escolha novas pastas com `--out`; dados não são sobrescritos.

## Resultados

O `main.py` cria uma pasta em `runs/`. Abra `SENSOR_tempo.png`, `sync.png` e
`diagnostico.md`: mostram reconstrução, atraso, drift acumulado e alertas.
`summary.csv` resume as métricas; os NPZ guardam sinais e tempos para outras análises.
**Valores intactos não significam chegada no tempo correto:** a análise compara
cada sinal usando seu próprio vetor temporal, sem remover o atraso.

[Imagens de exemplo e resultados medidos](docs/resultados.md).

Os dados ficam em RAM até a gravação final. São medições de software, não de
latência física de sensores. Testes automatizados: `python -m pytest -q`.
