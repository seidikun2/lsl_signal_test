# Testes de transmissão LSL

Simula EEG, EMG e IMU com marcadores, transmissor e receptor LSL separados.
Varia taxas, canais e buffers; salva sinais, gráficos e métricas de sincronização.

## Executar no VS Code

1. Abra **a pasta deste projeto** no VS Code e instale as extensões recomendadas **Python** e **Python Debugger**.
2. Com Python 3.12 instalado, crie o ambiente no terminal: `python -m venv .venv` (ou `python3` no Linux).
3. Use **Python: Select Interpreter** na paleta de comandos e selecione `.venv`.
4. Use **Tasks: Run Task → Instalar dependências**. A tarefa usa o Python selecionado.
5. Abra **Run and Debug**, escolha **LSL: EEG**, **LSL: Multissensor** ou **LSL: Extremo** e execute.

Use **Ctrl+F5** para medir sem depurador e **F5** para investigar o código.
Breakpoints e o depurador alteram a temporização. Os testes unitários aparecem
na aba **Testing**; também há a tarefa **Testar código**.

## Configurar e rodar

Edite os JSONs em `configs/`:

| Arquivo | Cenário |
| --- | --- |
| `single.json` | EEG: 500 Hz, 8 canais |
| `multi.json` | EEG, EMG e IMU com taxas e canais diferentes |
| `extreme.json` | Buffer curto, receptor com pausas e drift por sensor |

Os parâmetros principais são `rate` (Hz), `channels`, `duration` (s),
`chunk` (amostras por envio), `inlet_buffer` (segundos de retenção) e
`drift_ppm` (desvio simulado da taxa, global ou por sensor).

No terminal com o ambiente ativado:

```bash
python -m lsltest run --config configs/multi.json
python -m lsltest sweep --rates 100 500 1000 2000 --channels 1 8 32 --chunks 1 16 128 --repeats 3 --duration 10
python -m lsltest analyze runs/NOME_DO_TESTE
```

Cada execução cria uma pasta em `runs/`. Na varredura, `f2000c32b128r2`
significa 2000 Hz, 32 canais, chunk 128 e repetição 2. A tarefa **LSL: Varredura curta**
no VS Code executa quatro casos de validação. Pastas existentes não são sobrescritas.

## Ver os resultados

Abra os PNGs diretamente no VS Code; o CSV permite comparar experimentos.

| Saída em cada pasta | O que observar |
| --- | --- |
| `SENSOR.png` | Sinal transmitido/recebido, atraso, jitter e drift acumulado |
| `sync.png` | Diferença temporal entre amostras e marcadores por sensor |
| `summary.csv` / `report.json` | Perdas, duplicações, RMSE, p95/p99 e drift em ppm |
| `tx_*.npz` / `rx_*.npz` | Valores e timestamps para outras análises |
| `online.npz` | Sensores alinhados numa grade temporal comum |
| `config.json` | Parâmetros usados naquele teste |

**Resultados já medidos e interpretação:** [resultados.md](docs/resultados.md).

A sincronização online usa a correção de relógio do LSL e interpolação causal;
lacunas viram `NaN`. Um pulso comum e os marcadores permitem comparar os sensores.
O transporte inclui um canal extra de índice para detectar perdas; ele é removido dos sinais salvos.

O atraso inclui agrupamento, filas e leitura pelo Python; não mede apenas a rede.
Os dados ficam em RAM até a gravação final: divida campanhas longas em casos menores.
São testes de software, sem validação física dos sensores. Seu transmissor existente
poderá ser integrado em `lsltest/transport.py`.

Para instalar sem VS Code: ative um ambiente virtual e execute
`python -m pip install -e ".[test]"`. O `requirements.lock` registra as versões
validadas em Python 3.12/Linux; em outras plataformas, `pylsl` pode exigir `liblsl`.
