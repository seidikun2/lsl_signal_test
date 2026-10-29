# Resultados locais

Medições em Linux/Python 3.12, entre processos na mesma máquina, sem depurador.
Valores de referência; a carga da máquina e o agendamento afetam cada execução.

| Cenário / sensor | Hz × canais | Recebidas / previstas | Perdidas | Atraso p99 (ms) |
| --- | --- | --- | --- | --- |
| EEG isolado, 10 s | 500 × 8 | 5000 / 5000 | 0 | 32,24 |
| Multissensor EEG, 15 s | 500 × 32 | 7500 / 7500 | 0 | 32,20 |
| Multissensor EMG, 15 s | 2000 × 8 | 30000 / 30000 | 0 | 10,20 |
| Multissensor IMU, 15 s | 100 × 9 | 1500 / 1500 | 0 | 151,80 |
| Extremo EEG, 12 s | 2000 × 64 | 21040 / 24000 | 2960 | 1006,66 |
| Extremo EMG, 12 s | 2000 × 16 | 20912 / 24000 | 3088 | 992,93 |
| Extremo IMU, 12 s | 100 × 9 | 948 / 1200 | 252 | 2152,12 |

RMSE zero em todos os canais das amostras recebidas; isso não significa ausência de perdas.
O extremo usa chunks de 128, buffer de 1 s e pausas de 1,5 s no receptor.
O EEG isolado e o multissensor usam chunks de 16. Os cenários multissensor e extremo
foram executados simultaneamente neste ambiente, portanto não são uma comparação
de desempenho sob carga controlada.

O drift medido no extremo foi **−200,04 ppm (EEG), +100,24 ppm (EMG) e −49,85 ppm (IMU)**,
coerente com os desvios de aquisição injetados. Uma taxa mais alta produz erro de tempo
negativo em relação à taxa nominal. Quatro casos curtos adicionais, a 100/2000 Hz
com chunks 1/128, terminaram sem perda. Os 11 testes unitários passaram.
A campanha completa de 216 casos e o transporte entre duas máquinas não foram executados.

## Como interpretar um novo teste

- **Perdas e RMSE:** confirme completude da execução, contagens de amostras e integridade dos valores.
- **Atraso/p99 e jitter:** incluem agrupamento, filas LSL, agendamento e leitura no Python.
  O p99 é o valor abaixo do qual estão 99% dos atrasos observados.
- **Drift acumulado:** erro de tempo contra a taxa nominal desde a primeira amostra pareada;
  sua inclinação é expressa em ppm. `latency_trend_ppm` é tendência do atraso, não drift de relógio.
- **Sincronização:** compare `sync.png` e os timestamps corrigidos. Igualdade entre
  timestamps transmitidos/recebidos comprova preservação pelo transporte, não acurácia física.

Exemplo: chunk 128 a 100 Hz acrescenta até **1,27 s** de espera à primeira amostra.
A correção de relógio do LSL não elimina esse atraso nem recupera amostras perdidas.
Para validar sensores reais, registre um pulso físico comum nos dispositivos.

Os dados completos ficam em `runs/`, fora do controle de versão. Para comparar
novas execuções, guarde `config.json`, `summary.csv`, `report.json` e os PNGs;
nas varreduras, consulte também `sweep.csv`.
