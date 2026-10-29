# Sumário de exemplo

Resultados locais de simulações LSL. Abra as figuras pelos links abaixo.

| Teste | Resultado | Figura |
| --- | --- | --- |
| EEG normal, 500 Hz / 8 canais | 5000/5000 amostras; atraso p99 32,24 ms | [Normal](figuras/eeg_normal.png) |
| EEG extremo, 2000 Hz / 64 canais | 2960 perdas; atraso p99 1006,66 ms | [Perdas e atrasos](figuras/eeg_extremo.png) |
| Drift crescente, 500 Hz / 4 canais | 6000/6000 intactas; drift acumulado −71,11 ms | [Drift crescente](figuras/drift_crescente.png) |
| Marcadores no extremo | Timestamps e disponibilidade no receptor | [Marcadores](figuras/sync_extremo.png) |

Mesmo com RMSE por índice igual a zero, o sinal pode chegar fora de tempo.
As figuras comparam as curvas pelos seus vetores temporais e mostram os alertas.
O drift crescente deste exemplo foi injetado na aquisição, não causado pela rede.
