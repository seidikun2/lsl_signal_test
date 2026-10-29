# Teste de transmissão LSL

**Configure em `configuracao.py`. Execute `main.py`. Veja o sumário e as figuras.**

```bash
python -m pip install -r requirements.txt
python main.py
```

No VS Code, selecione seu interpretador Python, abra `main.py` e use **Run Python File**.
Para atualizar uma cópia já clonada: `git pull`.

## Configuração

Em `configuracao.py`, edite a duração, a frequência (`rate`) e os canais (`channels`)
de EEG, EMG e IMU. `chunk` é o agrupamento em amostras; `inlet_buffer` é a retenção
em segundos. Remova da lista os sensores que não quiser testar.

Para testar combinações de frequências, canais e chunks, altere `VARREDURA["ativar"]`
para `True` e ajuste as listas no mesmo arquivo. O comando continua `python main.py`.

## Arquivos e resultados

```text
main.py                  ← executar
configuracao.py          ← editar
codigo/                  ← gerador, receptor, análise e testes
resultados/
  data_do_teste/
    sumario.md           ← abrir primeiro: métricas e problemas detectados
    sumario.csv          ← tabela para comparar resultados
    figuras/             ← reconstrução, atrasos, drift e marcadores
    dados/               ← sinais e vetores temporais gravados
```

[Exemplo de sumário e figuras](resultados/exemplo/sumario.md).
No VS Code, **Ctrl+Shift+V** mostra o sumário formatado.

A reconstrução usa o vetor temporal de cada sinal, sem remover o atraso.
Valores intactos podem chegar atrasados. Os limites de alerta são editáveis;
as medidas são de software, não de latência física de sensores.
Os dados ficam em RAM até a gravação: divida campanhas longas em testes menores.
