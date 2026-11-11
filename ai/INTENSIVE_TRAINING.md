# Ciclo intensivo de evidências — 11/11/2026

Continuação autorizada do [ciclo de robustez](EVIDENCE_ROBUSTNESS.md), usando os
mesmos pesos, arquitetura e tokenizer locais. O treino amplia volume e variedade
de exemplos; não muda o contrato da API ou o checkpoint padrão do chat.

## Corpus e verificações

`prepare_evidence_intensive.py` acrescenta pares contrafactuais autorais de
afirmação, negação, menção, múltiplas evidências e conflitos. Também varia ordem,
posição, distratores, tamanho do texto e prefixos Unicode. Afirmação e negação
podem aparecer na mesma unidade, inclusive no formato textual recebido pela API.

O replay estratificado inclui conversas completas, com o histórico original,
selecionadas por família. O corpus final tem:

- 20.362 exemplos de treino: 14.257 documentais e 6.105 turnos de conversa.
- Os mesmos 195 exemplos de validação: 120 documentais e 75 turnos de conversa.
- Maior sequência supervisionada de 756 tokens, com contexto de 1.024 e reserva
  de até 256 tokens para resposta, sem truncamento.
- Nenhum exemplo reservado de teste ou documento real incorporado.

Rótulos são autorais por construção, sem revisão independente. A validação já
foi inspecionada nos ciclos anteriores e não é um teste cego. Duplicatas de
contexto documental são removidas; famílias e históricos preservam seus splits.

A primeira geração foi recusada na checagem de contexto por conter trechos de
fundo muito repetitivos. O gerador foi ajustado e passou a verificar o orçamento
com o tokenizer do checkpoint antes de gravar o corpus. Os diretórios anteriores
foram preservados e não foram usados no treino intensivo.

Corpus efetivamente usado: `ai/data/private/evidence-intensive-ready-2026-11-11/`.
SHA-256: `7b3ca6a5bc3236b49391839465d5b9960523f1df939708a8d1fb713cead3ebee`.
Código do gerador: commit `367ebef`.

## Dimensionamento e execução

Mac com 24 GiB de memória unificada. O benchmark fez seis atualizações por lote,
com duas de aquecimento e quatro medidas, usando os exemplos de treino mais
longos, com 741 tokens. É uma medição curta, sem promessa de velocidade constante.

| Lote | Exemplos/s medidos | Memória do driver observada |
| --- | ---: | ---: |
| 4 | 29,6 | 2,41 GB |
| 8 | 45,9 | 3,75 GB |
| 12 | 49,9 | 5,12 GB |

Foi escolhido o lote 12. As memórias foram consultadas após cada passo; não são
picos contínuos, nem representam todo o consumo de RAM do sistema.
O [benchmark em JSON](../output/ai/evidence-intensive-benchmark-2026-11-11.json)
preserva os números e o hash do corpus.

Execução `ai/runs/evidence-intensive-2026-11-11/`, inicializada de
`evidence-robustness-2026-11-11/best.pt`, SHA-256
`77fbc2da64323c59bd27263768c6d10e09f73910d60e46f1108ce436e530395e`.

- MPS, lote 12, taxa `0.00002`, seed 42.
- Até 6.000 passos ou 1.800 segundos no laço, o que ocorrer primeiro.
- 65% de probabilidade de lotes documentais e 35% de replay de conversa.
- Loss de validação e checkpoints `last.pt`/`best.pt` a cada 500 passos e ao final.
- Pesos das execuções anteriores preservados; nenhum peso entra no Git.
- `caffeinate -i` acompanha somente o processo de treinamento, evitando suspensão
  por inatividade enquanto ele roda, sem mudar configurações persistentes.

A janela de 30 minutos cobre o laço de treino e suas medições de loss. Preparação,
avaliação gerativa inicial/final e geração do relatório ficam fora desse prazo.
O checkpoint com menor loss não é automaticamente o melhor na qualidade semântica.

## Reprodução

Use pastas novas para preservar os resultados existentes:

```sh
ai/.venv/bin/python ai/scripts/prepare_evidence_intensive.py \
  ai/data/private/meu-intensivo \
  --source ai/data/private/evidence-robustness-2026-11-11/examples.jsonl \
  --replay-source ai/data/private/document-long-contrasts-v2-2026-11-04/examples.jsonl \
  --checkpoint ai/runs/evidence-robustness-2026-11-11/best.pt
caffeinate -i ai/.venv/bin/python -m confere_ai.instruct \
  --data ai/data/private/meu-intensivo/examples.jsonl \
  --init-from ai/runs/evidence-robustness-2026-11-11/best.pt \
  --run-dir ai/runs/meu-intensivo --steps 6000 --batch-size 12 \
  --learning-rate 0.00002 --eval-every 500 --response-tokens 256 \
  --evaluation-context 1024 --document-fraction 0.65 --device mps --max-seconds 1800
ai/.venv/bin/python ai/scripts/evaluate_evidence_robustness.py \
  ai/runs/meu-intensivo --output ai/runs/meu-intensivo/audit.json
```

## Critérios

Documentos exigem 100% de formato/referências válidos e ao menos 90% de acerto
completo em cada perfil. Conversa exige ao menos 80% em cada família, sem queda
por família perante a referência. Falsos atendimentos e a falha conhecida de
conflito na mesma unidade serão conferidos separadamente na API real.

Resultados de qualidade ainda pendentes durante a execução; nenhum candidato
é promovido apenas por mais passos ou menor loss.
