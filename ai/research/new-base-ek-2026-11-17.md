# Base completa e classificador EK

Primeiro classificador sobre `pt-full-8k-2026-11-17` (ver `PRETRAINING.md`),
com pesos iniciados da base causal nova, sem herdar a linhagem CI. Nenhum peso
externo; nenhum padrão da API alterado; reserva DG de 54 pares sem inferência.

## Dados e treino

`prepare_evidence_mix.py`: 15 mil casos de controle CC amostrados, 6.567
relações DF de treino (InferBR/FaQuAD) e a matriz DG; validação = validação DG
(51 pares acadêmicos + 410 casos antigos), a mesma de DG a DJ. Encoder,
embeddings e cabeça treinados (taxas 1e-4 e 1e-3), batch 8, unidade inteira.
Parou por platô no passo 23.000 (23 min no MPS); selecionado o passo 15.000.
A curva de validação oscilou muito (206 a 359 de 461), sinal de taxa alta.

## Resultados de desenvolvimento

| Medida | Origem CI | DI selecionado | EK selecionado |
| --- | ---: | ---: | ---: |
| Acadêmico suporte | 4/9 | 2/9 | 4/9 |
| Acadêmico contradição | 2/8 | 1/8 | 3/8 |
| Acadêmico insuficiente | 3/34 | 24/34 | 16/34 |
| Falsas aprovações acadêmicas | 16 | 8 | 12 |
| Casos antigos | 407/410 | — | 324/410 |

Teste DF público (861, nunca usado em treino ou seleção), resposta completa:
origem 322, candidato DF 521, **EK 583**. O controle que lê só a consulta
acerta 586. O ganho é grande sobre a linhagem anterior, mas **não supera o
atalho da consulta**: não há prova de leitura da evidência.

TCCs conhecidos, BM25 com k=8 (mesmo protocolo da CI com k=8):

| | CI | EK |
| --- | ---: | ---: |
| Status coincidentes | 8/18 | 11/18 |
| Falsas aprovações | 3 | 5 |
| Trechos marcados como atende e não atende ao mesmo tempo | 67/144 | 25/144 |

## Conclusão

A base nova torna o classificador mais discriminativo (menos marcações duplas,
mais acertos em suporte/contradição), mas aumenta falsas aprovações, perde casos
antigos e empata com o atalho da consulta no DF. Reprovado; não promovido.

O próximo gargalo é a supervisão: no DF a consulta prevê o rótulo. É preciso
dados de relação em que a mesma consulta apareça com evidências de suporte,
contradição e neutralidade (InferBR permite agrupar por premissa), para que o
atalho deixe de funcionar, e taxa menor para estabilizar a seleção.

Rastros: `output/ai/evidence-mix-ek-2026-11-17.json`.
