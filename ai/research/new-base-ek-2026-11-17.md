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

## EL: contrastes sem atalho

`prepare_relation_contrasts.py` deriva de cada par InferBR do DF, no mesmo
split, a negação da hipótese com rótulo invertido e premissas vizinhas por BM25
sem relação (insuficiente). 617 derivados quase iguais a frases de outro split
foram removidos. No teste de contrastes (3.262), ler só a consulta acerta 2.115
e só o trecho 2.019, abaixo da classe majoritária (2.166).

EL usa a mesma mistura de EK trocando DF pelos contrastes (41.429 linhas de
treino) e taxas menores (3e-5 encoder, 3e-4 cabeça). Curva estável; platô no
passo 32.000, selecionado o 22.000.

| Medida | EK | EL |
| --- | ---: | ---: |
| Teste de contrastes (majoritária 2.166) | 1.328 | **2.657** |
| — suporte / contradição / insuficiente | 195/537 · 439/559 · 694/2.166 | 349/537 · 388/559 · 1.920/2.166 |
| Teste DF público (só consulta: 586) | 583 | **609** |
| Validação total (461) | 347 | 427 |
| Acadêmico suporte / contradição / insuficiente | 4/9 · 3/8 · 16/34 | 4/9 · 4/8 · 21/34 |
| Falsas aprovações acadêmicas | 12 | 6 |
| Casos antigos | 324/410 | 398/410 |

É o primeiro classificador que supera os controles de atalho nos dois testes
públicos: há evidência de leitura do trecho, ao menos em frases curtas.

TCCs com BM25 k=8: 6/18 status coincidentes e 4 falsas aprovações (EK: 11 e 5;
CI: 8 e 3). Classificando frase a frase (sem retreino), 11/18 com 7 falsas
aprovações. A transferência para documentos reais não acompanhou: os requisitos
são normativos ("O relatório deve identificar...") e as unidades são páginas,
enquanto a supervisão forte é de frases declarativas curtas. A matriz acadêmica,
única supervisão nesse formato, tem 90 pares. EL reprovado; não promovido.
Rastros: `output/ai/evidence-mix-el-2026-11-17.json`.
