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

O benchmark curto levou à escolha inicial do lote 12; o ensaio prolongado
abaixo invalidou essa escolha para o loteamento variável anterior. As memórias foram consultadas após cada passo; não são
picos contínuos, nem representam todo o consumo de RAM do sistema.
O [benchmark em JSON](../output/ai/evidence-intensive-benchmark-2026-11-11.json)
preserva os números e o hash do corpus.

A execução inicial `ai/runs/evidence-intensive-2026-11-11/` falhou por memória.
A execução corrigida é `ai/runs/evidence-intensive-bucketed-2026-11-11/`, inicializada de
`evidence-robustness-2026-11-11/best.pt`, SHA-256
`77fbc2da64323c59bd27263768c6d10e09f73910d60e46f1108ce436e530395e`.

- MPS, lote 8, taxa `0.00002`, seed 42.
- Até 10.000 passos ou 1.800 segundos no laço, o que ocorrer primeiro.
- 65% de probabilidade de lotes documentais e 35% de replay de conversa.
- Loss de validação e checkpoints `last.pt`/`best.pt` a cada 250 passos e ao final.
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
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.65 \
PYTORCH_MPS_LOW_WATERMARK_RATIO=0.5 \
caffeinate -i ai/.venv/bin/python -m confere_ai.instruct \
  --data ai/data/private/meu-intensivo/examples.jsonl \
  --init-from ai/runs/evidence-robustness-2026-11-11/best.pt \
  --run-dir ai/runs/meu-intensivo --steps 10000 --batch-size 8 \
  --learning-rate 0.00002 --eval-every 250 --response-tokens 256 \
  --evaluation-context 1024 --document-fraction 0.65 --device mps --max-seconds 1800
ai/.venv/bin/python ai/scripts/evaluate_evidence_robustness.py \
  ai/runs/meu-intensivo --output ai/runs/meu-intensivo/audit.json
```

## Critérios

Documentos exigem 100% de formato/referências válidos e ao menos 90% de acerto
completo em cada perfil. Conversa exige ao menos 80% em cada família, sem queda
por família perante a referência. Falsos atendimentos e a falha conhecida de
conflito na mesma unidade serão conferidos separadamente na API real.

A execução e as avaliações foram concluídas. Os resultados abaixo não atingem
todos os critérios; nenhum checkpoint foi promovido.


## Falha inicial e correção de memória

A primeira execução terminou por falta de memória no MPS. A mensagem registrou
8,59 GiB em tensores, 21,61 GiB em outras alocações e limite de 30,19 GiB.
Somente o checkpoint do passo zero havia sido preservado; o número de atualizações
executadas depois dele é desconhecido. A execução foi marcada como interrompida,
sem atribuir a ela resultados de treino ou qualidade que não foram salvos.

O benchmark inicial repetia poucos passos com o mesmo comprimento. O treino real
variava o comprimento máximo a cada lote. Um teste isolado dos mesmos exemplos
confirmou a associação entre muitos comprimentos distintos e o crescimento de
memória no backend. O [relato upstream de SDPA com comprimentos variáveis](https://github.com/pytorch/pytorch/issues/152550)
foi uma pista para esse diagnóstico, não substitui a medição local.

- Lote 8 com comprimentos livres: atingiu a proteção local de 6,5 GB após
  27 atualizações e 27 comprimentos distintos.
- Agrupamento em faixas de 128 tokens: completou 96 atualizações com cinco
  comprimentos e consumo estável do driver.
- A implementação corrigida completou 512 atualizações nos mesmos lotes, com
  cerca de 3,88 GB no driver e cinco comprimentos. O ensaio usou limite de
  memória equivalente a metade do recomendado pelo dispositivo.

A correção fica no loteador compartilhado: no MPS, com contexto informado,
completa a dimensão até a próxima faixa de 128, limitada à janela do modelo.
Não corta tokens. Posições extras usam PAD e rótulo ignorado pela loss.
Os testes comparam prefixos, alvos, loss e gradientes; CPU mantém o loteamento
anterior. Os três treinadores e a avaliação de loss passam o contexto explícito.
Código da correção: commit `a34c345`.

A nova execução usa limite alto `0.65` e baixo `0.5`, restritos ao processo,
seguindo as [definições oficiais do MPS no PyTorch 2.8](https://docs.pytorch.org/docs/2.8/mps_environment_variables.html).
O limite não foi desativado. As métricas agora registram tanto alocações de
Tensores como memória do driver após a avaliação; esta última inclui reservas
que o contador de tensores não mostra.

Registros locais preservados:

- [Teste com comprimentos livres](../output/ai/evidence-intensive-memory-dynamic-2026-11-11.json).
- [Controle com faixas de comprimento](../output/ai/evidence-intensive-memory-bucketed-2026-11-11.json).
- [Ensaio da implementação corrigida](../output/ai/evidence-intensive-memory-patched-2026-11-11.json).
- `ai/runs/evidence-intensive-2026-11-11/failure.json` e `last.pt`.
- `ai/runs/evidence-intensive-2026-11-11/memory-probe.py`, com o código do ensaio.

Os 132 testes Python passaram após a correção. A execução intensiva corrigida
foi concluída e avaliada conforme os registros seguintes.


## Resultado do ciclo concluído

O laço encerrou por completar os **10.000 passos**, em **996,69 s** (16 min 37 s),
antecipando o limite de 30 minutos. Processou **80.000 exposições a exemplos** e
**4.025.963 tokens supervisionados**, incluindo repetições. Foram 51.936 exemplos
em lotes documentais, com 3.511.230 tokens-alvo, e 28.064 exposições de conversa,
com 514.733 tokens-alvo. Não são quatro milhões de tokens inéditos de corpus.

A memória do driver observada no laço atingiu 11,72 GB; os tensores ativos
atingiram aproximadamente 691 MB e o RSS observado do processo, 1,64 GB.
São contadores distintos e não devem ser somados como memória física total.
O driver inclui reservas e estruturas do backend. Não houve nova interrupção.

A menor loss de validação selecionou o **passo 7.000**. O último checkpoint,
**passo 10.000**, também foi avaliado explicitamente, na mesma coorte e com
janela 1.024 / saída 256. Não se usou o teste reservado.

| Critério | Referência anterior | Menor loss, passo 7.000 | Último, passo 10.000 |
| --- | ---: | ---: | ---: |
| Status e conjunto completo de evidências | 81/120 | 106/120 | 103/120 |
| Formato e referências válidos | 106/120 | 120/120 | 119/120 |
| Falsos atendimentos nesses 120 casos | 2 | 2 | 0 |
| Turnos de conversa, igualdade exata | 65/75 | 68/75 | 66/75 |

O último checkpoint não foi o melhor no acerto completo. A ausência de falsos
atendimentos nessa coorte também não prova segurança: o teste adicional da API
abaixo encontrou um falso atendimento com ambos os pesos.

### Documentos por perfil

| Perfil | Referência | Passo 7.000 | Passo 10.000 |
| --- | ---: | ---: | ---: |
| Múltiplas evidências anteriores | 23/24 | 24/24 | 24/24 |
| Afirmação isolada | 10/12 | 12/12 | 11/12 |
| Negação isolada | 11/12 | 12/12 | 10/12 |
| Mera menção | 12/12 | 11/12 | 11/12 |
| Múltiplas afirmações | 6/12 | 11/12 | 10/12 |
| Múltiplas negações | 6/12 | 8/12 | 9/12 |
| Conflito entre unidades | 3/12 | 6/12 | 9/12 |
| Conflito na mesma unidade | 3/12 | 11/12 | 11/12 |
| Distrator sobre outro tópico | 7/12 | 11/12 | 8/12 |

No passo 7.000, os status isolados acertaram 117/120, mas o resultado completo
acertou 106/120. Seleção de evidências segue sendo um gargalo: uma unidade real
pode ser irrelevante ou não sustentar o status atribuído.

### Conversa

O passo 7.000 não teve quedas por família perante a referência, mas cinco
famílias ficaram abaixo de 80%: definição, tópico de conversa, saudação informal,
negação informal e orientação. Os grupos têm um a quatro turnos e não substituem
uma avaliação ampla de conversa aberta.

O passo 10.000 teve queda de 3/3 para 2/3 em `conversation_slang_praise`, além
de seis famílias abaixo de 80%. O replay maior não autoriza presumir preservação
completa da conversa.

### API com inferência real

Foram executados sete casos autorais, totalizando oito requisitos, com cada
checkpoint pela ponte TypeScript → Python. Ambos acertaram integralmente cinco
dos sete casos, considerando status, formato válido e flags de conflito.

| Caso | Esperado | Ambos os checkpoints |
| --- | --- | --- |
| Afirmação curta | atendido | atendido |
| Negação curta | nao_atendido | nao_atendido |
| Afirmação e negação na mesma unidade, com emoji | insuficiência e conflito | insuficiência e conflito |
| Mesmo conflito em ordem inversa | insuficiência e conflito | insuficiência e conflito |
| Testes presentes / backup ausente, dois requisitos | atendido / nao_atendido | insuficiência / insuficiência, com conflitos indevidos |
| Mera discussão de testes | insuficiência, sem conflito | insuficiência, sem conflito |
| Afirmação no início e negação no fim de texto longo | insuficiência e conflito | atendido, sem conflito |

O caso curto que havia falhado no ciclo anterior foi corrigido nesse controle.
Isso não se estendeu ao texto longo: todo o texto foi processado em quatro lotes,
mas o modelo não selecionou a evidência negativa final e a proposta
consolidada permaneceu positiva. A saída tinha formato e
referências válidos e ainda assim resultou em falso atendimento. No exemplo
com dois requisitos, o modelo confundiu polaridades referentes a assuntos
diferentes. Os offsets Unicode e a cobertura foram conferidos nas duas versões.

Esses casos são de desenvolvimento, com frases e estruturas relacionadas ao
corpus autoral. Não são teste cego, documentos reais ou prova de generalização.

**Os dois candidatos reprovaram os critérios conjuntos. Não houve promoção
para o chat, alteração do ambiente da API ou aprovação persistida de submissões.**
Os seis testes TypeScript passaram, incluindo a ponte HTTP com Python e pesos
reais; essa verificação de software não equivale a aprovação semântica.

## Artefatos finais

- [Configuração, contadores e hashes do treino](../output/ai/evidence-intensive-training-2026-11-11.json).
- [Auditoria do passo 7.000](../output/ai/evidence-intensive-best-2026-11-11.json).
- [Auditoria do passo 10.000](../output/ai/evidence-intensive-last-2026-11-11.json).
- [API com o passo 7.000](../output/ai/evidence-intensive-api-best-2026-11-11.json).
- [API com o passo 10.000](../output/ai/evidence-intensive-api-last-2026-11-11.json).
- [Falha da primeira tentativa](../output/ai/evidence-intensive-failure-2026-11-11.json).
- Relatórios automáticos em `output/pdf/training/evidence-intensive-2026-11-11.pdf`
  e `output/pdf/training/evidence-intensive-bucketed-2026-11-11.pdf`.
- Execução completa em `ai/runs/evidence-intensive-bucketed-2026-11-11/`, incluindo
  `training-summary.json`, avaliação completa e `last-evaluation.json`.

Checkpoint do passo 7.000: SHA-256
`082b4e1f7c2a84c1599f1377c1675a3aa741ed302e339816d9a3ef68ba50cde9`.
Checkpoint do passo 10.000: SHA-256
`b2bea76668ae20c34f033cea2d31f0bcc98c54c210660f39719e526f25682361`.

O próximo ciclo deve reproduzir os lotes efetivos de textos longos e trabalhar
seleção de evidências e separação entre requisitos. Aumentar passos neste mesmo
corpus não substitui corrigir essa diferença entre treino e uso pela API.
