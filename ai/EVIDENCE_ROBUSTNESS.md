# Evidências difíceis e replay de conversa — 11/11/2026

Continuação na branch `ai/treinamento-integracao-api`, após o ciclo de
[múltiplas evidências](MULTIPLE_EVIDENCE.md). Sem promoção de pesos.

## Corpus e execução

O novo gerador preserva os exemplos V2, acrescenta oito perfis de evidência e
seleciona conversas completas do corpus anterior, com seus históricos e splits.
Não incorpora documentos reais, fontes externas ou exemplos reservados de teste.
Rótulos documentais são autorais por construção, sem revisão independente.

- Treino: 2.744 exemplos, sendo 2.160 documentais e 584 turnos de conversa.
- Validação: 195 exemplos, sendo 120 documentais e 75 turnos de conversa.
- Os 24 casos documentais anteriores foram preservados; 96 casos acrescentam
  distratores, menções sem comprovação, múltiplas evidências e conflitos dentro
  da mesma unidade. A validação permanece de desenvolvimento, sem cegamento.
- Maior sequência supervisionada: 756 tokens, sem truncamento na janela de 1.024.
- Inicialização: `multiple-evidence-v2-2026-11-11/best.pt`.
- Execução: `evidence-robustness-2026-11-11`, MPS, batch 4, taxa `0.00003`,
  65% de probabilidade de lotes documentais e 35% de replay de conversa.
- Limite do laço: 240 segundos. Encerrou no passo 589, também selecionado pela
  menor loss de validação. Avaliação inicial, avaliação gerativa final e relatório
  ficam fora desse limite; quatro minutos não representam o tempo total do ciclo.
- Comparação pareada: janela 1.024 e saída de até 256 tokens para os dois pesos.

Corpus SHA-256:
`0388ff6a1664082d3489e744606d454bd33d09bb29684afafb9db8fdb3a3e51c`.
Pesos selecionados SHA-256:
`77fbc2da64323c59bd27263768c6d10e09f73910d60e46f1108ce436e530395e`.

## Resultados separados

Documentos: 22/120 na referência e 81/120 no candidato. Formato e referências
válidos em 106/120. O acerto exige status e conjunto completo de evidências.

| Perfil documental | Referência | Candidato | Formato e referências válidos |
| --- | ---: | ---: | ---: |
| Múltiplas evidências anteriores | 22/24 | 23/24 | 23/24 |
| Afirmação isolada | 0/12 | 10/12 | 12/12 |
| Negação isolada | 0/12 | 11/12 | 12/12 |
| Mera menção | 0/12 | 12/12 | 12/12 |
| Múltiplas afirmações | 0/12 | 6/12 | 9/12 |
| Múltiplas negações | 0/12 | 6/12 | 10/12 |
| Conflito entre unidades | 0/12 | 3/12 | 11/12 |
| Conflito na mesma unidade | 0/12 | 3/12 | 7/12 |
| Distrator sobre outro tópico | 0/12 | 7/12 | 10/12 |

Houve dois falsos atendimentos nos 12 casos de conflito na mesma unidade.
Algumas saídas selecionam os dois status opostos, mas declaram `atendido`
no campo global; o validador as recusa e devolve insuficiência. Outras omitem
a negação, selecionam uma unidade errada ou repetem o ponteiro positivo.
Referência válida continua sem garantir suporte semântico.

Conversa: 55/75 na referência e 65/75 no candidato, com histórico gerado por
cada versão. Apesar do total maior, `conversation_slang_hello` e
`conversation_slang_negation` caíram de 2/3 para 1/3 cada. Seis famílias ficaram
abaixo de 80%; os grupos têm somente um a quatro turnos cada. Esse recorte não
substitui a avaliação completa dos conjuntos de conversa anteriores.

Critérios: 100% de formato/referências válidos, ao menos 90% por perfil
documental, ao menos 80% por família de conversa e nenhuma queda por família
de conversa. **Ambos os objetivos reprovaram. Candidato não adotado.**

O código de aceitação mantém a regra anterior para avaliações sem perfis e
passa a exigir o mínimo separadamente quando há `evaluation_profile`.
O auditor também bloqueia aprovação sem algum dos dois objetivos ou com
coortes diferentes entre referência e candidato.

## Inferência real na API

Os seis testes TypeScript passaram com o teste local opcional habilitado.
Esse teste usa o adaptador HTTP em memória, sessão sintética e subprocesso
Python real. Não acessa banco, login real ou servidor de rede.

Uma checagem semântica adicional pelo serviço da API registrou:

| Texto autoral | Esperado | Retornado |
| --- | --- | --- |
| O projeto inclui testes. | atendido | atendido |
| O projeto não inclui testes. | nao_atendido | nao_atendido |
| 🙂 O projeto inclui testes. O projeto não inclui testes. | evidencia_insuficiente | atendido |

No terceiro caso, `model_outputs_valid` foi `true` e `conflicting_evidence` foi
`false`: formato/referência eram válidos, mas o modelo não detectou o conflito.
O resultado segue experimental e exige revisão humana; não foi persistido como
aprovação de submissão. Os testes de transporte não validam qualidade semântica.

Os 130 testes Python, tipos do servidor/app e formatação passaram.
Sem alterações no checkpoint padrão do chat ou na configuração local da API.

## Reprodução e registros

Use diretórios novos; o gerador e o auditor recusam sobrescrever resultados.

```sh
ai/.venv/bin/python ai/scripts/prepare_evidence_robustness.py \
  ai/data/private/meu-robusto \
  --replay-source ai/data/private/document-long-contrasts-v2-2026-11-04/examples.jsonl
ai/.venv/bin/python -m confere_ai.instruct \
  --data ai/data/private/meu-robusto/examples.jsonl \
  --init-from ai/runs/multiple-evidence-v2-2026-11-11/best.pt \
  --run-dir ai/runs/meu-robusto --steps 1200 --batch-size 4 \
  --learning-rate 0.00003 --eval-every 300 --response-tokens 256 \
  --evaluation-context 1024 --document-fraction 0.65 --device mps --max-seconds 240
ai/.venv/bin/python ai/scripts/evaluate_evidence_robustness.py \
  ai/runs/meu-robusto --output ai/runs/meu-robusto/audit.json
```

O limite por tempo pode produzir outro passo final. A comparação exata dos
pesos exige o mesmo número de passos e ambiente; seed fixa não elimina essa
variação de duração.

Código do corpus e treino deste ciclo: commit `85c8305`.
Dados e pesos permanecem locais, ignorados pelo Git. Resultados rastreáveis:

- [Auditoria por objetivo e perfil](../output/ai/evidence-robustness-2026-11-11.json).
- [Três casos reais na ponte da API](../output/ai/evidence-robustness-api-2026-11-11.json).
- Relatório automático: `output/pdf/training/evidence-robustness-2026-11-11.pdf`.
- Execução completa: `ai/runs/evidence-robustness-2026-11-11/`.

O próximo diagnóstico deve separar falhas de seleção de ponteiros das falhas
de polaridade e consolidação, mantendo replay de conversa e avaliação por
família. Mais passos ou menor loss, isoladamente, não justificam promoção.

## Continuação intensiva

O [ciclo intensivo](INTENSIVE_TRAINING.md) ampliou o corpus e completou
10.000 passos. Registra a correção de memória, os dois checkpoints avaliados
e as limitações ainda observadas na API.
