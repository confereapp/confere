# Treino com lotes efetivos da API — 11/11/2026

Esta etapa trabalha as falhas restantes do [ciclo intensivo](INTENSIVE_TRAINING.md):
evidência negativa no fim de texto longo e mistura de polaridades entre requisitos.
Os alvos usam a divisão real de `project_review.batches_for`, com o mesmo contexto,
reserva de resposta e aliases locais do serviço textual.

## Dados e contrato

O gerador cria documentos autorais com intervalos conhecidos para cada afirmação
ou negação. Para cada requisito, o alvo contém somente as evidências desse assunto
presentes no lote atual. Um lote sem evidência relevante recebe insuficiência;
evidências opostas do mesmo assunto recebem insuficiência com ambas as referências.

Os mesmos textos são usados com requisitos diferentes e alvos diferentes. Isso
permite treinar o escopo do requisito em vez de tratar toda negação como conflito.
Os seis perfis cobrem escopo curto/longo, conflito no final, evidência no meio,
conflito curto e mera menção em documento longo.

Frases de evidência cortadas entre lotes não recebem rótulo inventado: o gerador
excluiu 24 documentos de treino por esse motivo e registrou a exclusão. Nenhum
documento da validação gerada precisou ser excluído. Os textos originais não
foram alterados para forçar um rótulo nos fragmentos.

Corpus `ai/data/private/api-batches-2026-11-11/`:

- 13.603 exemplos de treino, incluindo 6.105 turnos de conversa preservados.
- Replay documental anterior e até 200 exemplos por perfil do currículo intensivo.
- Alvos locais informativos preservados; lotes vazios de evidência amostrados
  para não dominar o novo treino.
- Os 195 exemplos anteriores de validação preservados integralmente.
- 86 novos contextos únicos de validação, gerados de 18 documentos completos.
- Total de validação: 281 exemplos, incluindo 75 turnos de conversa.
- Nenhum teste reservado ou documento real utilizado.

SHA-256 do JSONL:
`de45c98fff5a1c4c488a3114568ed2410e5feb95bf5b6e9c286ec7d9d92c2cfa`.
Código do currículo: commit `4826383`.

Os documentos completos e os novos lotes de validação compartilham fontes.
Essa avaliação é de desenvolvimento, com rótulos por construção e sem revisão
semântica independente. Não é teste cego nem valida generalização para TCCs.

## Memória da geração no MPS

Textos longos exigem gerar respostas com prefixos próximos do limite. Um ensaio
isolado, com prompt aleatório de 768 tokens e EOS desativado apenas nesse ensaio,
reproduziu crescimento de memória na geração com comprimentos distintos.
A proteção de 5,5 GB interrompeu o ensaio original após 112 tokens.

A geração MPS agora completa a dimensão para faixas de 128 e lê os logits na
última posição real do prefixo. O padding fica no futuro da atenção causal;
nenhum token de origem é removido ou retornado como parte do texto gerado.
CPU mantém o caminho anterior. Guardas de contexto, EOS, callbacks e interrupção
continuam ativos. Não houve mudança de pesos, arquitetura ou tokenizer nessa correção.

O ensaio corrigido completou 256 tokens, preenchendo a janela de 1.024, com
aproximadamente 1,4 GB observados no driver. Os primeiros 112 tokens foram
idênticos aos do ensaio anterior com o checkpoint real. Um teste no MPS também
compara geração gulosa sem padding e com padding em prefixos curtos, longos e
próximos da borda. Os 136 testes Python passaram após essas alterações.

- [Ensaio original](../output/ai/api-batches-generation-dynamic-2026-11-11.json).
- [Ensaio com geração agrupada](../output/ai/api-batches-generation-bucketed-2026-11-11.json).
- Código da correção: commit `7abf7dc`.

Os ensaios medem memória e preservação de tokens, sem medir qualidade de resposta.
Contadores do driver incluem reservas e não são pico contínuo nem RAM total.
Os resultados antigos foram preservados; a referência desta etapa é novamente
avaliada com o mesmo caminho de inferência do candidato.

## Execução

Inicialização no checkpoint de menor loss do ciclo intensivo, passo 7.000,
SHA-256 `082b4e1f7c2a84c1599f1377c1675a3aa741ed302e339816d9a3ef68ba50cde9`.
Execução: `ai/runs/api-batches-2026-11-11/`.

Configuração: MPS, lote 4, taxa `0.00001`, até 8.000 passos ou 1.200 segundos
no laço, checkpoints a cada 250 passos e 65% de probabilidade de lotes documentais.
Contexto 1.024 e saída de até 256 tokens. O restante dos lotes usa replay de conversa.
Os limites de memória ficam restritos ao processo, com níveis alto `0.65` e baixo
`0.5`. `caffeinate -i` acompanha a duração do processo.

```sh
ai/.venv/bin/python ai/scripts/prepare_api_batches.py \
  ai/data/private/meus-lotes \
  --source ai/data/private/evidence-intensive-ready-2026-11-11/examples.jsonl \
  --checkpoint ai/runs/evidence-intensive-bucketed-2026-11-11/best.pt
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.65 \
PYTORCH_MPS_LOW_WATERMARK_RATIO=0.5 \
caffeinate -i ai/.venv/bin/python -m confere_ai.instruct \
  --data ai/data/private/meus-lotes/examples.jsonl \
  --init-from ai/runs/evidence-intensive-bucketed-2026-11-11/best.pt \
  --run-dir ai/runs/meus-lotes --steps 8000 --batch-size 4 \
  --learning-rate 0.00001 --eval-every 250 --response-tokens 256 \
  --evaluation-context 1024 --document-fraction 0.65 --device mps --max-seconds 1200
```

Pastas de saída devem ser novas. Preparação e avaliações gerativas ficam fora
do limite de tempo do laço. A menor loss não implica melhor qualidade conjunta.

## Avaliação

Foram separados os 120 casos documentais antigos, os 86 lotes novos e os 75
turnos de conversa. Os 18 documentos completos permitem avaliar 36 requisitos
pela consolidação real, incluindo cobertura contígua e citações literais.
Os controles conhecidos da API também foram executados com inferência real.

Nenhum checkpoint foi promovido. Os dois candidatos reprovaram os critérios
conjuntos e os controles críticos da API permanecem com falhas.


## Resultado do treino

O laço completou 8.000 passos em 806,26 segundos (13 min 26 s), processando
32.000 exposições e 1.346.535 tokens supervisionados, incluindo repetições.
Foram 20.796 exposições documentais, com 1.142.866 tokens-alvo, e 11.204 de
conversa, com 203.669 tokens-alvo. O limite de 20 minutos não foi atingido.

A memória do driver observada no laço chegou a 3,04 GB; os tensores ativos,
a aproximadamente 346 MB; o RSS do processo, a 870 MB. São contadores distintos,
sem soma direta como RAM física total. A execução não foi interrompida.

A menor loss selecionou o passo 2.000. O passo final 8.000 também foi avaliado,
com a mesma coorte e o mesmo caminho de inferência. A referência foi novamente
executada com a geração MPS agrupada, preservando a comparação pareada.

| Coorte | Referência | Passo 2.000 | Passo 8.000 |
| --- | ---: | ---: | ---: |
| Documentos antigos, status e evidências completos | 106/120 | 113/120 | 104/120 |
| Novos lotes da API, status e evidências completos | 23/86 | 61/86 | 63/86 |
| Conversa, igualdade exata com histórico gerado | 68/75 | 69/75 | 65/75 |

No passo 2.000, todos os 206 casos documentais tiveram formato/referências válidos,
mas houve nove falsos atendimentos: um nos casos antigos e oito nos lotes novos.
O passo 8.000 também teve 206/206 de formato/referências válidos e dez falsos
atendimentos, um antigo e nove novos. Integridade estrutural não comprovou suporte
semântico das propostas.

O passo 2.000 não teve quedas por família de conversa ante a referência, mas
continua abaixo do mínimo em algumas famílias. O passo 8.000 caiu em saudação,
elogio e negação informais. Mais passos não resultaram no melhor candidato conjunto.

## Documentos completos

`evaluate_api_projects.py` carrega os pesos e executa inferência real em
`review_extracted`, reconstruindo os lotes para cada requisito. Confere hashes,
intervalos contíguos, citações literais e a correspondência entre as chamadas
locais e os alvos anotados. Não usa os rótulos para produzir respostas do modelo.

| Medida nos 18 documentos / 36 requisitos | Referência | Passo 2.000 | Passo 8.000 |
| --- | ---: | ---: | ---: |
| Status, conflito e validade corretos por requisito | 5/36 | 12/36 | 14/36 |
| Requisitos com todas as saídas válidas | 24/36 | 36/36 | 36/36 |
| Falsos atendimentos consolidados | 1 | 2 | 3 |
| Chamadas locais com status/evidências exatos | 23/86 | 61/86 | 63/86 |
| Documentos com todos os status/flags corretos | 0/18 | 1/18 | 1/18 |

A última linha mede status e flags de todos os requisitos com respostas válidas;
não exige que toda seleção local de evidência esteja correta. Essa seleção é
medida separadamente na linha de chamadas locais. Uma conclusão consolidada
pode coincidir com o rótulo apesar de erros em consultas intermediárias.

Os documentos completos são as fontes dos novos lotes de validação. Os resultados
não são uma avaliação independente: exercitam a consolidação do mesmo material
por outra medida. Ambos os checkpoints continuam confundindo assuntos diferentes.

Para repetir a auditoria em um novo arquivo de saída:

```sh
ai/.venv/bin/python ai/scripts/evaluate_api_projects.py \
  --checkpoint ai/runs/api-batches-2026-11-11/best.pt \
  --cases ai/data/private/api-batches-2026-11-11/validation-projects.json \
  --output ai/runs/api-batches-2026-11-11/nova-auditoria.json
```

## Controles reais da API

Os dois checkpoints passaram integralmente em cinco de sete casos autorais,
totalizando oito requisitos. Os resultados foram iguais nesse recorte:

- Afirmação, negação, conflito curto nas duas ordens e mera menção: corretos.
- Texto com testes presentes e backup ausente: os dois requisitos receberam
  insuficiência com conflitos indevidos, em vez de atendimento e não atendimento.
- Conflito em texto longo: retornou atendimento sem conflito, apesar de processar
  todo o texto em quatro lotes. A evidência negativa final não foi selecionada.

Esses dois problemas não foram resolvidos por esta etapa. Não houve promoção
para o chat nem alteração persistente da configuração da API. Todas as propostas
continuam experimentais e exigindo revisão humana; nenhuma submissão foi aprovada
ou persistida por estes testes.

Os 136 testes Python e seis TypeScript passaram, incluindo teste no MPS e
ponte HTTP com subprocesso Python real. O teste HTTP usa sessão sintética e não
exercita login real, rede ou banco. Testes de software não aprovam a semântica.

## Rastreabilidade

Checkpoint do passo 2.000: SHA-256
`b7da213ff2645f7277bea58ab0c0a50fa89bf427f2d4274ec4f0cbd9b8742168`.
Checkpoint do passo 8.000: SHA-256
`70e59305b0fecfb1c303cb7de24e9f228917414c505f7fabf415d8d0a97a4844`.

- [Contadores e configuração do treino](../output/ai/api-batches-training-2026-11-11.json).
- [Avaliação local do passo 2.000](../output/ai/api-batches-best-2026-11-11.json).
- [Avaliação local do passo 8.000](../output/ai/api-batches-last-2026-11-11.json).
- [Documentos completos, referência](../output/ai/api-batches-projects-reference-2026-11-11.json).
- [Documentos completos, passo 2.000](../output/ai/api-batches-projects-best-2026-11-11.json).
- [Documentos completos, passo 8.000](../output/ai/api-batches-projects-last-2026-11-11.json).
- [API, passo 2.000](../output/ai/api-batches-api-best-2026-11-11.json).
- [API, passo 8.000](../output/ai/api-batches-api-last-2026-11-11.json).

O relatório automático está em `output/pdf/training/api-batches-2026-11-11.pdf`.
Pesos, corpus, respostas completas das consultas e traces dos documentos ficam
em `ai/runs/api-batches-2026-11-11/` e `ai/data/private/api-batches-2026-11-11/`,
ignorados pelo Git. Os resumos versionados preservam hashes e os resultados.

O próximo diagnóstico deve isolar a ligação entre o requisito e cada afirmação,
e o efeito da granularidade/contexto das unidades. Repetir somente mais passos
neste corpus não demonstrou resolver essas falhas.
