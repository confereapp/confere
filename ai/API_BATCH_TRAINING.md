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

## Avaliação planejada

Serão separados os 120 casos documentais antigos, os 86 lotes novos e os 75
turnos de conversa. Os 18 documentos completos permitem avaliar 36 requisitos
pela consolidação real, incluindo cobertura contígua e citações literais.
Os controles conhecidos da API também serão executados com inferência real.

Nenhum checkpoint será promovido somente por concluir o treino ou reduzir loss.
Os resultados semânticos continuam pendentes durante a execução.
