# Ciclo de múltiplas evidências — 11/11/2026

O contrato corrigiu a limitação descrita em `EVIDENCE_DIAGNOSTIC.md`: uma resposta
pode selecionar várias unidades e classificar cada evidência como `atendido`
ou `nao_atendido`. Evidências opostas consolidam em `evidencia_insuficiente`,
preservando as citações e `conflicting_evidence: true`.

Formato documental V2:

```json
{"status":"evidencia_insuficiente","evidence":[{"status":"atendido","unit":"page:1"},{"status":"nao_atendido","unit":"page:2"}]}
```

Sem evidências, a lista é vazia e o status é `evidencia_insuficiente`. Referências
inexistentes, duplicatas, campos adicionais e status incoerentes são recusados.
As citações continuam copiadas da fonte; sua existência não garante que sustentam
a classificação. O checkpoint declara `settings.grounding_schema: 2` e usa
192 tokens de resposta neste ciclo. Checkpoints sem essa configuração continuam
com o prompt, contrato `status/unit` e reserva de 64 tokens anteriores.

## Duas iterações locais

Inicialização exclusivamente nos nossos pesos de
`document-long-contrasts-2026-11-04/best.pt`, sem modelo externo.

| Execução | Treino | Passos finais / selecionados | Acertos completos |
| --- | ---: | ---: | ---: |
| `multiple-evidence-2026-11-11` | 240 | 400 / 200 | 10/24 |
| `multiple-evidence-v2-2026-11-11` | 1.200 | 800 / 400 | 22/24 |

A segunda iteração acrescentou formas de afirmação/negação e variou a redação
entre as duas evidências do mesmo caso. A validação ficou idêntica, com 24 casos,
seis modos balanceados, vocabulário e frases separados do treino. A validação
foi inspecionada para orientar a iteração: não é avaliação cega.
Nenhum teste reservado ou TCC candidato entrou neste ciclo.

No V2, o modo de negação isolada e o modo de conflito acertaram 3/4 cada;
os outros quatro modos acertaram 4/4. Um erro combinou insuficiência com apenas
uma evidência negativa, incoerente com o contrato. Outro selecionou uma unidade
extra inexistente. Apenas 22/24 respostas tiveram formato e referências válidos.
O critério exige 100% de formato/referências válidos e ao menos 90% de acertos
completos: **reprovado; pesos não promovidos**.

A avaliação estrita adicional dá 0/24 ao checkpoint anterior quando solicitado
o novo contrato. Isso mede incompatibilidade com V2, não perda ou ausência de
capacidade anterior. A primeira avaliação permissiva registrou 8/24 na referência
ao aceitar o formato legado. A reavaliação estrita está em `paired-audit.json`
e no relatório atualizado; `original-instruction-evaluation.json` e
`original-training-report.pdf`, na primeira execução, preservam os registros
anteriores à correção do avaliador.

Uma regressão curta de seis turnos conhecidos de conversa teve 6/6 nas três
versões, com a mesma reserva de 192 tokens e janela de 1.024. São apenas seis
turnos, sem avaliação abrangente de conversa aberta ou dos conjuntos anteriores.
O checkpoint padrão da TUI permanece inalterado.

## Rastreabilidade e reprodução

Pesos selecionados V2: passo 400, SHA-256
`3619313cd2abcfc7c061961c3825e9bd1cf71ef329fef86e0e60c1173c342a1f`.
Corpus V2: SHA-256
`150ce33b136855e50aeb3ed20b154f25d881e7cd6c64339addabf97aa7bfc3bb`.

Para reproduzir o segundo corpus e o ajuste, use pastas novas:

```sh
ai/.venv/bin/python ai/scripts/prepare_multiple_evidence.py ai/data/private/meu-multi
ai/.venv/bin/python -m confere_ai.instruct \
  --data ai/data/private/meu-multi/examples.jsonl \
  --init-from ai/runs/multiple-evidence-2026-11-11/best.pt \
  --run-dir ai/runs/meu-multi --device mps \
  --steps 800 --batch-size 4 --learning-rate 0.00003 \
  --eval-every 200 --response-tokens 192 --max-seconds 180
```

Dados, pesos e resultados completos permanecem nos diretórios locais ignorados:
`ai/data/private/multiple-evidence*-2026-11-11/` e
`ai/runs/multiple-evidence*-2026-11-11/`. Cada execução contém manifesto,
configuração, métricas, checkpoints e avaliação. O V2 também contém
`paired-audit.json`, `api-smoke.json` e `http-smoke.json`.
Os PDFs automáticos ficam em `output/pdf/training/`. O [resumo verificável
em JSON](../output/ai/multiple-evidence-2026-11-11.json) preserva hashes,
contagens por modo e o retorno do teste HTTP com texto autoral.

A [ponte da API](API.md) foi verificada com inferência real. Integração no app,
upload, persistência e promoção do modelo continuam pendentes. O próximo ciclo
de qualidade deve variar distratores, tamanho e distribuição das evidências,
incluindo afirmações opostas dentro da mesma unidade e uma avaliação nova
revisada. Repetir apenas estes 24 casos não valida projetos reais.
