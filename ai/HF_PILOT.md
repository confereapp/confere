# Piloto de conversas públicas — 15/11/2026

Fontes: [GigaVerbo-v2-SFT](https://huggingface.co/datasets/Polygl0t/gigaverbo-v2-sft),
subconjunto general, e [OpenAssistant/oasst1](https://huggingface.co/datasets/OpenAssistant/oasst1).
Ambos declaram Apache 2.0. A licença/proveniência dos dados continua registrada;
a ausência de atribuições nas mensagens de commit não remove essas informações.
Nenhum peso externo é importado.

Foram consultadas as primeiras 2.500 linhas de general pelo dataset viewer.
As respostas JSON originais estão em ai/data/private/hf-conversation-2026-11-15,
com SHA256 por arquivo. A revisão da origem foi conferida antes e depois da coleta;
o parâmetro revision do viewer não garante snapshot imutável. Os JSON preservados
são a referência exata desta execução. A amostra inicial não é aleatória.

Filtro de pontuação >=4, alternância user/assistant, ausência de mensagens de
sistema (não representadas no formato atual), limite real de 1.024 tokens e
resposta de até 256 tokens. Respostas com identidades de outros assistentes
foram excluídas. Filtros automáticos e inspeção informal de oito exemplos não
constituem revisão humana completa de qualidade ou factualidade.

Resultado: 192 exemplos públicos sintéticos e 45 respostas humanas de treino.
São preservados 13.603 exemplos anteriores de treino e os 281 de validação.
Acrescentam-se sete validações sintéticas e uma humana. Quatorze exemplos de
teste foram reservados e não são usados. Separação por conversa; duplicação
semântica e eventual exposição das validações humanas em checkpoints antigos
não foram auditadas. O corpus final tem 13.840 linhas de treino e 289 de validação.

Comando, na raiz:

```sh
rtk proxy env PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.65 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.5 caffeinate -i ai/.venv/bin/python -m confere_ai.instruct \
  --data ai/data/private/hf-pilot-filtered-2026-11-15/examples.jsonl \
  --init-from ai/runs/api-batches-2026-11-11/last.pt \
  --run-dir ai/runs/hf-pilot-filtered-2026-11-15 \
  --steps 2000 --batch-size 4 --learning-rate 0.00001 --eval-every 250 \
  --response-tokens 256 --evaluation-context 1024 --human-fraction 0.05 \
  --public-fraction 0.25 --document-fraction 0.5 --device mps --max-seconds 600
```

Mistura por lote: 5% humano, 25% público, 50% documental e 20% autoral de conversa.
O limite de tempo cobre o loop de treino, não preparação e geração avaliativa.
Igualdade exata dos casos controlados não mede qualidade de conversa aberta.

Uma primeira execução, hf-pilot-2026-11-15, foi interrompida durante a avaliação
inicial, antes de atualizações de pesos: a inspeção detectou identidades de outros
assistentes. Corpus e diretório originais foram preservados. O treino corrigido
usa outro diretório e o filtro de identidades.

## Resultado

Treino concluído: 2.000 passos, 168,34 segundos no loop, 591.578 tokens-alvo
processados com repetição. Pico observado do driver MPS: 3.041.411.072 bytes.
O checkpoint de menor loss de validação foi o passo 1.750; não foi promovido.
SHA256: `3086901d0889b6b40557b94ad76bdd2c60116143779835f166fc4fd8fa211015`.

| Avaliação | Referência anterior | Candidato selecionado |
| --- | ---: | ---: |
| Status + evidências locais corretos | 167/206 | 172/206 |
| Conversa autoral, igualdade exata | 65/75 | 64/75 |
| Loss das oito validações públicas, teacher-forced | 5,5547 | 3,4134 |
| Igualdade exata nas oito validações públicas | 0/8 | 0/8 |

A inspeção das oito saídas públicas não encontrou respostas satisfatórias:
há respostas fora do assunto e repetição extensa. A redução de loss não comprovou
conversa livre útil. Igualdade exata não é um critério suficiente para conversa
aberta, mas a inspeção também encontrou falhas claras. A loss foi calculada
com respostas de referência; a geração é autoregressiva.

Nos 18 documentos completos de desenvolvimento, o candidato acertou 14/36
requisitos, com uma aprovação incorreta, 36/36 saídas estruturalmente válidas e
nenhum documento com todos os requisitos corretos. Os mesmos casos tiveram
14/36 requisitos corretos na referência anterior de passo 8.000, conforme
`output/ai/api-batches-projects-last-2026-11-11.json`; não houve ganho agregado
nessa avaliação completa. Não são documentos reais ou teste cego.

Passaram 141 testes Python; após o ajuste de paginação, passaram os quatro
testes específicos do relatório. As três páginas do PDF foram renderizadas e
inspecionadas. Não houve mudança TypeScript nem nova checagem de UI/login/API
em rede nesta etapa. Nenhum checkpoint padrão ou configuração da API foi trocado.
Sem push.

A emissão automática do PDF falhou após o treino e a avaliação, pois o novo
manifesto usava `source` como objeto, enquanto o relatório espera uma string.
O builder foi corrigido para usar `source_snapshot` nos metadados completos,
e o PDF foi regenerado sem repetir o treino. As amostras agora começam em nova
página para evitar conflito com o rodapé em relatórios densos.

Artefatos: [manifesto](../output/ai/hf-pilot-2026-11-15-manifest.json),
[resultados e gerações públicas](../output/ai/hf-pilot-2026-11-15-results.json),
[PDF](../output/pdf/training/hf-pilot-filtered-2026-11-15.pdf).
Pesos, corpus e registros brutos permanecem locais/ignorados em
`ai/runs/hf-pilot-filtered-2026-11-15` e `ai/data/private/`.

Próxima hipótese a avaliar: ampliar a diversidade de exemplos curtos sem
superamostrar este conjunto pequeno, e medir separadamente interpretação de
texto e geração livre. Esta execução não justifica aumentar passos repetindo
os mesmos exemplos ou promover o candidato.
