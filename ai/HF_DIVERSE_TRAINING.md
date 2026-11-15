# Amostra pública diversificada — 15/11/2026

Continuação do [piloto público](HF_PILOT.md), sem importar pesos externos.
Inicialização: ai/runs/hf-pilot-filtered-2026-11-15/best.pt, passo 1.750.

## Coleta e seleção

Fonte: https://huggingface.co/datasets/Polygl0t/gigaverbo-v2-sft,
Apache 2.0, revisão observada 845713c9330519809c104fa904481c57ae948789.
Planejamento: 200 páginas de cem linhas de quatro subsets, offsets amostrados
sem reposição, seed 151126. A primeira coleta paralela atingiu HTTP 429 após
81 páginas de general (8.100 linhas); outros subsets não foram recebidos.
Não houve repetição de chamadas para contornar o limite. Os JSON brutos e seus
hashes foram preservados; a coleta parcial foi explicitamente registrada.
A revisão foi registrada após essa coleta. O viewer não garante snapshot
imutável, por isso a referência exata são os JSON locais com SHA256.

O coletor agora espaça chamadas em um segundo e para no primeiro HTTP 429,
registrando páginas recebidas, pendências e Retry-After. Não retoma implicitamente
um diretório existente. O teste simula 429 e prova que não há novas consultas
às linhas após a recusa.

Filtros: instruct_score >=4, alternância de papéis sem system (o formato atual
não o representa), exclusão de identidades alheias, deduplicação exata de contexto,
contexto de até 1.024 tokens e resposta de até 128 tokens incluindo EOS, sem
truncamento. O teto de 128 só se aplica aos novos exemplos públicos/humanos:
os exemplos documentais e de conversa anteriores permanecem intactos.

Seleção: 1.270 exemplos públicos e 19 humanos de treino. O snapshot humano é o
mesmo OpenAssistant do piloto, novamente filtrado, não uma nova coleta humana.
Validação pública: 74 exemplos sintéticos e um humano. Teste reservado: 70
sintéticos e dois humanos, fora do arquivo consumido pelo treino. Os 13.603
exemplos anteriores de treino e 281 de validação foram preservados. Total:
14.892 de treino e 356 de validação. Inspeção informal de 12 amostras públicas
identificou inferência, paráfrase e classificação, mas também ruídos de tradução.
Não é revisão humana completa.

O novo corpus foi construído sobre api-batches, não sobre o corpus do piloto;
o ponto de partida dos pesos já viu parte dos dados públicos anteriores.
Os splits públicos são determinísticos por conversa, mas sem auditoria de
duplicação semântica ou contaminação da linhagem. Não se trata de teste cego.

## Treino

```sh
rtk proxy env PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.65 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.5 caffeinate -i ai/.venv/bin/python -m confere_ai.instruct \
  --data ai/data/private/hf-diverse-2026-11-15/examples.jsonl \
  --init-from ai/runs/hf-pilot-filtered-2026-11-15/best.pt \
  --run-dir ai/runs/hf-diverse-2026-11-15 --steps 4000 --batch-size 4 \
  --learning-rate 0.00001 --eval-every 500 --response-tokens 256 \
  --evaluation-context 1024 --human-fraction 0.01 --public-fraction 0.39 \
  --document-fraction 0.4 --device mps --max-seconds 600
```

Mistura por lote: 1% humano, 39% público, 40% documental e 20% autoral de conversa.
O orçamento de geração avaliativa continua 256 para comparação documental;
a API de chat gera no máximo 128. O limite de 600 segundos cobre apenas o loop
de treino. Novos dados, pesos e saídas brutas ficam locais/ignorados.
