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
