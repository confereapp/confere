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

## Resultados

Concluídos 4.000 passos em 294,19 segundos no loop, com 856.383 tokens-alvo
processados incluindo repetições. O passo 4.000 foi selecionado pela menor loss
agregada de validação. SHA256 dos pesos:
`0732c0cad4c01a2e025b3835110a42a6fd2971aebced089a25af161ed1f45b70`.
Pico observado do driver MPS: 3.041.394.688 bytes. Não houve falha de memória.

| Critério | Piloto anterior (referência) | Candidato |
| --- | ---: | ---: |
| Evidências e status locais corretos | 172/206 | 168/206 |
| Conversa autoral: igualdade exata | 64/75 | 65/75 |
| Loss pública separada, teacher-forced (75 casos) | 3,5042 | 1,8249 |
| Igualdade exata pública, geração limitada a 128 tokens | 0/75 | 0/75 |
| Loss da única validação humana | 4,9102 | 5,1800 |

A loss publicada durante o treino mistura todo o conjunto de validação;
não deve ser apresentada como loss pública isolada. Os dois primeiros critérios
usam geração de 256 tokens, preservando a comparação documental. A avaliação
pública separada usa 128, igual ao teto do chat. Igualdade exata penaliza paráfrases
corretas e não é uma medida suficiente de qualidade aberta.

Inspeção informal de 12 gerações públicas, com seed fixa 151126: continuam
explicações incoerentes e repetição de frases. Identificou-se também um rótulo
questionável da fonte: uma reação crítica a um comentário racista foi marcada
como discurso de ódio apenas por mencioná-lo. Pontuação alta do classificador
da fonte não garante rótulo correto. Não foi feita revisão humana independente.

Composição aproximada por expressões literais nos prompts de treino: 344 de
inferência, 272 de similaridade, 393 de discurso de ódio e 261 outros. São grupos
heurísticos, não rótulos formais; mostram a concentração desta amostra em tarefas
de classificação, insuficiente para representar conversa livre. Não é correto
atribuir a melhora de loss a uma melhora geral de diálogo.

Nos mesmos 18 documentos completos de desenvolvimento, o candidato acertou
15/36 requisitos, com zero aprovações incorretas nesse conjunto, 36/36 saídas
estruturalmente válidas e dois projetos inteiramente corretos. O piloto anterior
teve 14/36, uma aprovação incorreta e zero projetos inteiramente corretos. É um
pequeno ganho nesse conjunto autoral, não evidência de segurança geral; a avaliação
local de 206 casos regrediu e inclui uma saída inválida. Não há teste cego nem
documentos reais. Nenhum candidato foi promovido.

Passaram 143 testes Python. As três páginas do PDF foram renderizadas e
inspecionadas. O smoke adicional usa a ponte Python real do chat; não testa HTTP,
login, Insomnia ou a qualidade de conversa geral. Não houve mudança TypeScript,
checagem de tipos adicional ou alteração do checkpoint padrão da API. Sem push.

Artefatos: [manifesto](../output/ai/hf-diverse-2026-11-15-manifest.json),
[resultados completos](../output/ai/hf-diverse-2026-11-15-results.json),
[inspeção qualitativa](../output/ai/hf-diverse-2026-11-15-qualitative.json),
[composição](../output/ai/hf-diverse-2026-11-15-composition.json),
[smoke do chat](../output/ai/hf-diverse-2026-11-15-chat-smoke.json),
[PDF](../output/pdf/training/hf-diverse-2026-11-15.pdf).

A próxima etapa útil requer curadoria dos alvos, balanceamento de tarefas e
exemplos de resposta curta sustentada por contexto. Os resultados não justificam
continuar ampliando volume desta mistura de classificação sem verificar os rótulos.
