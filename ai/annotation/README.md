# Anotação de TCCs reais

Conjunto de teste independente: 8 TCCs de computação da Mauá (2025, handles
2231, 2232, 2234, 2236, 2239, 2241, 2242 e 2244), baixados do repositório
público, com OCR local. Uso somente para avaliação: a licença de treinamento
não está confirmada. Os PDFs, o OCR e as anotações ficam em `ai/data/private/`.

O checklist `checklist-v1.txt` tem 16 critérios genéricos aplicados a todos os
TCCs (128 decisões). Como o mesmo critério recebe rótulos diferentes em
trabalhos diferentes, o texto do requisito sozinho não decide.

## Como anotar

1. Abra `ai/annotation/annotate.html` no navegador (direto do repositório).
2. Carregue `ai/data/private/annotation-tasks-v1.json` e escreva seu nome.
3. Para cada critério, leia os 8 trechos candidatos (BM25) e escolha:
   - **Atende**: marque os trechos que sustentam;
   - **Contradiz**: o texto afirma o oposto; marque o trecho;
   - **Não encontrado**: nada mostra o critério.
   Se a evidência estiver fora dos candidatos, cole o trecho e a página.
   "Ver página" abre a imagem original para conferir o OCR.
4. Exporte o JSON ao terminar. O progresso fica salvo no navegador.

O que "atende" exige em cada critério está em `criterios-v1.md`: o TCC
precisa fazer ou descrever o item, não só citar a palavra.

Divergências entre anotadores vão para adjudicação: um arquivo de tarefas só
com os itens divergentes, cada um mostrando a regra do critério e as duas
anotações (`ai/data/private/annotation-adjudication-v1.json`). A decisão da
adjudicação substitui os rótulos desses itens.

Anote sem consultar saídas do modelo. Para medir concordância, uma segunda
pessoa anota ao menos parte dos itens de forma independente.

## Importar e avaliar

```sh
PYTHONPATH=ai/src ai/.venv/bin/python ai/scripts/import_annotations.py \
  --tasks ai/data/private/annotation-tasks-v1.json \
  --annotations anotacoes-NOME.json [--second anotacoes-OUTRA.json]

PYTHONPATH=ai/src:ai/scripts ai/.venv/bin/python ai/scripts/evaluate_real_documents.py \
  --checkpoint ai/runs/evidence-mix-el-2026-11-17/best.pt --top-k 8 --device mps \
  --candidates ai/data/private/maua-*/examples.blind.jsonl --output ai/data/private/blind-v1-el
```

Critério para deixar de ser experimental, fixado antes do teste: precisão de
"atendido" de pelo menos 90% (com intervalo de confiança), evidência correta em
pelo menos 90% das aprovações e desempenho acima de BM25 sozinho e do controle
que lê só a consulta.

Para regenerar as tarefas:

```sh
PYTHONPATH=ai/src ai/.venv/bin/python ai/scripts/prepare_annotation.py \
  --documents ai/data/private/maua-{2231,2232,2234,2236,2239,2241,2242,2244} \
  --checklist ai/annotation/checklist-v1.txt --output ai/data/private/annotation-tasks-v1.json
```
