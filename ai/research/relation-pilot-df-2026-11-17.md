# Relações entre afirmação e parágrafo — piloto DF

O piloto foi executado e rejeitado. Não foi adotado na API. O ganho agregado
em inferência textual não superou um controle que ignora a evidência e houve
regressão nas tarefas anteriores. Mais treino nesses mesmos dados não foi
iniciado após a reprovação.

## Redução de risco aplicada

- Fontes e licenças já auditadas, com hashes conferidos antes da preparação:
  [InferBR](https://huggingface.co/datasets/hapaxlegomenon/InferBR) e
  [FaQuAD](https://huggingface.co/datasets/eraldoluis/faquad). Somente os arquivos
  de treino das fontes; nenhum peso externo ou serviço pago.
- InferBR é supervisão auxiliar de inferência, com geração/revisão na origem.
  Não representa documentos de projetos. Os critérios locais de FaQuAD são
  editoriais, não rótulos oficiais nem avaliação humana independente.
- Documentos agrupados por título, inclusive entradas repetidas do mesmo título.
  Frases normalizadas de ambos os lados do par são conectadas por sobreposição
  de tokens de pelo menos 80%, com fechamento transitivo. Um componente inteiro
  recebe um único split. O loader recompõe a checagem, sem confiar só nos IDs.
- Nenhum parágrafo foi truncado ou alterado. Os 24 parágrafos têm âncora literal,
  origem e justificativa para suporte, contradição ou informação insuficiente.
  Os 72 pares incluem detalhes do mesmo assunto que o trecho não comprova.
- Treino: 6550 pares NLI + 45 pares acadêmicos, de cinco documentos. Validação:
  779 + 18, de dois documentos. Teste interno: 861 + 9, de um documento acadêmico.
  A reserva acadêmica é pequena demais para demonstrar generalização.
- Parágrafos são avaliados inteiros. Um alvo válido para o parágrafo não é
  atribuído automaticamente a todas as suas frases.
- Encoder próprio congelado; cabeça residual pequena; normalização e âncora
  calculadas só no treino. Origem zero da cabeça verificada contra a base.
- Orçamento fixado em 1000 passos, seed 42, seleção somente pela validação.
  Passo 250 selecionado. O teste não participou do treino nem da seleção.
- Teste executado uma vez, com arquivos/pesos congelados por hash. Controles
  simples que leem somente consulta ou somente trecho aprendem apenas no treino.
  Barreiras: piso por família, falsas aprovações, superar controles e preservar
  cada acerto antigo. Nenhuma média entre domínios libera promoção.

## Coincidências de status do teste interno (resumo histórico)

| Avaliação | Origem | Candidato | Só consulta | Só trecho |
| --- | ---: | ---: | ---: | ---: |
| NLI, 861 pares | 322 | 523 | 586 | 284 |
| Acadêmico, 9 pares | 2 | 1 | 5 | 3 |

Correção posterior do resumo: esses números contam coincidência de status.
Exigindo também o conjunto correto de evidências, os acertos NLI são 322 na
origem e 521 no candidato; nos nove pares acadêmicos, zero e um. A seleção já
usava resposta completa e não muda. O resumo do avaliador foi corrigido com
regressão para inconclusivo acompanhado de evidências incorretas. Rastro:
`output/ai/relation-df-metric-correction-2026-11-17.json`.

O controle que ignora a evidência supera o candidato nos dois domínios. Isso
mostra que o aumento de acertos não é uma demonstração suficiente de compreensão
da relação. O candidato fez 90 falsas aprovações em NLI e cinco nos nove pares
acadêmicos. Na regressão local, caiu de 407/410 para 331/410. Todos os gates
falharam. O teste agora é desenvolvimento se seus resultados orientarem ajustes.

## Exposição anterior e limites

Foram auditados metadados de 44 execuções ancestrais. Há corpora públicos de
conversa e pré-treino em Wikipedia na ancestralidade, portanto não se afirma
que o encoder seja inédito a qualquer texto público. Uma busca de sequências
normalizadas de oito palavras dos pares reservados nos textos/prompts/alvos
desses corpora não encontrou ocorrências. Isso não detecta toda paráfrase nem
prova ausência de acessos não registrados. Os dois TCCs conhecidos continuam
fora deste treino.

Também permanecem riscos de viés lexical do dataset, rótulos editoriais sem
segunda revisão e domínio acadêmico pequeno. O corpus contém proposições
factuais: ainda não valida critérios amplos como explicar uma arquitetura.

## Próxima etapa necessária

Antes de ampliar passos ou trocar arquitetura: revisar e ampliar pares
acadêmicos com a **mesma consulta** recebendo rótulos diferentes conforme o
trecho, e detalhes parecidos com relações diferentes. Esse balanceamento deve
ser anotado explicitamente, sem declarar contradição por omissão ou gerar
negativos aleatórios não revisados. É necessária reserva de mais documentos
inteiros, além de segunda revisão humana das anotações, para uma evidência
mais forte de qualidade. Este piloto não substitui essa revisão.

## Artefatos

Validação do software: 251 testes passaram; `git diff --check` sem erros.
Isso não altera a reprovação semântica dos pesos.

- `output/ai/relation-data-df-2026-11-17.json`: manifesto e hashes do corpus.
- `output/ai/relation-pilot-df-2026-11-17.json`: resultados e gates.
- `output/ai/relation-source-lineage-df-2026-11-17.json`: ancestralidade auditada.
- `output/ai/relation-history-overlap-df-2026-11-17.json`: busca de exposição.
- `ai/data/private/relation-test-df-2026-11-17/`: rastros integrais privados.

Não houve teste HTTP nem nova avaliação integral dos PDFs: a etapa de relações
já reprovou. Os padrões anteriores continuam intactos; isso não os torna bons.
