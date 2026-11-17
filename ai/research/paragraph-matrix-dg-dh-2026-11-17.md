# Matriz de consultas e parágrafos — DG/DH

Os dois ensaios foram rejeitados na validação. Não houve atualização da API
nem inferência nos 54 pares da nova reserva. O avanço desta etapa é o controle
explícito da dependência da evidência e a identificação de memorização; não é
uma solução para a qualidade dos documentos reais.

## Dados

As consultas autorais foram anotadas contra três parágrafos integrais do mesmo
documento público. Cada relação foi revisada localmente como suporte,
contradição ou informação insuficiente. Não houve troca aleatória de trechos
com rótulos presumidos. Suportes presentes em mais de um parágrafo foram
mantidos, inclusive em TCC, monografia e estágio. O texto da fonte não mudou.

Uma consulta foi excluída por ter o mesmo rótulo nos três parágrafos. Seus
rótulos originais e o motivo foram preservados; não se fabricou uma mudança
de classe. Todas as 65 consultas mantidas têm pelo menos dois rótulos.

O arquivo `ai/data/paragraph-relation-matrix-v2.json` registra 195 pares:

| Uso | Documentos | Consultas | Pares |
| --- | ---: | ---: | ---: |
| Treino | 5 | 30 | 90 |
| Validação de desenvolvimento | 3 | 17 | 51 |
| Reserva ainda sem inferência | 3 | 18 | 54 |

Os títulos já examinados ficaram no desenvolvimento. A nova reserva contém
FAPDF, PESQUISA e RES_APROVEITAMENTO_DE_ESTUDOS, separados antes do treino.
Componentes lexicais/documentais não atravessam splits. Cada consulta ocorre
com rótulos diferentes dentro do seu próprio split. Mesmo um oráculo que
memorize somente a consulta acertaria no máximo dois terços dos pares da
reserva e nenhum grupo inteiro de relações diferentes.

Foram adicionados 501 exemplos autorais de replay (projetos inteiros em ordem
determinística) e os 410 casos antigos de validação de desenvolvimento.
A inferência pública não participou do treino novo. Os dois TCCs reais
conhecidos continuam fora do treino. Proveniência e licença FaQuAD preservadas;
as relações são editoriais locais, não rótulos oficiais nem segunda revisão
humana independente.

## Ensaios controlados

Mesma origem CI, dados, normalização só do treino, representação de parágrafo
inteiro, seed 42, orçamento de 1000 passos e encoder congelado. A primeira
unidade é uniforme e sua dupla usa a mesma consulta com outro rótulo quando
disponível. O teste da amostragem verifica também mudança de texto. Os pesos
da origem e seus hashes ficam preservados em cada execução.

DG usa a cabeça residual anterior, com 295554 parâmetros treináveis.
DH troca somente a cabeça por uma correção linear de 4614 parâmetros. As duas
começam com correção zero e têm equivalência por resposta verificada antes do
treino. O carregador experimental restaura ambos os formatos, verifica a origem
e rejeita formatos desconhecidos. Nenhum deles é aceito pela API nativa.

## Resultado e diagnóstico

O passo zero foi selecionado nos dois ensaios: nenhum ajuste superou a origem
na prioridade pela pior família. A soma de acertos isolada não decide seleção.

Para diagnosticar memorização, foi avaliado também o último ponto de cada
ensaio, **não selecionado**:

| Último ponto | Treino acadêmico | Validação acadêmica | Consultas inteiras corretas na validação |
| --- | ---: | ---: | ---: |
| DG, não linear | 86/90 | 25/51 | 2/17 |
| DH, linear | 90/90 | 21/51 | 1/17 |

Reduzir a capacidade não resolveu a generalização. Mais passos sobre esses
pares não se justificam. Os exemplos e documentos ainda são poucos e podem
ensinar associações locais em vez da relação necessária. Não se afirma que o
encoder entende critérios acadêmicos amplos a partir de proposições factuais.

## Proteção da reserva e correção da métrica

O avaliador agora bloqueia matrizes novas sem aprovação explícita da validação
e retenção. A reserva também exige que passo e fingerprint do checkpoint
correspondam à seleção e ao corpus; um `last.pt` diferente não herda aprovação
de `best.pt`. A chamada real para DH encerrou antes de carregar o preditor e não
criou a pasta de resultados da reserva. A barreira falha fechada quando os
metadados de elegibilidade não existem. Os dois ensaios precederam a gravação
dessa chave; seus resultados já mostram reprovação de validação. Artefatos de
execução e hashes de código anteriores foram mantidos, sem reescrita.

Também foi corrigido o resumo de resultados para exigir status **e evidências**
corretos. Uma saída inconclusiva com evidência conflitante incorreta não conta
como acerto completo. A seleção anterior já aplicava essa regra; só o resumo
diagnóstico estava mais permissivo. A correção histórica foi registrada
separadamente, sem nova inferência.

Verificação de software: 260 testes passaram; `git diff --check` sem erros.
Isso não transforma os ensaios reprovados em modelos confiáveis.

## Limites e próximo passo

A prioridade agora é mais diversidade de **documentos e relações revisadas**,
incluindo critérios de explicar/descrever mecanismos, não mais passos ou
capacidade. A avaliação final precisa continuar por documentos inteiros, com
grupos completos da mesma consulta corretos e comparação contra controles.
Três documentos reservados e anotações sem segunda revisão humana ainda não
sustentam uma conclusão ampla de qualidade. Não houve teste HTTP ou nova rodada
dos PDFs completos: a validação intermediária já reprovou.

A busca de sequências normalizadas de oito palavras da reserva nos corpora
registrados da ancestralidade não encontrou ocorrências. É uma checagem de
exposição, sem inferência na reserva; não exclui paráfrases ou acessos fora
dos registros. Rastro: `output/ai/matrix-history-overlap-dg-2026-11-17.json`.

Manifesto e resultados resumidos:
`output/ai/paragraph-matrix-dg-data-2026-11-17.json` e
`output/ai/matrix-dg-dh-results-2026-11-17.json`.
