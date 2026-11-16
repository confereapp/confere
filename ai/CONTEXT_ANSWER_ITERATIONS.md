# Iterações com respostas sustentadas por contexto — 15/11/2026

Motivo: a mistura pública reduziu loss, mas apresentou explicações incoerentes,
rótulos questionáveis e regressões documentais. Este ciclo retira os novos alvos
públicos ruidosos do treino e usa somente cenários autorais verificáveis. Os pesos
iniciais continuam herdando o histórico público anterior; não há pesos externos.

## Critérios antes do treino

Cinco famílias: extração de fato, informação ausente, conflito no texto, presença/
ausência explícita de testes e memória após correção de nome. Exige-se >=90% de
igualdade exata em cada família curta, sem regressão por família legada, além do
critério documental já existente: formato e referências 100% válidos e >=90%
por perfil. Não basta uma média global. Esses gates são de desenvolvimento e
não demonstram conversa aberta geral nem confiabilidade em documentos reais.

## Corpus

Base: api-batches-2026-11-11/examples.jsonl. Preserva documentos e os 281 casos
legados de validação; replay conversacional por projetos inteiros com o helper
existente balanced_replay. Nenhum teste reservado é consultado.

Novos exemplos: 1.200 cenários por família, com três turnos por cenário de memória,
8.400 linhas de treino ao todo. Validação adicional: 12 cenários por família,
84 linhas; formas de pergunta e nomes diferentes do treino. Total do corpus:
21.881 treino e 365 validação. SHA256 dos dados:
d2a8d3d413db62a5fe47157653fca1cfebebb5ffc9ad03109dee10b7c3614d83.

O alvo é derivado dos fatos do cenário; ausências e conflitos são construídos
explicitamente. A ordem das frases é embaralhada. O identificador no prompt é
opaco, sem o nome da família que revelaria a classe. Os dados são sintéticos
autorais, sem revisão humana independente. Formas e nomes de validação são
separados, mas esta validação será usada para desenvolvimento; não é teste cego.
O corpus não ensina conhecimento factual aberto nem linguagem geral.

## Ciclo A

Inicialização: hf-diverse-2026-11-15/best.pt (passo 4.000).
4.000 passos adicionais, batch 4, lr 3e-5, avaliação a cada 500 passos, MPS,
contexto 1.024, reserva de resposta 256, 50% de lotes documentais e 50% de
conversa autoral. Máximo de 600 segundos no loop, excluindo preparação e geração
avaliativa. Diretório novo: ai/runs/context-answers-a-2026-11-15.

### Resultado A

4.000 passos, 544.202 tokens-alvo (inclui repetições), 300,1 s no loop.
Checkpoint selecionado: passo 4.000, SHA256
114fad58805838ae4cefd0d972ac6f7aa9c3fbc7b11699a49676026503e5b711.
Todos os tensores de modelo, configuração e tokenizer de best.pt e last.pt foram
comparados e são iguais; não foi repetida a geração de pesos idênticos.

Na validação original: fatos 4/12, ausências 12/12, conflitos 11/12,
presença/ausência de testes 9/12, memória 23/36. Os gates reprovaram, inclusive
por regressões legadas. Parte das falhas envolveu perguntas novas sobre usuário
respondidas com a identidade do assistente, nomes novos e escolha do campo errado.
As perguntas “Qual nome aparece no texto?” eram ambíguas: há nome de projeto e
nome de pessoa no contexto. Foram preservadas no relatório original e corrigidas
explicitamente na próxima versão; não se atribui a correção de pergunta a ganho
de treinamento.

## Ciclo B

Inicialização: checkpoint A selecionado. Novos nomes combinatórios (excluindo
Clara, Davi, Renata e Victor, usados na validação), perguntas por composição de
formas/aliases e contraste entre o nome do usuário e o nome do assistente. A ordem
dessas duas perguntas é aleatória, evitando atalho por posição do turno.
Nos casos de fato/ausência, faltas e conflitos de outro assunto não devem alterar
a resposta à pergunta atual. Nenhuma resposta pública nova é adicionada.

Fonte definitiva: context-answers-contrasts-2026-11-15. Dois diretórios
intermediários expanded/expanded-v2 foram preservados e não alimentaram treino;
a fonte definitiva acrescenta os contrastes e alterna os turnos de identidade.
23.081 exemplos de treino e 377 de validação. A validação mantém IDs/alvos/históricos
originais, com sete perguntas explicitamente corrigidas para consultar o nome do
projeto e mais 12 contrastes de distração. Ambos os pesos comparados serão
avaliados nesse mesmo conjunto versionado. Perguntas completas de validação e
nomes reservados não são colocados no treino; os testes verificam isso nas amostras.
É desenvolvimento inspecionado, não avaliação cega nem revisão humana independente.

6.000 passos, batch 4, lr 2e-5, avaliações a cada 500 passos, 65% de lotes documentais,
35% de conversa autoral, contexto 1.024 e reserva de resposta 256, MPS com limites
high=0,65 e low=0,5. Limite de 600 segundos no loop. Diretório:
ai/runs/context-answers-b-2026-11-15.

Nove diagnósticos adicionais foram definidos em evaluate_context_smoke.py antes
da conclusão do ciclo B, fora do corpus de treino, sem o preâmbulo/registro do
currículo. Usam 128 tokens, como a API. Não são benchmark de conversa geral;
igualdade literal deve ser interpretada com leitura semântica.

### Resultado B

O loop encerrou pelo limite de tempo no passo 5.630 (600 s mais a última avaliação),
não nos 6.000 planejados. O mínimo de loss selecionou o passo 2.000, mas a avaliação
gerativa dos últimos pesos foi melhor nas famílias contextuais:

| Família | Selecionado 2.000 | Último 5.630 |
| --- | ---: | ---: |
| Fato, incluindo distrações | 18/24 | 23/24 |
| Informação ausente | 5/12 | 8/12 |
| Conflito | 11/12 | 12/12 |
| Presença/ausência explícita | 12/12 | 12/12 |
| Memória | 30/36 | 33/36 |

A família de ausência continuou reprovada: quatro perguntas sobre quantidade de
dias receberam linguagem de programação como resposta. Persistiram trocas/erros
de grafia em alguns nomes e regressões documentais. Os critérios completos
reprovaram ambos os candidatos. Não se promove nenhum deles.

Os nove diagnósticos fora do formato de treino tiveram quatro respostas literais
corretas no último checkpoint: responsável Lia, ausência de prazo, nome corrigido
Rui e identidade do assistente. Os demais recusaram responder, inclusive quando
havia dados suficientes. É evidência de dependência de formulação/estilo, não
comprovação de diálogo geral. Os casos já foram inspecionados e passam a ser
apenas diagnóstico de desenvolvimento, não teste independente para próximos ciclos.

## Ciclo C

Inicializa do último checkpoint B, escolhido pela geração contextual e não pela
loss. Troca o preâmbulo rígido por quatro formatos (incluindo pergunta antes do
texto), varia a realização das mesmas informações autorais e acrescenta 2.000
pares de campos presentes/ausentes no mesmo contexto, 500 casos de atribuição de
tarefas e 500 cenários com perguntas de presença, ausência explícita e assunto
não informado. Mais 2.000 linhas documentais curtas foram geradas pelo pipeline
existente de lotes reais, com requisitos diferentes no mesmo texto. A validação
B fica literalmente intacta: 377 casos. Não foi consultado teste reservado.

Corpus: context-answers-natural-v2-2026-11-15, 31.081 linhas de treino e 377 de
validação. O diretório natural anterior foi preservado e não foi usado no treino.
Não há novo download ou alvos externos. SHA256 dos dados:
da3b46b7f735d713cbef8d5401dedde03370369727ac59d833c1aa548fc7fc81.

Planejado: 6.000 passos, batch 4, lr 1e-5, avaliação a cada 500, 60% de lotes
documentais e 40% de conversa autoral, contexto 1.024 e reserva 256, MPS com
limites de memória anteriores e máximo de 600 segundos no loop.
Diretório: ai/runs/context-answers-c-2026-11-15.

Artefatos A e B: [A](../output/ai/context-answers-a-2026-11-15.json),
[B](../output/ai/context-answers-b-2026-11-15.json),
[manifesto C](../output/ai/context-answers-natural-2026-11-15-manifest.json).
Os diagnósticos não validam fluência, conhecimento aberto ou documentos reais.
Os relatórios históricos de pré-treinamento já registravam falta de fluência;
estes currículos focam competências limitadas e não removem essa ressalva.

### Resultado C

O loop parou no passo 5.893 por prazo. O mínimo de loss selecionou o passo 1.500.
Esse checkpoint passou no gate contextual: fatos 24/24, ausências 11/12,
conflitos 12/12, presença/ausência 12/12 e memória 33/36. Não passou no gate
conjunto: persistiram regressões legadas e perfis documentais abaixo do mínimo.
O último checkpoint piorou (18/24, 11/12, 12/12, 12/12 e 28/36 respectivamente),
portanto não foi usado como inicialização da próxima etapa. O diagnóstico de
linguagem natural do selecionado ficou em 4/9 respostas literais corretas, com
falhas ainda claras; não comprova conversa livre útil fora desse escopo.

Uma ablação offline usou os primeiros seis documentos (um por família) do conjunto
autoral já inspecionado. Dividir todo o texto em frases, sem remover caracteres,
piorou de 4/12 para 2/12 requisitos corretos e aumentou de 28 para 134 chamadas.
Ambas as versões tiveram 12 saídas válidas e zero aprovações incorretas nesse
piloto. A mudança foi rejeitada: a API e a estratégia de divisão de produção
permaneceram intactas. O regex era apenas diagnóstico, não segmentador geral.

## Ciclo D

Prioridade confirmada pelo usuário: conversa e documentos com o mesmo peso na
aceitação; não promover ganho de um objetivo às custas do outro. Inicialização:
C/best.pt, passo 1.500, que foi melhor que C/last.pt na avaliação gerativa.

Adiciona 4.000 linhas documentais pareadas: o mesmo texto é consultado sobre dois
requisitos e os alvos/referências são derivados das afirmações explícitas por
unidade. Nove modos cobrem presença, ausência, menção, múltiplas evidências,
conflitos entre unidades, conflito na mesma unidade e assuntos diferentes na
mesma unidade. Um terço dos cenários usa dois nomes de recurso inventados, um
terço mistura recurso inventado e substantivo comum e outro usa dois termos
comuns. Isso impede que a seleção dependa apenas de um vocabulário fechado de
seis recursos. As frases afirmam presença/ausência; não rotulam intenção futura
como implementação. Validação C preservada literalmente, sem teste reservado.

A primeira geração binding foi preservada e não usada no treino; a versão
binding-v2 força também comparações entre dois termos inventados no mesmo texto.
Fonte definitiva: context-answers-binding-v2-2026-11-15, 35.081 linhas de treino
e 377 de validação. SHA256:
082c8a107b91e6ed49334b4544e0dec2385db51245af47d387e54ea85ee7f8e3.

6.000 passos planejados, batch 4, lr 2e-5, avaliações a cada 500 passos, 50% de
lotes documentais e 50% de conversa autoral, contexto 1.024, reserva de resposta
256 e limite de 600 s no loop. Diretório: ai/runs/context-answers-d-2026-11-15.

A avaliação inicial C foi reaproveitada explicitamente com --baseline-from.
As verificações exigem pesos SHA256 iguais, mesma coorte inteira incluindo
histórico/metadados, dados originais intactos, mesmos contexto/orçamento e código
de modelo, prompt, tokenizer, função score e versão torch. Para o registro C,
um sidecar documenta a verificação retrospectiva contra o commit cb81ffd; corpo
de score e arquivos model/chat/tokenizer foram comparados byte a byte/por fonte.
Nenhuma métrica foi reconstruída ou alterada. Novas execuções registram a
impressão digital do protocolo automaticamente. Cache ausente/incompatível é
recusado, sem reaproveitamento silencioso ou aprovação de candidato.

Passaram 151 testes Python antes de iniciar D. A API não foi reconfigurada e
nenhum checkpoint foi promovido. Os PDFs A, B e C foram renderizados e inspecionados.

### Resultado D

6.000 passos concluídos, sem superar a loss de validação inicial. A seleção foi
o passo zero (pesos do C), sem ganho: 330/377 casos exatos. Os últimos pesos
regrediram: fatos 13/24, ausências 11/12, conflitos 12/12, presença/ausência 12/12
e memória 23/36, além de regressões documentais. O candidato foi rejeitado.
Não confundir execução do treino com melhora: o checkpoint selecionado não
incorpora essas atualizações.

## Ciclo E: ajuste conservador e diagnóstico MPS

Volta a C/best.pt, reduz lr para 3e-6 e congela embeddings e quatro blocos iniciais.
Treina os dois últimos blocos, norm e saída: 3.942.912 parâmetros treináveis.
Mesmos dados D, mesma validação, batch 4, 50% de lotes documentais, até 8.000 passos
ou 600 segundos no loop. A arquitetura e o formato de pesos não mudam.

A primeira execução E interrompeu por gradiente não finito. O último checkpoint
preservado era o passo zero; o número de atualizações não salvas da execução
original é desconhecido. Não foi desativada a proteção contra gradientes inválidos.

Um ensaio reproduziu o problema no passo 39: MPS gerou gradiente inválido em
blocks.4.norm1.weight, apesar de loss finita (0,0577043); CPU com mesmos pesos e
lote teve loss equivalente e gradientes finitos. A correção faz a entrada
congelada da primeira LayerNorm treinável ser uma folha que recebe gradiente no
MPS; isso não atualiza o encoder congelado. O hook não atua sob no_grad, portanto
não muda a inferência. Reconfigurar o modelo remove o hook antigo.

Após a correção: 512 atualizações reais sem gradientes inválidos; no lote do
passo 39, loss MPS 0,0555913858 versus CPU 0,0555913411, diferença máxima absoluta
de gradiente 1,39e-7. Testes CPU confirmam pesos congelados intactos; teste MPS
confirma gradientes finitos e hook desabilitado na inferência. Não é alegação de
que o backend inteiro esteja corrigido, apenas a execução exercitada.

A execução interrompida e probes foram preservados em context-answers-e-2026-11-15.
O reinício usa context-answers-e-fixed-2026-11-15 e novamente inicializa do C,
sem reaproveitar atualizações não salvas do ensaio. A verificação posterior dos
pesos congelados e os gates de ambos os objetivos continuam obrigatórios.

Os gates agora registram também a aceitação preexistente completa de conversa
(>=80% em cada família legada), além de contexto >=90%, documentos >=90% por
perfil/formato 100% e ausência de regressões. Nenhum critério anterior é removido;
os candidatos anteriores já estavam reprovados nos critérios documentais.
