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

### Resultado E corrigido

8.000 passos concluídos em 281,7 s, com 1.090.841 tokens supervisionados contando
repetições. Todos os 50 tensores congelados ficaram idênticos aos do checkpoint C.
A loss de validação subiu de 0,01884 para 0,02163; o melhor checkpoint permaneceu
no passo zero. Os últimos pesos preservaram fatos 24/24, ausências 11/12,
conflitos 12/12 e presença 12/12, mas memória caiu de 33/36 para 28/36.
Também houve regressões documentais. Resultado rejeitado, sem promoção na API.
O PDF da execução foi renderizado e suas três páginas foram inspecionadas.

Uma segunda ablação offline no C comparou geração greedy com escolha por
probabilidade da resposta JSON completa entre candidatos finitos, incluindo EOS,
sem normalização por comprimento. Nos mesmos 26 casos estratificados, ambos
acertaram 21. Não houve ganho que justificasse a troca de decodificação; a API
continua inalterada. A enumeração usa ordem canônica e não cobre todas as
serializações equivalentes. O piloto já inspecionado é desenvolvimento, não teste
cego. Evidências completas nos resumos JSON C e E corrigido em output/ai.

## Ciclo F: peso nos valores de decisão

Hipótese: a média por token do JSON dedica muitos termos à sintaxe já aprendida;
a pequena loss não garante status e referências corretos. Novo ensaio inicializa
novamente de C, com os mesmos dados D/E e a mesma coorte de validação. Cada token
que se sobrepõe ao valor de um campo status ou unit recebe peso 4; demais tokens
supervisionados, inclusive EOS, peso 1. Prefixos e padding continuam mascarados.
A média é normalizada pela soma dos pesos ativos. O alinhamento usa bytes UTF-8
do tokenizer e confere literalmente os labels, sem procurar palavras no prompt.

Mesmos lr 3e-6, batch 4, últimos dois blocos treináveis e fração documental 0,5
do ciclo E. 4.000 passos planejados, avaliações a cada 500, limite 600 segundos.
Validação, seleção pela loss e avaliação gerativa permanecem sem pesos. Não há
mudança no contrato nem na inferência da API; peso padrão 1 preserva o treino
anterior. O ensaio testa documentos sem abandonar os gates de conversa.

## Currículo G: continuidade de intenção

A inspeção do corpus atual encontrou 197 exemplos de definição, mas zero turnos
de exemplo após a definição. O banco autoral original já tinha esses alvos.
Novo currículo repõe definição seguida de exemplo e escolha guiada em dois
turnos, além de contrastar identidade, função, escopo, saudação, agradecimento
e ausência de projeto em diferentes históricos. Não altera os dados documentais.
São 3.000 linhas novas, total 38.081 de treino; mesmas 377 de validação, preservadas
literalmente. As novas raízes são recusadas se coincidirem com uma formulação
reservada; continuações existentes como "dê um exemplo" reaproveitam alvos
autorais de treino. Grupos com conflito de contexto/alvo são descartados inteiros.
Teste verifica sequência, alvos autorais e preservação da validação.
Fonte context-answers-intent-2026-11-16, SHA256
8de71ef39c4b3c64115fbb138bdfe6f1ca4872415b085a259c5d8fe86d9fc484.
É um currículo controlado, sem alegação de fluência geral ou revisão independente.

### Resultado F e protocolo G

F concluiu 4.000 passos; o menor valor de validação foi no passo 500. Acerto
controlado subiu de 330/377 para 332/377, mas houve regressão em insuficiência
documental e agradecimentos informais. As cinco famílias contextuais mantiveram
24/24, 11/12, 12/12, 12/12 e 33/36. O gate documental e o de conversa legada
continuaram reprovados; nenhum checkpoint promovido. O ganho agregado não supera
os critérios separados.

G usará F/best.pt (passo 500) como candidato experimental de partida, sem adoção
na API, para testar a reposição de continuidade sem descartar os pequenos ganhos
documentais. Mesmos lr 3e-6, batch 4, últimos dois blocos, peso documental 4 e
fração documental 0,5; 4.000 passos planejados, avaliação a cada 500 e limite
600 segundos. Baseline F reaproveitado somente com o mesmo protocolo e validação.

### Resultado G

4.000 passos em 154,7 s; 556.617 tokens supervisionados com repetição. Melhor
checkpoint pela loss agregada: passo 500, 337/377 casos exatos, documentos 178/206,
conversa legada 67/75. As famílias contextuais ficaram em 24/24, 11/12, 12/12,
12/12 e 33/36. Saudação informal subiu para 2/3, escopo 3/3, agradecimento 3/3;
definição 1/4 e orientação 2/4 ainda falham. Houve regressão em grounded_atendido.
Os últimos pesos (4.000) mantiveram contexto, mas regrediram em vários perfis
documentais e em estado de projeto. Ambos reprovados; nenhuma promoção.
PDF de três páginas renderizado e inspecionado.

## Ciclo H: seleção com objetivos equilibrados e atualização integral

A validação contém 11.544 tokens documentais e 2.289 de conversa: a média por
token anterior dava 83,45% do peso aos documentos. Para a prioridade igual
confirmada pelo usuário, H seleciona pela média das duas losses por objetivo,
50% cada. A média por token continua registrada, assim como as perdas separadas;
a loss individual não recebe os pesos de treino na avaliação. Os gates gerativos
continuam idênticos e separados, sem aprovação por média. O modo anterior segue
como padrão das outras execuções; H ativa --balanced-objective-selection.

Inicializa de G/best.pt (passo 500), mantém dados G, batch 4, lr 3e-6, peso 4 nos
valores documentais e 50% dos lotes documentais. Atualiza a rede inteira para
testar se congelar os primeiros blocos limita a associação entre pergunta e fonte.
4.000 passos planejados, avaliações a cada 500, limite 600 segundos. A mudança
de seleção e o descongelamento acontecem juntos: este ciclo não isola seus
respectivos efeitos. Baseline G reaproveitado com coorte e protocolo idênticos.

### Resultado H

4.000 passos em 269,2 s, 556.617 tokens supervisionados com repetição. Seleção
pela média dos objetivos escolheu 3.500; a média por token teria escolhido 4.000.
Ambos reprovados. No selecionado, acerto caiu de 337/377 para 322/377, documentos
167/206. Definições melhoraram de 1/4 para 4/4, memória contextual de 33/36 para
35/36, mas ausências caíram para 9/12 e houve regressões documentais e legadas.
Os últimos pesos também falham: ausências 8/12 e memória 26/36. A nova seleção
expõe os dois objetivos, mas loss continua sem substituir os gates gerativos.
PDF de três páginas renderizado e inspecionado; nenhum checkpoint promovido.

## Ensaio I: média de pesos compatíveis

G preservou melhor os documentos/contexto; H aprendeu definições e correções,
mas regrediu em outros casos. O ensaio offline I testa a média 50/50 de G/best.pt
e H/best.pt, ambos descendentes da mesma rede, arquitetura, tokenizer, corpus e
contrato V2. Sem novas atualizações, sem trocar a decodificação nem a API.
O resultado é um checkpoint de passo zero com fontes e hashes explícitos e
optimizer reinicializado; não representa retomada do optimizer de qualquer origem.
Tensores incompatíveis/não finitos ou buffers inteiros divergentes são recusados.
Mesmos 377 casos controlados e mesmos gates, baseline G preservado. A média não
é presumida superior: precisa preservar os dois objetivos na geração.

### Resultado I

Média 50/50: 334/377 casos exatos, abaixo de G (337/377); ausências 10/12,
memória 34/36, documentos 176/206. Houve regressões em estado de projeto e
perfis documentais. A média foi rejeitada, sem adoção. Os pesos originais foram
preservados; não há alegação de que combinar checkpoints retenha suas habilidades.

## Comparação pelo fluxo completo e repetição do requisito

C e G foram avaliados nos mesmos 18 documentos autorais / 36 requisitos, usando
review_extracted, as mesmas unidades e o mesmo fracionamento da API textual.
SHA256 da coorte: 7bca5d9de232aee793240711b5d3358f92ab65663028ce543a6ca4e59988ac64.
C: 16/36 requisitos, 2/18 projetos inteiros, 66/86 chamadas locais corretas.
G: 19/36 requisitos, 5/18 projetos inteiros, 69/86 chamadas locais corretas.
Ambos: 36/36 formatos válidos e uma aprovação indevida. A mesma falha crítica
persiste em requisito explicitamente ausente, confundido com outro recurso
presente na mesma unidade. Citações são literais; isso não prova suporte semântico.
Os traces completos e hashes estão preservados nos diretórios das execuções;
a comparação compacta está em output/ai/context-complete-document-comparison-2026-11-16.json.
É validação de desenvolvimento já inspecionada, não documento real ou teste cego.

Uma ablação offline repetiu o requisito depois das unidades, preservando todo o
texto original, alvos e referências, em todos os 206 casos documentais G.
A hipótese de ajudar a associação pergunta/fonte foi refutada: 178/206 caiu para
153/206. A saída tinha espaço para o alvo inteiro, sem truncar nenhuma fonte.
A repetição foi rejeitada; nenhum prompt de produção foi alterado.

## Estado após esta retomada

F, G e H somam 12.000 atualizações e 1.659.492 tokens-alvo processados, incluindo
repetições. G/best.pt é o candidato experimental com maior total desta retomada:
337/377, mas permanece reprovado; não substitui o checkpoint da API ou do terminal.
Em comparação com C: documentos 175/206 para 178/206, conversa legada 63/75 para
67/75; contexto manteve 92/96 e >=90% em cada uma das cinco famílias. Os totais
não escondem as falhas por perfil e o falso positivo no fluxo completo.

O diagnóstico de linguagem natural G manteve 4/9 correspondências literais.
A inspeção confirmou erros reais: Lia virou "Li.", prazo explícito não foi
recuperado e documentação presente foi tratada como informação ausente. Algumas
respostas inconclusivas não são perigosas, mas também não respondem corretamente.
Os casos são desenvolvimento já inspecionado, não teste cego de conversa aberta.
A meta de qualidade nos dois objetivos ainda não foi atingida.

Nenhum contrato/prompt da API foi alterado nesta retomada. Sem teste HTTP
autenticado novo, Insomnia, interface ou deploy. Os gates gerativos e a revisão
humana obrigatória continuam. Checkpoints e dados privados foram preservados;
todos os ensaios rejeitados estão documentados, sem promoção baseada em loss.

Verificação final desta retomada: 162 testes Python passaram, incluindo execução
MPS de gradientes, alinhamento dos pesos documentais, seleção por objetivo,
continuidade dos diálogos e recusas de combinações incompatíveis. git diff --check
sem erros. PDFs E corrigido, F, G e H renderizados e inspecionados. Autoria Git
verificada como Guilherme Brasil, mensagens sem nomes ou atribuições a modelos.

## Continuação J: especialização por objetivo e seleção gerativa

O usuário reforçou continuar até a meta conjunta, sem encerrar em ganho parcial.
F/G/H mostraram interferência: H aprendeu definições, mas perdeu acerto documental;
a média I não reteve ambas as habilidades. J separa os ajustes mantendo a mesma
arquitetura e todos os casos originais. Não muda o contrato da API nesta fase.

Currículo documental: 13.498 linhas de treino e as mesmas 206 de validação,
SHA256 710d70b6711d71d6fc9f61b50a142a825e2e3aef490b0d8c6c26d73b8dae11ac.
Currículo de conversa: 24.583 de treino e as mesmas 171 de validação,
SHA256 10905137fcc5975a43552bdb524278acaae69ccd92c1eab8c60c6345f9be4912.
Nenhum alvo ou caso reservado foi acrescentado, removido ou alterado na separação.
A aceitação conjunta deverá recompor os 377 IDs e exigir os mesmos gates, além
de documentos completos sem aprovações indevidas e diagnóstico de conversa fora
dos templates. Dois ajustes isolados não constituem aprovação do sistema.

Nova opção --generative-selection escolhe pesos pelas respostas em cada avaliação:
aprovação integral vem primeiro; depois formato/referências válidos, pior família
relativa ao limiar, cobertura das demais famílias, falsos positivos e acerto total.
Limiares permanecem 90% por perfil documental/contextual e 80% por família legada.
A aprovação gerativa também exige zero falsos positivos documentais. A loss segue
registrada, mas não escolhe o checkpoint. Resultados completos por passo ficam
em generation-N.json. A validação é desenvolvimento repetidamente consultado,
sem alegação de teste cego. O cache da geração selecionada só evita repeti-la no
fim da mesma execução; o teste de regressão verifica essa correspondência.

J-documentos inicializa de G/best.pt, rede inteira, batch 4, lr 1e-5, perda padrão
por token (sem peso extra), 16.000 passos planejados, avaliação gerativa a cada
2.000 e limite de 3.600 s no loop incluindo avaliações. Não há lotes de conversa
nesse ajuste. Treino de conversa e recomposição dos gates continuam pendentes.

### Resultado documental J e correções de currículo

As avaliações completas de J foram 178/206 no passo zero, 169/206 no 2.000 e
164/206 no 4.000, com uma saída inválida neste último. A seleção gerativa manteve
o passo zero. Foi solicitada interrupção intencional por regressão; o processo
já havia salvo o checkpoint 6.000 e estava avaliando esse passo. A avaliação
6.000 foi interrompida e não tem resultado completo. Nenhuma falha numérica.
Checkpoints e registros preservados; PDF interrompido de três páginas renderizado
e inspecionado. O loop não concluiu os 16.000 passos planejados.

O treino passa a poder encerrar após N avaliações consecutivas sem melhora
(--selection-patience). O teste de integração confirma que geração melhor vence
loss pior e que um empate prolongado encerra com selection_plateau. Não muda os
gates de qualidade nem apaga as execuções rejeitadas. Passaram 168 testes Python.

A inspeção encontrou 625 linhas de treino contendo "A especificação prevê",
rotuladas pelo gerador como evidência de presença. Isso é ambíguo para comprovar
implementação. K exclui essas linhas apenas de um novo snapshot, preservando os
originais, e acrescenta pares com termos inventados sem o prefixo constante
"recurso" e com distratores antes/depois das afirmações. Ambos os requisitos
pareados continuam lendo exatamente as mesmas unidades. Duplicatas exatas são
omitidas, alvos conflitantes recusados. São 24.867 linhas de treino e as mesmas
206 de validação, SHA256 9cc27c7499efbd08d3e86544c70cdfa5593c0de8b6444bd22a82a47936f0953a.

O currículo de conversa recebe 6.200 casos autorais adicionais de fatos, ausência,
conflito, atribuição de tarefas e presença explícita: respostas derivadas do texto,
sem copiar diagnósticos completos. Nomes dos diagnósticos e da validação ficam
fora dos novos nomes; não é prova de teste cego, pois os padrões orientaram o
desenvolvimento. Total 30.783 de treino, mesmas 171 validações; SHA256
093f35fad1c8885e6f0970c7f8b37e49b385c89a6905ccaa9cfc2690144e6a91.

O auditor conjunto verifica integralmente casos/históricos/alvos de validação,
protocolo e pesos selecionados, recusando coortes incompletas ou duplicadas.
O treino pode evoluir; os casos de validação e seus gates permanecem intactos.

### Protocolo documental K

Fonte G/best.pt; 16.000 passos planejados, lr 1e-5, batch 4, geração a cada 2.000,
paciência 3 avaliações sem ganho e limite de 3.600 segundos no loop. Usa o corpus
K e treina os seis blocos, norm e saída, congelando apenas embeddings de tokens
e posições (--train-last-blocks 6). Hipótese: preservar as representações de entrada
e ampliar associação lexical reduz a dependência do vocabulário visto. Seleção
pela pior família gerada, peso por token normal (1), sem mistura de conversa.
São mudanças conjuntas de currículo e atualização, sem atribuição causal isolada.
Nenhum peso é promovido sem a avaliação conjunta e os fluxos completos.

### Interrupção K e conversa J

O ambiente interrompeu K com checkpoint 4.000 salvo, antes da conclusão da geração
4.000. No 2.000 houve 181/206 acertos, mas cinco saídas/referências inválidas e
queda em split_conflict (3/12); melhor selecionado ainda zero. Registros preservados
com interrupção explícita; PDF de três páginas renderizado e inspecionado.

A conversa J usou o currículo natural, lr 1e-5, batch 4, seis blocos treináveis
com embeddings congelados, geração a cada 2.000 e paciência 3. Terminou por
selection_plateau no passo 6.000: 159/171 no início, 136 no 2.000, 124 no 4.000
e 130 no 6.000. Pesos iniciais preservados. Fatos e memória contextual foram as
principais regressões; não houve promoção nem relaxamento dos gates.

A próxima conversa L usa o mesmo corpus e inicialização G, lr 3e-6, apenas dois
últimos blocos treináveis, e amostragem hierárquica: 50% de lotes legados, 50%
contextuais; cada família equiprovável dentro do grupo. Lotes não misturam
famílias. Teste confirma que tamanho da família não domina sua probabilidade.
16.000 passos planejados, batch 4, geração a cada 2.000, paciência 3, limite
3.600 s. Objetivo é aprender os casos novos preservando as habilidades anteriores.

## Reforço da base de linguagem N

As variações L/M ainda não atenderam aos critérios conjuntos. O próximo ensaio
retoma o pré-treino próprio foundation-1024-2026-11-02/best.pt, passo 2.748,
com o mesmo corpus congelado wikipedia-pt-10k (SHA256
8482048a8285d4c7355323521a31c781df7fc691b105fd5f65ea5972db84d0d1), mesmo
tokenizer, arquitetura e contexto. --resume preserva otimizador e RNG: batch 4,
lr 1e-4, seed 42, contexto/vocabulário 1.024 e mesmas 32 janelas de avaliação por
split. Planejados 30.000 passos adicionais (total 32.748), limite de 7.200 s no
loop, registros a cada 2.000. Teste reservado continua fora do preparo/métricas.

Hipótese: os ajustes curtos estão limitados pela base de linguagem pouco treinada;
é necessário verificar capacidade após mais previsão de tokens antes de insistir
nos mesmos templates. Melhora de loss causal não aprova chat ou conferência.
Depois do pré-treino, comparar validação comum e saídas reais, reajustar tarefas
e exigir novamente todos os gates gerativos e fluxos completos. Nenhum peso
externo importado, nenhuma publicação de corpus/pesos e nenhuma troca na API.

### Verificação intermediária N

Checkpoint 8.000 congelado em diagnostics/candidate.pt e comparado na CPU com a
base 2.748, enquanto a GPU continua o treino. Mesmas 3.230 janelas de 256 tokens,
773.505 alvos da validação completa; nenhum teste consultado. Loss 2,636929 para
2,556101. As gerações greedy ainda repetem palavras e produzem afirmações sem
sentido; a melhora estatística não é qualidade de conversa. Saídas completas,
dispositivo e hashes estão em output/ai/foundation-continue-n-intermediate-2026-11-16.json.
A execução N continua; checkpoint intermediário não foi promovido.

## Transferência documental e continuação N–Q — 16/11

O classificador por unidade no passo 5000 atingiu 206/206 casos controlados,
36/36 requisitos no conjunto original de documentos completos e 90/90 casos
curtos reservados. Isso não generalizou para a nova coorte de documentos
completos: 21/36 requisitos, 6/18 projetos completos e três falsos positivos.
Esses pesos **não foram aprovados**. A coorte completa antes reservada agora é
explicitamente desenvolvimento, só em validação, e exige novo teste após
congelamento. JSON e referências do classificador são construídos; relevância e
polaridades são predições aprendidas. Os detalhes estão em
`output/ai/evidence-classifier-development-2026-11-16.json`.

N concluiu mais 30.000 passos causais (94.186.392 posições de token processadas,
incluindo repetições). Nas mesmas 3.230 janelas de validação de 256 tokens,
a perda mudou de 2,636929 na origem para 2,469154 no melhor checkpoint de N;
o último marcou 2,474342. As continuações ainda repetem e inventam: essa redução
não aprova conversa. O teste do corpus não foi usado.

O, inicializado de um snapshot de N, ficou em 148/171 respostas exatas e perdeu
famílias. P restaurou 7.200 exemplos históricos de treino e selecionou 156/171,
mas ainda falhou em famílias legadas e nas perguntas com informação ausente.
Os PDFs e JSONs correspondentes registram as execuções sem promover pesos.

Q acrescenta perguntas pareadas sobre texto idêntico e nomes de recursos
próximos. A amostragem de conversa mantém 50% legado/50% contexto e famílias
uniformes, agora escolhidas por exemplo dentro do lote para reduzir oscilações.
A validação de conversa original permanece intacta; os 88 lotes documentais da
coorte que falhou são validação de desenvolvimento adicional, nunca treino.
Os critérios finais de família, cobertura e falsos positivos permanecem iguais.

## Rodadas R–T e transferência por consulta — 16/11

A rodada documental R, mesmo com novas molduras longas, não superou o checkpoint
inicial no critério por família (275/294; seis falsos positivos locais). Um
snapshot no passo 2500 acertou 99/100 fragmentos longos de treino amostrados,
mas só 63/100 pares curtos novos: acerto em textos conhecidos não demonstrou
associação aberta entre requisito e trecho. O formato experimental que move a
pergunta para depois do trecho foi encerrado deliberadamente: S selecionou
230/294, com dez falsos positivos, e nunca foi aprovado.

O gerador foi corrigido para que tipo de nome não determine polaridade; novos
fragmentos longos também incluem termos inventados próximos. A rodada T mantém
o protocolo anterior e amostra duas consultas sobre o mesmo trecho juntas,
preferindo alvos diferentes quando existem. São 14.719 fontes de treino com
contrastes entre 30.740 fontes distintas; nenhuma amostra de validação participa
da escolha dos lotes.

Na conversa, R chegou a 167/171 e foi encerrada deliberadamente no checkpoint
preservado de 11000 passos (geração selecionada completa no passo 10000; geração
11000 interrompida e não utilizada). O ajuste focado T alcançou 168/171 no passo
1000 e passou 8/9 diagnósticos naturais, contra 4/9 na referência G. O erro
natural restante confunde atributo ausente com outro incluído no mesmo texto;
o novo currículo U acrescenta perguntas de presença/ausência sobre texto
idêntico. Esses resultados intermediários **não atingem a meta conjunta**.
A validação original continua intacta e não há promoção ou mudança dos pesos
padrão. Os relatórios identificam encerramentos e medições incompletas.

## Validação ampliada e calibração — continuação de 16/11

U selecionou o passo 6000 com 171/171 casos originais, porém marcou só 7/9
nos diagnósticos naturais (nome do usuário e atributo incluído confundidos).
O acerto controlado não autoriza promoção. W acrescenta os nove diagnósticos
conhecidos como família separada de validação de desenvolvimento; nenhum entrou
no treino. Os 171 casos originais continuam intactos e são auditados à parte.
X acrescenta 1536 exemplos de treino com atributos opostos ligados por “mas”,
excluindo a combinação completa do diagnóstico, sem alterar a validação 180.

A calibração documental em pesos congelados de V (passo 1000) encontrou
limiares positivo 0,8 e negativo 0,6 com 293/294 acertos locais e nenhum falso
positivo. No fluxo completo dessa coorte, foram 35/36 requisitos e zero falsos
positivos; o perfil curto ficou em 5/6, ainda abaixo da meta. Os limiares são
metadados explícitos do checkpoint, não alteração dos alvos nem evidência
inventada. A compatibilidade anterior preserva os limiares 0,5/0,5.

Y amplia pares técnicos próximos (importação/exportação, autenticação/autorização
etc.). Esses termos antes reservados agora estão explicitamente no treino;
a coorte antiga é desenvolvimento e não mede vocabulário inédito. É obrigatório
um novo teste independente após congelar os pesos. Z libera toda a rede no
ajuste de conversa com taxa pequena para investigar a limitação das últimas
camadas congeladas. Nenhum desses candidatos foi promovido.

## Platôs e replay de erros — continuação de 16/11

W, X, Z, AC, AE e AG não superaram o snapshot congelado de V: 179/180,
incluindo 171/171 originais e 8/9 naturais. AA foi encerrada deliberadamente
após o checkpoint 7500 para corrigir o currículo e a ponderação das respostas.
AH ajustou somente os 384 parâmetros do viés final, sem superar a base.
O ajuste convexo desse mesmo viés usou exclusivamente os primeiros tokens de
3.200 exemplos de treino; nenhum dos sete fatores de interpolação melhorou a
base. Os demais tensores foram conferidos como idênticos ao checkpoint fonte.

O relatório de aceitação passou a exigir os mesmos 90% nas famílias contextuais
que o seletor já exigia. Resultados históricos desta etapa foram recalculados
sob esse critério, sem modificar respostas, alvos ou pesos. Em nove exemplos,
8/9 fica abaixo do limiar. Os PDFs registram essa reprovação.

Nas revisões, AD e a cabeça categórica AF também selecionaram o passo inicial:
292/294. As duas sondas residuais com encoder congelado não melhoraram esse
resultado e não foram integradas. A melhor calibração anterior ainda falha
em um requisito do fluxo documental completo; a meta conjunta segue pendente.

O replay novo identifica erros exclusivamente no treino. Na conversa, o snapshot
fonte erra o primeiro token em 8.314 de 68.843 exemplos; as conversas completas
desses casos são repetidas, preservando IDs distintos, sequência e os mesmos
180 exemplos de validação. Na revisão, o encoder fonte erra ao menos um dos
dois estados em 1.170 de 75.046 unidades de treino. AJ dedica metade de cada
lote a esses erros; nenhuma unidade de validação entra na mineração. AI e AJ
são experimentos em execução, sem promoção. Resultados das rodadas encerradas:
`output/ai/replay-iterations-2026-11-16.json`.

## Segmentação e primeiras verificações independentes — continuação de 16/11

AL preserva integralmente os pesos de Y e classifica cada frase antes de reunir
os dois estados por unidade original. As evidências continuam apontando para
os mesmos aliases e intervalos do fluxo da API. Isso resolveu os dois erros
locais restantes: 294/294 e 36/36 requisitos em cada coorte documental anterior,
sem falsos positivos. A primeira ablação com novas unidades por frase também
passou, mas a implementação selecionada preserva as unidades de produção.
O modo é explícito em `settings.unit_segmentation`; checkpoints anteriores
continuam usando a unidade inteira. A regex não é segmentação linguística geral.

Após congelar AL, foram criados 24 projetos, 48 requisitos, com novos termos
compostos e molduras. Houve apenas rejeição prévia de três fixtures cuja frase
anotada seria cortada pelo loteador; nenhuma seleção por resultado. A primeira
inferência marcou **30/48**, sem falsos positivos. Essa reprovação impede a
promoção, apesar do acerto nas coortes anteriores.

AI selecionou o passo 4000 com 179/180 e todas as famílias de desenvolvimento
aprovadas: 170/171 originais e 9/9 naturais. A primeira coorte nova de conversa,
72 exemplos completos ausentes do treino, marcou **37/72**: presença 11/36,
fatos 9/12, conflitos 0/6, informação ausente 5/6 e memória 12/12. O resultado
não demonstra generalização suficiente e também impede promoção.

As duas coortes reprovadas passaram explicitamente a desenvolvimento. AO
acrescenta 24.000 exemplos de recursos compostos e 24 pares de predicados ao
treino documental, mantendo as 410 avaliações separadas. AR acrescenta 36.000
exemplos candidatos de conversa com quantidade variável de fatos, pergunta
na mesma linha ou separada e recursos compostos; os 36.000 pedidos foram
conferidos contra os pedidos completos da avaliação. A validação ampliada tem 252 exemplos.
Nenhum alvo da avaliação é usado para treinar. Novas coortes independentes serão
necessárias depois do congelamento dos próximos candidatos.

Evidências: `output/ai/independent-checks-al-aq-2026-11-16.json` e os manifestos
AO/AR em `output/ai/`. A meta conjunta permanece pendente; pesos padrão da API
não foram alterados.

## Contrastes corrigidos e limite observado — continuação de 16/11

A calibração AU preservou os pesos de AS/500 e selecionou limiares 0,3/0,1:
403/410 casos locais, zero falsos positivos. No fluxo completo ampliado,
41/48 requisitos foram corretos, contra 30/48 na primeira avaliação dessa
coorte. Ainda há perfis abaixo de 90%; nenhum candidato foi promovido.
As calibrações posteriores AY e AZ não superaram AU.

O gerador de conflitos usava valores alternativos fixos, incluindo 200 dias,
Mar e Pascal. Esses valores permitiam atalhos. Os novos contrastes sorteiam
ambos os valores da mesma distribuição e incluem o mesmo valor isolado,
repetido e contraposto a outro. Os testes verificam a diversidade dos valores
e a equivalência semântica dos alvos. Apenas o treino foi substituído;
os corpora anteriores permanecem no disco.

AX melhorou uma amostra de treino de comparação de 16/24 para 23/24 no passo
3500, mas selecionou somente 218/252 na validação ampliada e não atingiu os
limiares. O segundo valor variável resolveu uma correlação indevida, sem
demonstrar generalização suficiente. As paráfrases de comparação também eram
restritas. BA amplia essas paráfrases conhecidas, mantendo os 252 pedidos
completos de avaliação separados. Não é teste de frases inéditas.

BB libera os embeddings no ajuste documental, antes congelados, para investigar
o aprendizado de disponibilidade e indisponibilidade. BA e BB estão em execução.
Os testes de código passaram em 212 casos Python; a suíte TypeScript passou
com nove testes e um teste opcional omitido; a checagem de tipos passou.
O teste opcional real pelo adaptador HTTP também passou com AU e AQ, incluindo
decisões positivas, negativas, conflitantes e offsets Unicode. Usou sessão
sintética, sem provar login real ou qualidade geral.

Os relatórios de encerramento identificam a interrupção deliberada de AV e AO.
Nenhuma medição incompleta foi usada para selecionar pesos. Evidências compactas
em `output/ai/comparison-iterations-2026-11-16.json` e manifestos correspondentes.

## Escopo da consulta e supervisão semântica — 17/11

BA terminou em 218/252; as duas tentativas com cabeça auxiliar usada somente
no treino, BD e BE, selecionaram 212/252 e 219/252. A cabeça não é exportada
como inferência nativa. Esses resultados não atingiram os critérios.

A expansão BJ separou prazos de números distratores e consultas sobre recursos
compostos. O decoder BL selecionou 233/252: suporte novo 20/36, informação
ausente nova 5/6 e memória contextual antiga 34/36. As demais famílias passaram.
BO ampliou verbos, posições, recursos e combinações de atributos; selecionou
235/252 no passo 500, com suporte novo 22/36, ausentes 5/6 e memória 34/36.
O mínimo para suporte novo é 33/36. Vocabulário conhecido agora participa do
treino; as 252 consultas completas permanecem separadas, mas não são teste cego.

O pipeline experimental BM combina classificação aprendida com candidatos do
decoder próprio. Selecionou 242/252 no passo 14000: 26/36 em suporte novo e
acerto integral nas demais famílias. Não é geração greedy nativa, não está
integrado à API e permanece explicitamente não promovível. As descrições do
protocolo nos artefatos identificam essa diferença. BQ continua esse experimento
com o corpus BO; o resultado ainda está pendente.

Na revisão documental, BK selecionou 399/410, sem falsos positivos; sua calibração
BN chegou a 400/410, ainda abaixo de AU (403/410 local e 41/48 no fluxo completo).
A auditoria refutou a hipótese de que a palavra que aparecia somente em negativos:
havia 400 exemplos positivos e 400 negativos entre as unidades examinadas.

BP inicia um piloto com InferBR, revisão fixa registrada no manifesto, licença
MIT preservada e 8190 pares apenas do split oficial de treino. A origem usa
geração semiautomática/automática com revisão na publicação, sem revisão local;
não foi marcada como diálogo humano nem como exemplo autoral. Os rótulos são
0 contradição, 1 confirmação e 2 neutro; neutro não vira ausência explícita.
A validação documental de 410 casos é preservada; teste e validação oficiais
não foram usados. A adequação dessa supervisão à revisão de projetos é experimental.

Os 219 testes Python passaram. Os relatórios BA, BD, BE, BL e BO tiveram todas
as páginas verificadas visualmente. Nenhum peso padrão foi promovido. As metas
por família, documentos completos e novas coortes independentes após o próximo
congelamento continuam obrigatórias.

## Auditoria de vocabulário e próxima comparação pareada — 17/11

A auditoria de BK2 contou zero unidades de treino com atesta e 20 na validação.
Isso é uma lacuna observada, sem provar causalidade. BS acrescenta quatro pares
positivos/negativos com esse verbo, em construções diferentes da moldura de
diagnóstico, preservando a validação e os corpora anteriores. O corpus contém
135335 exemplos de treino e os mesmos 410 de validação, incluindo os pares
públicos de inferência de BP. BS usa AU como origem e ainda está em execução.

Em BM, nove dos dez erros de resposta já tinham classificação errada. Duas
diferenças adicionais de classe entre fato e memória delegam ao mesmo decoder
e não implicam erro de resposta. BQ não superou a origem até a avaliação completa
4000 e foi interrompido para testar consultas pareadas. O último passo de treino
executado após essa avaliação não foi registrado; o checkpoint selecionado
permaneceu no passo zero, com 242/252. Nenhum tempo final foi inferido.

BT mantém metade do lote com consultas sobre o mesmo texto e respostas distintas;
a outra metade preserva a amostragem por estado. Os grupos são exclusivamente
de treino e exigem o mesmo corpo. BR ajusta a rede nativa inteira, com ancoragem
na origem própria, para testar o limite dos blocos antes congelados. BP, BS, BR
e BT estão pendentes. Nenhum deles é aprovado só pelo acerto agregado.

Os 221 testes Python passaram; os commits desta retomada mantêm a autoria
normal, sem atribuições adicionais. Nenhum peso padrão da API foi alterado.

BP encerrou no passo 6000 por platô, preservando o passo zero (403/410).
Os pesos do passo 6000 acertaram 382/410; a calibração BU chegou a 386/410
com um falso positivo e não recuperou a qualidade. Não foi promovida.
A seleção de erros de treino foi corrigida para respeitar os limiares da
origem calibrada; antes usava 0,5 fixo. Um teste cobre a diferença.
Os 222 testes Python passaram. BS, BR e BT seguem em execução.

## Exposição ao corpus ampliado — continuação de 17/11

A auditoria de amostragem de BS encontrou 291122 unidades e 86688 grupos de
texto. Destes, 1049 contêm atesta (1,21%) e 2989 são premissas públicas (3,45%).
A amostragem uniforme por fonte sorteia quatro grupos por lote; esses percentuais
são probabilidades teóricas, não a cobertura medida das execuções.

BS foi substituído após a avaliação completa do passo 3000. O melhor candidato
continuou no passo zero (403/410); o checkpoint 3000 acertou 399/410, com um
falso positivo. O passo final após essa avaliação e o tempo total não foram
inferidos. Os arquivos anteriores permanecem no disco.

BV parte desse último checkpoint salvo para 50000 passos solicitados, avaliação
a cada 2500 e paciência de 20 avaliações. Isso amplia a exposição em vez de
encerrar outra tentativa antes de uma quantidade equivalente ao corpus. O
otimizador reinicia, conforme o treinador registra. Não há promessa de melhora:
os mesmos critérios, revisão de documentos completos e novo teste independente
continuam necessários. Os valores planejados não são contagens executadas.

BR encerrou os 12000 passos e selecionou o passo 8000: 234/252, com suporte
novo 25/36, memória antiga 32/36 e naturais 8/9. Esses três grupos não passam;
fatos antigos 22/24 passam o piso de 90%. O relatório teve suas três páginas
verificadas visualmente. Não foi promovido.

BT selecionou provisoriamente o passo 8000 com 244/252: suporte novo 28/36,
acerto integral nas demais famílias. Precisa de pelo menos 33/36 em suporte.
Essa é uma medição intermediária do pipeline experimental, não de geração
greedy nativa. BT e o treino documental longo BV continuam em execução.

## Checagem do fluxo completo durante o ciclo longo — 17/11

BV/2500 selecionou 400/410, sem falsos positivos locais. Falham positivos
35/40, negativos 33/37 e conflitos 6/7. A calibração BW preservou os limiares
0,3/0,1 e não alterou o resultado. No documento completo, BW marcou 38/48,
com um falso positivo: um conflito em tail_conflict perdeu a evidência negativa
e terminou como atendido. Formato válido 48/48; isso não substitui correção
semântica. O checkpoint permanece reprovado e o treino longo continua.

A auditoria das 291122 unidades de treino encontrou 220392 entradas distintas
para requisito e texto da unidade, sem entradas exatamente iguais com rótulos
conflitantes. Isso refuta essa hipótese específica; não prova qualidade
semântica das anotações públicas nem generalização. Não foram usados exemplos
de avaliação nessa auditoria. Evidência em document-label-consistency.

BT encerrou 16000 passos, selecionando o passo 14000: 247/252, suporte novo
31/36 e acerto integral nas demais famílias. Ainda precisa de pelo menos 33/36
nessa família. BX continua a partir desse checkpoint salvo, com 32000 passos
solicitados, avaliação a cada 4000 e os mesmos contrastes pareados. O otimizador
reinicia. O pipeline permanece experimental e não está integrado à API.
Os números do plano BX são planejados; não representam treino já concluído.

## Primeiro congelamento aprovado em desenvolvimento — 17/11

BX encerrou no passo 24000 com 249/252 e suporte 33/36; todas as demais
famílias passaram integralmente. Encoder, decoder e classificador foram
congelados em conversation-independent-ca antes de construir 72 casos novos.
A primeira avaliação CA reprovou: suporte 34/36, fatos 3/12, conflito 5/6,
ausentes 6/6 e memória 11/12. Essa coorte passa agora a desenvolvimento;
qualquer consulta posterior não será apresentada como teste independente.
O candidato não foi promovido nem integrado à API.

BV concluiu 50000 passos sem superar a seleção do passo 2500 (400/410),
já reprovada pelo fluxo completo em BW. O aumento de exposição não resolveu
a revisão documental. A remoção de espaços nas bordas piorou a avaliação
do snapshot 12500 de 389 para 387; a codificação foi preservada. A interpolação
de pesos locais também não superou AU. Provas isoladas de construções com
atesta não equivalem ao resultado no fluxo completo.

O gerador de qualificadores foi ampliado e o corpus BZ está preparado,
com 24000 novas consultas de treino e os mesmos 252 casos de validação.
BZ ainda não foi usado para treinar BX. Os 225 testes Python passaram.
Uma comparação de decoders próprios sobre CA, já desenvolvimento, está
em andamento para investigar erros de cópia de números e nomes.

A comparação terminou: o decoder BR corrigiu os 12/12 fatos de CA, com
classificador BX congelado. Conflito (5/6) ainda falha; memória (11/12) passa;
suporte permanece 34/36 e ausentes 6/6. A combinação precisa ser reavaliada
nos 252 casos anteriores antes de qualquer adoção. CA já é desenvolvimento
nessa comparação, e será necessário outro teste novo após o próximo congelamento.

## Roteamento aprendido entre decoders — continuação de 17/11

O decoder BR sozinho reprovou a coorte anterior por memória 32/36. O roteamento
pelo tipo aprendido (fatos para BR, demais tipos para AQ) passou os critérios
dos 252 casos anteriores: 248/252, memória 36/36 e fatos 23/24. Não preservou
cada acerto individual; preservou os pisos por família e o legado. Em CA, já
desenvolvimento, marcou 68/72: fatos 12/12, suporte 34/36, ausentes 6/6,
conflito 5/6 e memória 11/12. Não há aprovação conjunta nem promoção.

TypeSelectedDecoder foi implementado no protótipo e reavaliado como CE.
Um teste verifica todos os seis tipos aprendidos e rejeita componentes
incompatíveis; os 226 testes Python passaram. O runtime da API não foi alterado.

CD iniciou um ajuste do decoder legado AQ com 5504 novas linhas de cópia
de nomes e ancoragem na própria origem. As histórias completas da avaliação
foram excluídas, e a validação anterior permanece intacta. A seleção por loss
desse componente não autoriza adoção: deve ser medido pelo roteamento completo.

CC iniciou um controle documental a partir de AU, retirando apenas os 8190
pares públicos do corpus BS e preservando os 410 casos de validação. Não há
conclusão causal antes do término. Ambos os treinos seguem em andamento.

Correção da interpretação: 11/12 em memória equivale a 91,7% e passa o piso
de 90%. Na combinação CE sobre CA, somente conflitos (5/6) bloqueiam a
conversa. O ajuste CD de nomes busca melhoria adicional, não é condição
necessária para satisfazer o piso de memória. Os resultados brutos não mudaram.

## Validação do ajuste de memória e seleção pela combinação — 17/11

CD terminou 6000 passos. Seus checkpoints selecionado (1000) e final (6000)
foram avaliados com o roteamento completo. Ambos corrigem o nome de CA e
marcam memória 12/12, mas reprovam o legado: troca de assunto 2/3 e tópico
2/4 nos casos anteriores. Foram rejeitados; memória 11/12 da origem AQ já
passa o piso. As três páginas do relatório nativo CD foram verificadas.

O treinador do classificador agora aceita um decoder factual próprio,
validado contra configuração e tokenizer, e seleciona pelos resultados do
roteamento completo. A avaliação declara learned_type_fusion_v2. Um teste
verifica a execução dos dois decoders e o modo incompatível; 227 testes
Python passaram. CG iniciou 12000 passos solicitados com 9221 novos exemplos
numéricos contrastivos e 324 avaliações, mantendo CA explicitamente como
desenvolvimento em famílias separadas. Pedidos completos reservados foram
excluídos do treino. Nenhum peso padrão foi alterado.

CI calibrou CC/1000 para limiares 0,3/0,5: 407/410 e todos os pisos locais
passaram, sem falsos positivos. O fluxo completo passou 36/36 originais e
36/36 da coorte extra; na coorte ampliada marcou 46/48, sem falsos positivos,
com 113/116 lotes exatos. Ainda falham tail_conflict e scope_long: 7/8 por
perfil, abaixo de 90%. Não houve aprovação conjunta ou promoção. CC e CG
continuam em execução; nova coorte independente será exigida após congelamento.

## Segundo teste pós-congelamento da conversa — 17/11

CG selecionou o passo 12000 com 316/324, passando todos os pisos das famílias
de desenvolvimento. A combinação inteira foi congelada em CJ: classificador,
encoder, decoder factual e decoder legado, com hashes e instante anteriores
à criação dos casos. A coorte nova foi excluída por contexto completo contra
os corpora CG, BZ e CD. Não há alegação de vocabulário inteiramente inédito.

A primeira avaliação CJ reprovou: suporte 25/36, fatos 12/12, conflitos 4/6,
ausentes 6/6 e memória 8/12; total 55/72. Memória aqui é 66,7% e realmente
não passa, diferentemente dos 11/12 anteriores. CJ agora é desenvolvimento;
consultas posteriores não serão apresentadas como novo teste independente.
Não houve promoção ou integração desses candidatos no runtime da API.

CC concluiu 20000 passos; o melhor snapshot e a calibração CI permanecem
preservados. A conversa ainda precisa generalizar a relação entre a pergunta
e os trechos relevantes, especialmente em recursos novos e valores próximos.
Mais acertos nas coortes consultadas não substituem essa evidência nova.

## Limites da calibração global e leitura por atenção — 17/11

CC confirmou o encerramento em 20000 passos e seleção no passo 1000. CI
permanece preservado. A auditoria de probabilidades mostra sobreposição entre
escores de evidência verdadeira e falsa nos dois estados; não existe limiar
que torne todos os bits locais corretos. Isso não é prova de erro nas anotações.

CK enumerou 227 regiões de limiar positivo e 231 negativas. A consolidação
foi conferida requisito por requisito contra as três avaliações completas
reais de CI, além dos totais 36/36, 36/36 e 46/48. Nenhum par global satisfez
simultaneamente os pisos locais, zero falsos positivos e todos os perfis
completos conhecidos. Não foram usados testes novos nessa seleção. Nenhum
checkpoint foi exportado por essa busca sem solução.

CL iniciou um ensaio de atenção residual à consulta, com encoder e cabeça
originais congelados. A projeção de saída começa zerada; o autoteste confirmou
logits idênticos à origem, e a avaliação inicial reproduziu 407/410. Somente
a atenção recebe gradientes, com preservação do professor próprio nos bits
de treino em que ele acerta. O objetivo é experimentar uma leitura contextual
mais flexível; o resultado não está garantido. O artefato é não promovível e
o loader da API não foi alterado. Receitas e hashes foram preservados.
