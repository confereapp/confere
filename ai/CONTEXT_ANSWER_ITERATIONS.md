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
