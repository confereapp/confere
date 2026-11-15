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
