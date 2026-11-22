# Trechos do artigo sobre a IA (revisados)

Texto para colar nas seções 4.4, 5.2 e nas Considerações Finais. Os números vêm de
`output/ai/classifier-w640-2026-11-19.json` e da seção 5.10 de
`ai/research/RELATORIO_EVOLUCAO_IA_2026-11-17.md`. Um segundo teste cego, com TCCs de outras
universidades, está em andamento; se ele terminar antes da entrega, os números da seção 5.2
devem ser atualizados.

## 4.4 Modelo de IA

O Confere usa um modelo de linguagem próprio, treinado do zero, sem pesos de terceiros. É um
transformer do tipo GPT com 60,4 milhões de parâmetros (10 camadas, largura 640), tokenizador BPE
de 8.192 tokens e janela de contexto de 1.024 tokens. O pré-treino usou a Wikipédia em português,
num total de 983 milhões de tokens. Rodou em 8,7 horas numa GPU T4 gratuita do Kaggle, e a perda
de validação caiu de 3,17 (base anterior, menor) para 2,68.

Sobre essa base foi ajustado um classificador de evidências, chamado FN. O FN não escreve texto.
Para cada requisito, um buscador lexical (BM25) seleciona os 8 trechos do documento mais
parecidos com o requisito. O classificador diz se cada trecho sustenta o requisito, e o requisito
recebe "atendido" quando a maioria dos trechos o sustenta. O ajuste foi feito em duas etapas: a
primeira com pares de frases públicos e sintéticos, a segunda com o mesmo checklist aplicado a 23
artigos científicos de licença aberta (SciELO), anotados por IA.

No aplicativo, o resultado é uma proposta para o professor conferir. O sistema marca "atendido"
com a citação e a página do trecho; os demais requisitos ficam como "revisar". Ele nunca marca um
requisito como "faltando", porque o classificador não demonstrou saber afirmar ausência. A
justificativa mostrada ao lado de cada requisito é um texto fixo montado a partir da decisão e das
páginas citadas, não uma explicação gerada pelo modelo.

## 5.2 Teste cego

Para medir o classificador em documentos reais, anotamos 8 TCCs de computação do repositório
público da Mauá com um checklist de 16 critérios, num total de 128 decisões. Esses TCCs nunca foram
usados em treino nem na escolha do modelo. Os rótulos foram feitos por uma pessoa e por uma IA, de
forma independente, e as divergências foram resolvidas segundo critérios escritos antes da
comparação.

O FN aprovou 45 critérios e acertou 42, uma precisão de 93% (intervalo de confiança de 95%: 82% a
98%). Como comparação, uma regra que não lê o documento e só aprova os critérios que costumam ser
atendidos nos outros TCCs acerta 81%. O FN também separa trabalhos dentro do mesmo critério: entre
os itens que essa regra aprovaria, os que o FN aprova estavam certos em 95% dos casos, e os que ele
recusa, em 65%.

O resultado tem limites claros:

- O FN aprova só 57% dos critérios que estão atendidos; o resto vai para revisão humana.
- Em 67% das aprovações certas, a citação mostrada coincide com o trecho anotado. Nas demais, a
  decisão está certa, mas o trecho citado não é o melhor.
- A amostra é pequena: 8 trabalhos de um único curso.
- O mesmo conjunto foi usado para comparar vários modelos ao longo do projeto, o que tende a
  deixar o melhor resultado um pouco otimista.

Antes do teste, fixamos como meta precisão de pelo menos 90% com intervalo de confiança e citação
correta em pelo menos 90% das aprovações. A precisão chegou a 93%, mas o limite inferior do
intervalo (82%) e a citação correta (67%) ficaram abaixo da meta. Por isso o modelo continua
classificado como experimental.

## Considerações finais (frase de limitações)

A IA do Confere funciona como apoio à conferência, não como avaliador: ela aprova com boa precisão
pouco mais da metade dos requisitos atendidos, deixa o restante para o professor, não gera
explicações próprias e foi validada em apenas 8 TCCs de um curso, o que pede um teste maior antes
de qualquer uso sem revisão humana.
