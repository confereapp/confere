# Ajuste do encoder — DI/DJ

Os ensaios anteriores ajustavam somente cabeças residuais sobre o encoder
congelado. Nesta etapa foram ajustados os blocos do encoder próprio e a cabeça
nativa, mantendo embeddings congelados, mesma origem CI, matriz DG, seed 42,
limiares, leitura e orçamento de 1000 passos. Nenhum peso externo foi baixado.

O treinador nativo agora admite parágrafos integrais e duplas da mesma consulta
com trechos/rótulos diferentes. Os padrões anteriores permanecem os mesmos.
O teste verifica equivalência da amostragem com a sonda anterior. Suíte completa:
260 testes passaram; diff sem erros.

DI usa somente as duplas. DJ acrescenta a opção existente de replay de erros
iniciais do treino: 139 unidades, das quais 72 acadêmicas. Metade de cada lote
revisita esse conjunto. Nenhum erro da validação foi usado para escolher exemplos.
Como o replay sobrescreve metade do lote, somente a outra metade mantém as duplas.

## Resultados de desenvolvimento

DI selecionou o passo 250; DJ manteve a origem (passo zero). Ambos reprovaram.
A soma maior de acertos de passos posteriores não supera a reprovação por classe.

| Classe acadêmica | Origem | DI selecionado |
| --- | ---: | ---: |
| Suporte | 4/9 | 2/9 |
| Contradição | 2/8 | 1/8 |
| Informação insuficiente | 3/34 | 24/34 |

Portanto, o ganho agregado veio principalmente de inconclusivos, com piora nas
outras duas classes. Não é uma melhora suficiente da compreensão documental.
O replay de erros não superou a origem na seleção. Esses resultados não
justificam continuar aumentando passos ou aprovar os checkpoints.

## Limites e conclusão operacional

A reserva de 54 pares dos três documentos novos continua sem inferência.
Nenhum padrão da API foi alterado e não houve teste HTTP ou repetição dos PDFs.
As anotações são editoriais locais, sem segunda revisão humana independente.
Os dados acadêmicos ainda são somente 90 pares de treino em cinco documentos.

A hipótese de que bastava destravar o encoder não se confirmou como solução.
O próximo investimento deve ser supervisão mais ampla e diversa, com relações
contrastivas anotadas, em vez de novas variantes sobre estes mesmos 90 pares.
Ainda não há evidência para afirmar que o modelo atende a meta de qualidade.

Rastros resumidos: `output/ai/matrix-encoder-di-dj-2026-11-17.json`.
