# Relatório de evolução do modelo de IA do Confere

**Data de referência:** 17 de novembro de 2026

**Situação:** experimental; meta de qualidade ainda não atingida.

**Escopo:** treinamento local, conversa contextual, análise de requisitos e integração com a API.

## 1. Resumo executivo

O projeto construiu um modelo próprio de linguagem, um classificador de evidências e a infraestrutura para executá-los localmente e disponibilizar operações autenticadas na API. Foram preparados dados, realizados diversos ciclos de treinamento e criados mecanismos de avaliação, rastreabilidade e prevenção de regressões.

Há resultados positivos em tarefas curtas e controladas. Entretanto, o sistema ainda não demonstrou confiabilidade em documentos acadêmicos reais. Os principais problemas são generalização fraca, memorização dos exemplos, interpretação incorreta de evidências e perda de capacidades anteriores durante novos ajustes.

**O avanço mais consistente até agora está na infraestrutura e no diagnóstico das falhas. Não há evidência para apresentar o modelo como um avaliador acadêmico autônomo confiável.** Os últimos candidatos foram reprovados e não substituíram os padrões da API.

## 2. Objetivos e restrições adotadas

Os objetivos iniciais incluíram continuar o treinamento do modelo próprio, oferecer conversa contextual e integrar a análise de texto à API. Após os resultados nos PDFs, a investigação recente concentrou-se na relação entre requisito e evidência.

Foram respeitadas as restrições de execução no Mac, ausência de serviços pagos e ausência de download de pesos de modelos públicos pré-treinados. O treinamento usa pesos da própria linhagem do projeto. Isso não significa ausência de dados externos: o histórico inclui textos públicos, conversas anotadas, exemplos gerados na origem de alguns datasets e exemplos autorais.

As alterações foram feitas incrementalmente, com commits pequenos e sem atribuições adicionais nas mensagens. Não se apresenta uma estimativa de custo total, horas acumuladas ou número total de experimentos, pois este relatório não fez uma auditoria contábil completa desses valores.

## 3. O que foi construído

### Modelo e treinamento

A configuração própria usada como referência nas etapas recentes tem aproximadamente **11,8 milhões de parâmetros**, seis camadas, largura 384, seis cabeças de atenção, vocabulário BPE de 1024 tokens e contexto de 1024 tokens. O treinamento local utiliza PyTorch e aceleração MPS.

Foram explorados treinamento de linguagem e instruções, tarefas de conversa com histórico, classificadores documentais, ajustes somente da camada de decisão e ajustes dos blocos do encoder. Também foram ensaiadas diferentes leituras da consulta, atenção bidirecional com máscara de padding e classificação conjunta em quatro estados. Essas variantes foram experimentais; nenhuma melhora pontual foi tratada como aprovação geral.

### Pipeline documental

O pipeline extrai texto, divide documentos em unidades e lotes, consulta o classificador, reúne evidências e produz uma decisão por requisito. O contrato evoluiu para suportar múltiplas evidências, inclusive opostas. As decisões principais são `atendido`, `nao_atendido` e `evidencia_insuficiente`.

A rastreabilidade inclui identificação da fonte, hashes, página/unidade e posições das citações no texto extraído. Foram implementadas auditorias para verificar cobertura contígua e correspondência literal das citações. **Essas verificações comprovam a origem do texto citado, não que ele sustente semanticamente a conclusão.**

### Integração com a API

O código contém operações autenticadas `ai.chat` e `ai.reviewText`, implementadas como mutations tRPC. Há validação de entradas e saídas e serviço de execução dos módulos Python locais. A integração técnica e a qualidade do modelo são dimensões diferentes: uma requisição válida não comprova uma resposta correta.

Nesta consolidação, a existência das operações foi conferida no código. Não foi repetido teste HTTP, não foi verificada uma sessão autenticada atual e não foi auditado o checkpoint efetivamente ativo em um servidor em execução.

## 4. Dados utilizados e evolução da avaliação

Inicialmente foram usados muitos cenários autorais: fatos explícitos, ausência de informação, conflitos, memória contextual, presença de recursos e requisitos documentais. Esses exemplos permitem controlar o alvo, mas podem ensinar padrões específicos da redação.

Duas fontes públicas relevantes nas etapas recentes foram:

- **InferBR:** supervisão auxiliar de inferência textual em português, distinguindo suporte, contradição e neutralidade. Não é um conjunto de trabalhos acadêmicos avaliados por rubricas.
- **FaQuAD:** leitura de textos em português no domínio do ensino superior. Foi usado tanto em um piloto extrativo quanto como fonte de parágrafos preservados para relações editoriais locais. Os novos rótulos de requisito/relação não são anotações oficiais do dataset.

As revisões, hashes, licenças e adaptações foram registrados. Parte das anotações é editorial local e ainda não possui segunda revisão humana independente.

Na matriz mais recente, foram mantidas **65 consultas contra três parágrafos cada: 195 pares**. Cada consulta tem rótulos diferentes conforme o trecho. Uma consulta constante foi excluída explicitamente, sem alterar seus rótulos para criar contraste artificial.

| Separação acadêmica | Documentos | Consultas | Pares |
| --- | ---: | ---: | ---: |
| Treino | 5 | 30 | 90 |
| Validação de desenvolvimento | 3 | 17 | 51 |
| Reserva nova, sem inferência | 3 | 18 | 54 |

Há ainda 501 exemplos autorais de replay para retenção e 410 casos antigos de validação. Esses 410 casos são desenvolvimento conhecido, não um teste independente novo. Os dois TCCs usados na análise documental real não foram incorporados aos treinos recentes.

A separação agrupa documentos inteiros e frases lexicalmente semelhantes, inclusive entre os dois lados dos pares. Isso reduz vazamento direto, mas não elimina toda paráfrase ou possível exposição histórica. As buscas registradas de sequências de oito palavras não encontraram ocorrências dos exemplos reservados nos corpora ancestrais auditados; esse resultado não prova ineditismo absoluto.

## 5. Resultados observados

Os números abaixo pertencem a conjuntos e versões diferentes. **Não devem ser somados ou tratados como uma única curva de acurácia do projeto.**

### 5.1 Conversa contextual

| Avaliação | Resultado | Interpretação |
| --- | ---: | --- |
| Conjunto combinado conhecido, CG | 316/324 — 97,5% | Aprovação nos critérios locais daquele conjunto |
| Conjunto novo de conversas sintéticas, CJ | 55/72 — 76,4% | Reprovação fora do conjunto conhecido |

No CJ, fatos explícitos tiveram 12/12, informação ausente 6/6, suporte 25/36, conflitos 4/6 e memória 8/12. Portanto, há capacidade em tarefas simples, mas desempenho irregular quando a consulta exige relação contextual, conflito ou memória. Esses testes não medem conversa aberta geral.

### 5.2 Documentos controlados

A referência documental CI alcançou **407/410 — 99,3%** nos casos controlados, passando pelos pisos locais de formato, referências e desempenho por perfil. Esse resultado é positivo no seu escopo, mas não previu o comportamento nos PDFs reais.

### 5.3 Documentos reais

Foram processados dois TCCs conhecidos, com checklists editoriais experimentais:

| Documento | Páginas | Critérios | Lotes | Status coincidentes | Critérios com conflito |
| --- | ---: | ---: | ---: | ---: | ---: |
| Gestão documental de RH | 10 | 8 | 128 | 2 | 8 |
| Práticas ágeis e arquitetura | 60 | 10 | 758 | 2 | 10 |

Todos os 18 resultados da avaliação inicial foram inconclusivos. Os quatro status coincidentes já eram inconclusivos no checklist: isso **não representa quatro análises semanticamente corretas**. Houve evidências irrelevantes, como folha de rosto indicada para um critério sobre percentuais de commits.

O sistema encontrou um trecho pertinente sobre autenticação de dois fatores, mas outras classificações opostas impediram a conclusão. O segundo PDF possui uma página fisicamente vazia, preservada como limitação de cobertura; ela não explica os conflitos presentes também no primeiro PDF.

Não houve aprovações finais indevidas nessa rodada, mas isso decorreu de saídas inconclusivas e não demonstra ausência de evidências falsas. A auditoria mecânica de cobertura e citações passou.

Uma variante bidirecional produziu novas aprovações no primeiro TCC. Na revisão editorial, duas tinham suporte pertinente alternativo; outra citava uma afirmação genérica de escalabilidade sem explicar atendimento a múltiplos clientes. A variante também foi rejeitada. Ensaios posteriores voltaram a manter conflitos nos 18 critérios.

### 5.4 Leitura pública e controles contra atalhos

No piloto extrativo FaQuAD, o modelo respondeu corretamente a **0/111** perguntas reservadas, sob o critério de resposta extrativa adotado. Não houve ganho demonstrado nessa tarefa.

No piloto de relações DF, o candidato alcançou **521/861 respostas completas corretas** em inferência textual, contra 322/861 da origem. Entretanto, um classificador simples que lia somente a consulta, ignorando a evidência, acertou 586/861. O aparente ganho não superou um controle de atalho.

Nos nove pares acadêmicos reservados desse piloto, o candidato acertou somente um e fez cinco falsas aprovações. Nos casos antigos, caiu de 407/410 para 331/410. Foi rejeitado.

Há uma correção histórica importante: o resumo original de DF registrava 523/861 coincidências de status. Exigir também evidências corretas reduz o resultado a 521/861. A seleção dos pesos já usava o critério completo; o erro estava no resumo diagnóstico, não na escolha de checkpoints.

### 5.5 Memorização e ajustes mais recentes

Com a matriz de consultas fixas e parágrafos diferentes, os últimos pontos de dois ensaios apresentaram:

| Ensaio | Treino acadêmico | Validação acadêmica | Consultas com todos os pares corretos na validação |
| --- | ---: | ---: | ---: |
| Cabeça residual não linear, DG | 86/90 | 25/51 | 2/17 |
| Cabeça linear menor, DH | 90/90 | 21/51 | 1/17 |

São pontos finais **rejeitados**, usados para diagnóstico; a seleção manteve a origem nos dois ensaios. A diferença entre treino e validação evidencia memorização. Reduzir a cabeça de 295554 para 4614 parâmetros não resolveu a generalização.

Depois, DI ajustou também os blocos do encoder próprio. DJ repetiu o ajuste com replay de erros encontrados somente no treino. Ambos executaram 1000 passos e reprovaram. DI selecionou o passo 250; DJ manteve o passo zero.

| Classe acadêmica na mesma validação | Origem | DI selecionado |
| --- | ---: | ---: |
| Suporte | 4/9 | 2/9 |
| Contradição | 2/8 | 1/8 |
| Informação insuficiente | 3/34 | 24/34 |

O ganho agregado veio principalmente de inconclusivos, com piora nas classes que sustentam uma decisão. Destravar o encoder e repetir erros não se confirmaram como solução suficiente.

## 6. Principais falhas identificadas

1. **Generalização fraca:** bom desempenho em templates não se sustenta em documentos ou formulações novas.
2. **Memorização:** o ajuste consegue acertar os pares de treino sem transferir essa capacidade para outros documentos.
3. **Evidências semanticamente incorretas:** uma citação pode existir literalmente e ainda ser irrelevante para o requisito.
4. **Acúmulo de conflitos:** classificações espúrias ao longo de documentos extensos anulam evidências pertinentes.
5. **Confusão entre tarefas:** mencionar backup, possuir backup e registrar ausência de backup são exigências distintas. O modelo nem sempre separa essas relações.
6. **Regressões:** alguns ajustes melhoram uma família enquanto prejudicam capacidades anteriores.
7. **Atalhos lexicais e desbalanceamento:** o modelo pode associar palavras da consulta a rótulos ou aumentar respostas inconclusivas sem interpretar a evidência.
8. **Cobertura limitada:** OCR pode ter erros; imagens, tabelas e estrutura visual não foram demonstradas como compreendidas.

Os resultados também expuseram falhas do processo de desenvolvimento, não apenas dos pesos. Houve uma validação que ignorava a leitura selecionada no treino, amostragem que subestimava consultas repetidas sobre o mesmo texto, dados que confundiam inclusão/implementação/disponibilidade e resumo de métricas permissivo demais. Esses problemas foram corrigidos e os artefatos anteriores preservados. Os corpora com escopo antigo não servem como prova das distinções corrigidas.

## 7. Pontos positivos

- Existe uma infraestrutura local funcional para preparar dados, treinar, selecionar e inspecionar resultados, sem dependência de pesos externos.
- O modelo demonstra capacidades em tarefas curtas e explícitas, mesmo sem confiabilidade geral.
- A rastreabilidade permite localizar a origem de uma citação e investigar decisões erradas concretas.
- Os testes revelaram mecanismos de falha específicos, em vez de atribuir tudo a falta de passos de treinamento.
- Foram preservados conjuntos conhecidos, ensaios rejeitados, hashes e diferenças entre versões dos dados.
- A avaliação evoluiu para exigir resposta e evidências corretas, desempenho por família e retenção, além de controles que ignoram uma das entradas.
- A reserva nova permanece sem inferência. O avaliador bloqueia seu uso sem validação e retenção aprovadas e vincula a autorização ao checkpoint e ao corpus corretos.
- A última execução da suíte Python registrada passou com **260 testes**. Isso verifica o software, não a competência semântica do modelo.

## 8. Situação atual e próximos passos

**A meta de qualidade não foi atingida.** A base de software e de auditoria está mais madura que o desempenho do modelo. Não há suporte para uso autônomo na avaliação de trabalhos ou aprovação de requisitos acadêmicos.

A hipótese mais sustentada pelos resultados é uma combinação de supervisão insuficiente e pouco diversa, diferença entre os dados controlados e o uso real, e limitações da representação aprendida. A contribuição exata de cada fator ainda não foi isolada de forma conclusiva.

A prioridade recomendada é ampliar relações revisadas em mais documentos, incluindo critérios que exigem explicar mecanismos, manter a mesma consulta com evidências de relações diferentes e obter uma segunda revisão das anotações. A avaliação deve continuar por documento, classe e grupo completo de consultas, com comparação a baselines simples e revisão semântica das citações.

Mais passos ou mais variantes sobre os mesmos 90 pares não têm justificativa nos resultados atuais. Para uma entrega acadêmica, o sistema pode ser apresentado honestamente como protótipo experimental com análise de falhas e apoio à revisão humana; não como solução pronta cuja confiabilidade foi demonstrada.

## 9. Evidências e limites deste relatório

Este documento consolida artefatos existentes; não executa novos treinos, não reavalia pesos nem repete testes de API. Os 260 testes citados correspondem à última suíte registrada, não a uma execução feita só para redigir este texto. Não existe uma revisão humana independente completa dos rótulos nem um benchmark amplo de conversa aberta.

Registros principais no repositório:

- `ai/CONTEXT_ANSWER_ITERATIONS.md`: histórico das iterações e correções.
- `ai/runs/conversation-combined-cg-2026-11-17/evaluation.json` e `conversation-independent-cj-2026-11-17/evaluation.json`: conversa.
- `ai/runs/evidence-calibrated-ci-2026-11-17/evaluation.json`: avaliação documental controlada.
- `ai/research/real-documents-cq.md`: avaliação dos dois PDFs reais.
- `ai/research/document-correction-2026-11-17.md`: correções e ensaios documentais.
- `ai/research/relation-pilot-df-2026-11-17.md`: fontes públicas e controle de atalhos.
- `ai/research/paragraph-matrix-dg-dh-2026-11-17.md`: memorização e reserva.
- `ai/research/matrix-encoder-di-dj-2026-11-17.md`: ajuste do encoder por classe.
- `output/ai/relation-df-metric-correction-2026-11-17.json`: correção do resumo histórico.

As fontes públicas e atribuições de dados permanecem registradas em `ai/data/licenses/`. Os textos integrais dos TCCs e os rastros privados não são reproduzidos neste relatório.
