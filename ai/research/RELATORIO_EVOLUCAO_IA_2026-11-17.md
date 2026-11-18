# Relatório de evolução do modelo de IA do Confere

**Data de referência:** 17 de novembro de 2026

**Situação:** experimental; meta de qualidade ainda não atingida.

**Escopo:** treinamento local, conversa contextual, análise de requisitos e integração com a API.

## 1. Resumo executivo

O projeto construiu um modelo próprio de linguagem, um classificador de evidências e a infraestrutura para executá-los localmente e disponibilizar operações autenticadas na API. Foram preparados dados, realizados diversos ciclos de treinamento e criados mecanismos de avaliação, rastreabilidade e prevenção de regressões.

Há resultados positivos em tarefas curtas e controladas. Entretanto, o sistema ainda não demonstrou confiabilidade em documentos acadêmicos reais. Os principais problemas são generalização fraca, memorização dos exemplos, interpretação incorreta de evidências e perda de capacidades anteriores durante novos ajustes.

**O avanço mais consistente até agora está na infraestrutura e no diagnóstico das falhas. Não há evidência para apresentar o modelo como um avaliador acadêmico autônomo confiável.** Os últimos candidatos foram reprovados e não substituíram os padrões da API.

**Atualização da segunda rodada (seção 5.6):** uma base nova treinada em 303 milhões de tokens, recuperação por BM25 e contrastes sem atalho produziram o primeiro classificador que supera os controles que leem só a consulta ou só o trecho nos testes públicos. A transferência para os TCCs reais continua sem ganho distinguível; nada foi promovido.

## 2. Objetivos e restrições adotadas

Os objetivos iniciais incluíram continuar o treinamento do modelo próprio, oferecer conversa contextual e integrar a análise de texto à API. Após os resultados nos PDFs, a investigação recente concentrou-se na relação entre requisito e evidência.

Foram respeitadas as restrições de execução no Mac, ausência de serviços pagos e ausência de download de pesos de modelos públicos pré-treinados. O treinamento usa pesos da própria linhagem do projeto. Isso não significa ausência de dados externos: o histórico inclui textos públicos, conversas anotadas, exemplos gerados na origem de alguns datasets e exemplos autorais.

As alterações foram feitas incrementalmente, com commits pequenos e sem atribuições adicionais nas mensagens. Não se apresenta uma estimativa de custo total, horas acumuladas ou número total de experimentos, pois este relatório não fez uma auditoria contábil completa desses valores.

## 3. O que foi construído

### Modelo e treinamento

Até a primeira rodada, a configuração própria usada como referência tinha aproximadamente **11,8 milhões de parâmetros**, seis camadas, largura 384, seis cabeças de atenção, vocabulário BPE de 1024 tokens e contexto de 1024 tokens. O treinamento local utiliza PyTorch e aceleração MPS.

Foram explorados treinamento de linguagem e instruções, tarefas de conversa com histórico, classificadores documentais, ajustes somente da camada de decisão e ajustes dos blocos do encoder. Também foram ensaiadas diferentes leituras da consulta, atenção bidirecional com máscara de padding e classificação conjunta em quatro estados. Essas variantes foram experimentais; nenhuma melhora pontual foi tratada como aprovação geral.

Na segunda rodada, uma base nova da mesma arquitetura, com vocabulário de 8.192 tokens (**17,3 milhões de parâmetros**), foi treinada do zero em 303 milhões de tokens da Wikipédia em português (seção 5.6).

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

### 5.6 Segunda rodada: base completa, recuperação e contrastes sem atalho

**Recuperação por BM25.** O pipeline passou a classificar só os k trechos mais parecidos com cada requisito e a decidir por maioria estrita; página sem texto virou aviso em vez de veto. Antes, um único lote espúrio entre centenas, ou uma página vazia, tornava qualquer requisito inconclusivo. Com k=8, 21 das 22 citações anotadas estão entre os trechos recuperados e o custo cai de 886 para 136 lotes. Sem `top_k`, o comportamento anterior é preservado.

**Base nova.** As bases anteriores viram cerca de 25 milhões de tokens em orçamentos de minutos. A nova usa o dump completo da Wikipédia em português (1,2 bilhão de caracteres), tokenizer por palavras com 8.192 tokens, janelas empacotadas e agenda cosseno: 18.500 passos em 2 h 45 min no MPS. Nos mesmos artigos de validação, fora do treino de todas as bases:

| Base | Bits por caractere |
| --- | ---: |
| `foundation-1024-2026-11-02` | 1,656 |
| `foundation-continue-n-2026-11-16` (melhor anterior) | 1,466 |
| `pt-full-8k-2026-11-17` | 1,260 |

As amostras continuam repetitivas; a melhora é de modelagem de linguagem, não de conversa.

**Atalho da consulta.** No InferBR, cada hipótese aparece com uma única premissa, e palavras de negação quase sempre indicam contradição (869 contradições, 157 implicações e 5 neutras). Por isso a consulta prevê o rótulo. Os contrastes novos acrescentam, para cada par, a negação da hipótese com rótulo invertido e premissas vizinhas por BM25 sem relação. No teste de contrastes, ler só a consulta (2.115/3.262) ou só o trecho (2.019) fica abaixo da classe majoritária (2.166).

| Classificador | Contrastes (3.262) | DF público (861; só consulta: 586) | Casos antigos | Acadêmico S / C / I | Falsas aprov. acadêmicas |
| --- | ---: | ---: | ---: | --- | ---: |
| Origem CI | — | 322 | 407/410 | 4/9 · 2/8 · 3/34 | 16 |
| EK (base nova, dados antigos) | 1.328 | 583 | 324/410 | 4/9 · 3/8 · 16/34 | 12 |
| **EL (base nova, contrastes)** | **2.657** | **609** | 398/410 | 4/9 · 4/8 · 21/34 | 6 |

EL é o primeiro classificador que supera os dois controles de atalho nos testes públicos: há evidência de que lê o trecho, ao menos em frases curtas.

**Requisitos no formato de checklist.** EM continuou de EL com 1.668 pares derivados das perguntas FaQuAD ("O texto deve informar ..."), parágrafo com a resposta contra parágrafo vizinho sem ela. A validação desses requisitos subiu de 168 para 284 de 392, e suporte e insuficiência acadêmicos melhoraram (até 7/9 e 32/34). A contradição caiu para 1–2/8, porque essa fonte não tem casos de `nao_atendido`. A seleção, fixada antes, manteve EL.

**TCCs reais (BM25, k=8).**

| Classificador | Status coincidentes | Falsas aprovações | Trechos marcados nos dois sentidos |
| --- | ---: | ---: | ---: |
| CI | 8/18 | 3 | 67/144 |
| EK | 11/18 | 5 | 25/144 |
| EL por página / por frase | 6/18 · 11/18 | 4 · 7 | — |
| EM último por página / por frase | 10/18 · 11/18 | 7 · 5 | — |

O classificador ficou mais discriminativo, mas não há ganho distinguível nos TCCs: com 18 critérios conhecidos, diferenças de 1 a 3 acertos estão dentro do ruído, e as falsas aprovações persistem.

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
- A última execução da suíte Python registrada passou com **263 testes**. Isso verifica o software, não a competência semântica do modelo.

## 8. Situação atual e próximos passos

**A meta de qualidade não foi atingida.** A base de software e de auditoria está mais madura que o desempenho do modelo. A segunda rodada removeu dois gargalos técnicos (base subtreinada e atalho da consulta nos dados públicos) e mostrou leitura da evidência em frases curtas, sem transferência demonstrada para documentos reais. Não há suporte para uso autônomo na avaliação de trabalhos ou aprovação de requisitos acadêmicos.

A hipótese mais sustentada pelos resultados é uma combinação de supervisão insuficiente e pouco diversa, diferença entre os dados controlados e o uso real, e limitações da representação aprendida. A contribuição exata de cada fator ainda não foi isolada de forma conclusiva.

A prioridade recomendada é ampliar relações revisadas em mais documentos, incluindo critérios que exigem explicar mecanismos, manter a mesma consulta com evidências de relações diferentes e obter uma segunda revisão das anotações. A avaliação deve continuar por documento, classe e grupo completo de consultas, com comparação a baselines simples e revisão semântica das citações.

Os gargalos restantes não se resolvem com código ou mais treino: faltam supervisão de `nao_atendido` no formato de requisito e uma avaliação real maior que 18 critérios. O próximo passo concreto é anotar de 5 a 10 TCCs, cerca de 100 critérios com os três rótulos, e usá-los como treino e avaliação separados por documento.

Mais passos ou mais variantes sobre os mesmos 90 pares não têm justificativa nos resultados atuais. Para uma entrega acadêmica, o sistema pode ser apresentado honestamente como protótipo experimental com análise de falhas e apoio à revisão humana; não como solução pronta cuja confiabilidade foi demonstrada.

## 9. Evidências e limites deste relatório

Este documento consolida artefatos existentes; a seção 5.6 resume treinos e avaliações da segunda rodada, registrados nos arquivos citados abaixo. Não repete testes de API. Os 263 testes citados correspondem à suíte executada ao final da segunda rodada. Não existe uma revisão humana independente completa dos rótulos nem um benchmark amplo de conversa aberta.

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
- `ai/PRETRAINING.md` e `output/ai/pretrain-full-8k-2026-11-17.json`: base completa.
- `ai/research/new-base-ek-2026-11-17.md`, `output/ai/evidence-mix-ek-2026-11-17.json` e `output/ai/evidence-mix-el-2026-11-17.json`: classificadores EK, EL e EM.

As fontes públicas e atribuições de dados permanecem registradas em `ai/data/licenses/`. Os textos integrais dos TCCs e os rastros privados não são reproduzidos neste relatório.
