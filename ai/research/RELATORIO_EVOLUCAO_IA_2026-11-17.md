# Relatório de evolução do modelo de IA do Confere

**Data de referência:** 17 de novembro de 2026

**Situação:** experimental; meta de qualidade ainda não atingida.

**Escopo:** treinamento local, conversa contextual, análise de requisitos e integração com a API.

## 1. Resumo executivo

O projeto construiu um modelo próprio de linguagem, um classificador de evidências e a infraestrutura para executá-los localmente e disponibilizar operações autenticadas na API. Foram preparados dados, realizados diversos ciclos de treinamento e criados mecanismos de avaliação, rastreabilidade e prevenção de regressões.

Há resultados positivos em tarefas curtas e controladas. Entretanto, o sistema ainda não demonstrou confiabilidade em documentos acadêmicos reais. Os principais problemas são generalização fraca, memorização dos exemplos, interpretação incorreta de evidências e perda de capacidades anteriores durante novos ajustes.

**O avanço mais consistente até agora está na infraestrutura e no diagnóstico das falhas. Não há evidência para apresentar o modelo como um avaliador acadêmico autônomo confiável.** Os últimos candidatos foram reprovados e não substituíram os padrões da API.

**Atualização da segunda rodada (seção 5.6):** uma base nova treinada em 303 milhões de tokens, recuperação por BM25 e contrastes sem atalho produziram o primeiro classificador que supera os controles que leem só a consulta ou só o trecho nos testes públicos. A transferência para os TCCs reais continua sem ganho distinguível; nada foi promovido.

**Teste cego em 8 TCCs novos (seção 5.7, 18/11/2026):** nenhum classificador atingiu o critério fixado antes do teste. O melhor (EK) aprova com precisão de 63% (IC95% 52–73%) e perde para o controle que lê só a consulta (81%). O modelo continua experimental.

**Treino com artigos CC-BY anotados por IA (seção 5.8):** o classificador EN chegou a 76% de precisão, mas esse ganho vem de ter decorado com que frequência cada critério aparece. Com os dados equilibrados por critério (EO), o modelo passa a aprovar quase tudo. O gargalo é a leitura, não a quantidade de rótulos.

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

### 5.7 Teste cego em 8 TCCs novos

Oito TCCs de computação da Mauá de 2025 (handles 2231, 2232, 2234, 2236, 2239, 2241, 2242 e 2244), nunca usados em treino ou seleção, com OCR local e checklist genérico de 16 critérios (`ai/annotation/checklist-v1.txt`), somam 128 decisões. O critério de aprovação foi fixado antes de qualquer inferência: precisão de "atendido" de pelo menos 90% com intervalo de confiança, evidência correta em pelo menos 90% das aprovações e desempenho acima de BM25 sozinho e do controle que lê só a consulta.

**Rótulos.** Os 128 itens foram anotados de forma independente pela IA, sem ver saídas dos modelos. Uma pessoa anotou às cegas 2 TCCs (32 itens). Outra pessoa revisou as 62 aprovações e contradições da IA nos 6 TCCs restantes; mudou 1 status e acrescentou trechos em 18 itens. Os 34 itens "não encontrado" desses 6 TCCs não tiveram revisão humana. A concordância entre a anotação humana cega e a da IA foi baixa: 56% e kappa 0,18. Em 11 das 14 divergências, a pessoa marcou "atende" quando o tema aparecia no texto (por exemplo, "ausência inicial de testes automatizados" para o critério de testes automatizados), e a IA exigiu que o trabalho descrevesse o item. Para resolver isso, a regra de cada critério foi escrita em `ai/annotation/criterios-v1.md`: o TCC precisa fazer ou descrever o item, não só citar a palavra. Com essa regra, a pessoa adjudicou as 14 divergências. Ficou com a IA em 12 e, nos 2 casos de "implantado em produção", com "não encontrado" em vez de "contradiz". Os rótulos finais têm 74 atendido, 52 não encontrado e 2 contradiz.

| Sistema (top-8, maioria) | Aprovações | Precisão (IC95%), rótulos finais | Evidência correta nas aprovações certas |
| --- | ---: | ---: | ---: |
| Sempre aprovar | 128 | 58% (49–66) | — |
| BM25 sozinho (limiar entre documentos) | 105 | 61% (51–70) | — |
| Só a consulta (maioria do critério nos outros TCCs) | 84 | **81% (71–88)** | — |
| CI | 81 | 57% (46–67) | 80% |
| EK | 81 | 63% (52–73) | 77% |
| EL | 57 | 46% (33–58) | 69% |
| EM (último) | 111 | 59% (49–67) | 52% |

"Evidência correta" quer dizer que alguma citação do modelo coincide com um trecho anotado, ou cobre pelo menos 60% dos termos dele. O cálculo está em `ai/scripts/score_blind.py`. Com os rótulos antes da adjudicação (em que a pessoa aceitava menções), o EK chegava a 74% e o controle da consulta a 86%. A ordem dos sistemas não mudou.

**Leitura.** Nenhum candidato chega perto de 90%. Nenhum supera o controle que só lê a consulta: nesse checklist, saber qual critério está em jogo prevê melhor do que ler o TCC. O EL, que era o melhor nos contrastes públicos, foi o pior aqui. A conclusão é que o classificador ainda não transfere a leitura de evidência para documentos reais, e o modelo permanece experimental.

### 5.8 Treino com artigos SciELO anotados por IA

Os TCCs da Mauá não têm licença de treino confirmada, então os dados novos vieram de 23 artigos em português do SciELO Brasil, de seis periódicos, com licença CC BY conferida em cada artigo (`ai/scripts/fetch_scielo.py`, `ai/data/licenses/SciELO-NOTICE.md`). Foram 368 itens (artigo × critério do mesmo checklist), anotados por duas instâncias de IA independentes seguindo `criterios-v1.md`, sem revisão humana. A concordância foi de 96% (kappa 0,92), mas as duas anotações vêm do mesmo modelo e não são independentes no sentido estatístico. Entraram no treino só os 354 itens em que as duas concordaram (`ai/scripts/prepare_checklist_training.py`). Não houve nenhum "contradiz".

- **EN**: continua de EL, com 573 linhas de checklist em uma mistura de 7.268 (configuração do EM), e 3 artigos ficaram para validação.
- **EO**: igual, mas com atendido e insuficiente equilibrados por critério (256 linhas). Os critérios sem nenhum positivo nos artigos saem.

| Sistema | Aprovações | Precisão (IC95%) | Concordância com a regra da frequência nos artigos |
| --- | ---: | ---: | ---: |
| Só a consulta (TCCs) | 84 | 81% (71–88) | — |
| Regra "aprova se o critério é atendido na maioria dos artigos" | 72 | 82% | — |
| EL (ponto de partida) | 57 | 46% (33–58) | — |
| EN selecionado (passo 4.000) | 63 | 76% (64–85) | 97/128 |
| EN último | 66 | 77% (66–86) | 96/128 |
| EO selecionado (= EL) / último | 57 · 126 | 46% · 59% | — |

Com rótulos finais e top-8 por maioria, o EN subiu 30 pontos sobre o EL, mas reproduz em 3/4 dos itens uma regra que nem lê o texto: aprova os critérios que costumam ser atendidos em artigos (objetivo, motivação, metodologia, dados) e reprova os que nunca são (testes com usuários, testes automatizados, custos). Essa regra sozinha tem 82% de precisão. A evidência correta nas aprovações certas também caiu (58% contra 69% do EL). Quando o atalho é retirado (EO), a seleção fica no passo 0 e o último passo aprova 126 de 128 itens. O modelo próprio não aprende a separar trecho que cumpre de trecho que só menciona o tema, nem com rótulos de checklist no formato real. O limite está na capacidade de leitura do modelo, não na falta de rótulos.

### 5.9 Abstenção por confiança

Teste: o modelo decide só quando muitos dos 8 trechos concordam e manda o resto para "revisar" (`ai/scripts/abstention_blind.py`). Escore do item = fração dos 8 trechos com P(atendido) ≥ 0,5. Os limiares foram escolhidos deixando um TCC de fora: alvo de 90% de precisão nos outros 7, aplicado no oitavo.

| Modelo | Aprovações | Precisão (IC95%) | Itens decididos | Acerto nos decididos |
| --- | ---: | ---: | ---: | ---: |
| EK, unanimidade (no próprio conjunto) | 9 | 100% (70–100) | 22% | 93% |
| EK, limiar fora da amostra | 11 | 82% (52–95) | 23% | 87% (70–95) |
| EN | nenhum limiar atinge 90% nos outros TCCs | — | — | — |

A confiança separa pouco: no EK, subir o limiar leva a precisão de 68% a 83% só quando restam 18 aprovações. Fora da amostra, a abstenção não chega a 90%, e ela decide menos de um quarto dos itens. Abstenção não substitui um modelo que leia melhor; ela pode ser somada a uma base mais forte.

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
- A última execução da suíte Python registrada passou com **265 testes**. Isso verifica o software, não a competência semântica do modelo.

## 8. Situação atual e próximos passos

**A meta de qualidade não foi atingida.** A base de software e de auditoria está mais madura que o desempenho do modelo. A segunda rodada removeu dois gargalos técnicos (base subtreinada e atalho da consulta nos dados públicos) e mostrou leitura da evidência em frases curtas, sem transferência demonstrada para documentos reais. Não há suporte para uso autônomo na avaliação de trabalhos ou aprovação de requisitos acadêmicos.

A hipótese mais sustentada pelos resultados é uma combinação de supervisão insuficiente e pouco diversa, diferença entre os dados controlados e o uso real, e limitações da representação aprendida. A contribuição exata de cada fator ainda não foi isolada de forma conclusiva.

A prioridade recomendada é ampliar relações revisadas em mais documentos, incluindo critérios que exigem explicar mecanismos, manter a mesma consulta com evidências de relações diferentes e obter uma segunda revisão das anotações. A avaliação deve continuar por documento, classe e grupo completo de consultas, com comparação a baselines simples e revisão semântica das citações.

Os gargalos restantes não se resolvem com código ou mais treino: faltam supervisão de `nao_atendido` no formato de requisito e uma avaliação real maior que 18 critérios. A anotação de 8 TCCs (128 critérios, seção 5.7) foi feita e confirmou o diagnóstico: nenhum modelo supera o controle da consulta. O critério de “atende” foi padronizado em `ai/annotation/criterios-v1.md` depois de um kappa de 0,18 entre pessoa e IA, e as divergências foram adjudicadas. Os 8 TCCs agora são o conjunto fixo de teste. Treinar com 354 itens de artigos CC BY anotados por IA (seção 5.8) só ensinou a frequência de cada critério. Mais rótulos do mesmo tipo não devem resolver: o modelo precisa de mais capacidade de leitura, seja com uma base maior, seja com pré-treino mais longo, antes de novos ajustes de classificação. A abstenção por confiança (seção 5.9) também não chega a 90% fora da amostra. Por exigência do curso (modelo próprio, treinado do zero), o próximo passo é uma base própria maior: 54 milhões de parâmetros (largura 640, 10 camadas), pré-treinada na mesma Wikipédia em português com cerca de 1 bilhão de tokens.

Mais passos ou mais variantes sobre os mesmos 90 pares não têm justificativa nos resultados atuais. Para uma entrega acadêmica, o sistema pode ser apresentado honestamente como protótipo experimental com análise de falhas e apoio à revisão humana; não como solução pronta cuja confiabilidade foi demonstrada.

## 9. Evidências e limites deste relatório

Este documento consolida artefatos existentes; a seção 5.6 resume treinos e avaliações da segunda rodada, registrados nos arquivos citados abaixo. Não repete testes de API. Os 265 testes citados correspondem à suíte executada ao final da segunda rodada. A revisão humana dos rótulos dos TCCs é parcial (seção 5.7); não existe revisão independente completa dos rótulos nem um benchmark amplo de conversa aberta.

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
- `ai/annotation/`, `ai/scripts/score_blind.py`, `output/ai/blind-v1-score-final-2026-11-18.json` e `blind-v1-score-ia-2026-11-18.json`: teste cego em 8 TCCs.
- `ai/scripts/fetch_scielo.py`, `ai/scripts/prepare_checklist_training.py`, `ai/data/licenses/SciELO-NOTICE.md`, `output/ai/checklist-scielo-en-2026-11-18.json`, `checklist-scielo-eo-2026-11-18.json` e `blind-v1-score-en-eo-2026-11-18.json`: EN e EO.
- `ai/research/new-base-ek-2026-11-17.md`, `output/ai/evidence-mix-ek-2026-11-17.json` e `output/ai/evidence-mix-el-2026-11-17.json`: classificadores EK, EL e EM.

As fontes públicas e atribuições de dados permanecem registradas em `ai/data/licenses/`. Os textos integrais dos TCCs e os rastros privados não são reproduzidos neste relatório.
