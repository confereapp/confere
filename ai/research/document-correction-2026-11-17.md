# Correção da avaliação documental — 17/11/2026

O sistema continua experimental e não passou pelos critérios de promoção.
Os PDFs reais conhecidos não foram incorporados ao treino. As avaliações são
de desenvolvimento, com checklist e anotações editoriais; não são uma rubrica
oficial nem uma avaliação humana independente.

## Problemas corrigidos

- A validação por sentenças usava a leitura herdada da origem, mesmo quando o
  treino selecionava outra leitura. Agora treino e validação usam a mesma opção.
- A amostragem uniforme por texto subestimava consultas diferentes sobre o
  mesmo trecho. Agora a primeira unidade é uniforme e o segundo exemplo mantém
  o contraste de rótulo no mesmo texto, quando disponível.
- Os dados de critérios confundiam inclusão, implementação e disponibilidade.
  A versão `component_inclusion_v2` limita o alvo à inclusão no produto e contém
  controles em que um componente presente está temporariamente indisponível.
  A verificação de escopo se aplica somente às tarefas documentais relevantes.

Essas correções possuem testes de regressão. A suíte completa passou com 245
testes. Isso verifica o software, não comprova a qualidade semântica dos pesos.

## Histórico de dados preservado

Os corpora CU (`f5ff98549616f923a417b6f6a2ffaf65b49e69523a583aece85a7b916339e766`)
e CZ (`9c1c321221c817a03fbbb6a21d2872f500150846166a40b82184a9cfe57e043b`)
contêm essa conflatação e não servem como prova da distinção entre os conceitos.
Seus arquivos e resultados históricos foram preservados. As anotações dos PDFs
reais permanecem intactas. Não se deve comparar diretamente os totais de 750
casos antigos com os 590 casos da versão corrigida.

O corpus DD expandido tem SHA-256
`8641dc7c5da51ae0a5f420bbb23334b8cca504a7749b70cf3ba3a289da0c9785`.
O ajuste DD partiu de CI, selecionou o passo 2000 e acertou 464/590, com seis
falsas aprovações na validação sintética. Reprovou os pisos por família.

## Revisão das novas aprovações de DA

A atenção bidirecional elevou a coincidência de status no primeiro TCC, mas
coincidência de status não basta. Revisão editorial dos três novos positivos:

| Critério | Trecho retornado | Julgamento editorial |
| --- | --- | --- |
| R1: identificar limitação do legado | Resumo, página 1, descreve ineficiência e baixa satisfação | Suporte pertinente; há também trecho mais genérico na página 8 |
| R3: explicar atendimento a múltiplos clientes | Conclusão, página 8, afirma escalabilidade sem explicar o mecanismo | Evidência insuficiente para a aprovação |
| R4: descrever proteção de acesso | Conclusão, página 8, identifica autenticação de dois fatores | Suporte pertinente ao mecanismo solicitado |

As duas evidências pertinentes são alternativas às âncoras anotadas. A ausência
de uma âncora específica não invalida automaticamente uma alternativa correta.
Os demais trechos não receberam revisão semântica completa; não são aprovações
validadas. DA continua reprovado, sem promoção.

## Ensaio de relação explícita

DE congela o encoder próprio CI e aprende uma cabeça residual pequena a partir
de representações separadas da consulta e do trecho. A cabeça começa com saída
residual zero, cuja equivalência ao classificador de origem é verificada por
resposta antes do treino. Normalização e âncora de treinamento usam somente o
treino. O protocolo é experimental, possui verificação do hash da origem e é
rejeitado pelo carregador nativo da API. Nenhum peso externo foi baixado.

Resultados finais e rastros resumidos: `output/ai/document-correction-de-2026-11-17.json`.
DE selecionou o passo 750: 503/590, zero falsas aprovações nesse ponto, mas
reprovou os pisos e regrediu em casos antigos. Corrigiu o controle conhecido de
backup: inclusão → não atendido; registrar ausência → atendido; menção → atendido.
Esse controle é desenvolvimento, não prova independente.

Nos PDFs completos, DD e DE mantiveram 2/8 e 2/10 coincidências de status, com
conflitos em todos os 18 critérios. Logo, **o problema dos documentos reais não
foi resolvido**. A página fisicamente vazia do segundo PDF continua registrada;
ela não explica os conflitos também presentes no primeiro documento.

Não se recomenda mais passos sobre os mesmos templates como próxima ação.
O déficit restante exige exemplos de relação critério/trecho de prosa real,
com anotação revisada e divisão por documento, mantendo estes dois TCCs fora
do treino. As aprovações precisam de revisão semântica, não só de citação literal.
Os rastros integrais dos documentos ficam na pasta privada, com cobertura
contígua e offsets literais de citação verificados. Nenhum padrão da API foi
alterado; não houve teste HTTP desta solução.
