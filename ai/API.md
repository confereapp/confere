# Integração local experimental com a API

A branch `ai/treinamento-integracao-api` inicia a ponte tRPC → processo Python →
checkpoint próprio. A chamada autenticada `ai.reviewText` recebe texto e requisitos
manuais e retorna propostas com evidências, cobertura, hash dos pesos e revisão
humana obrigatória. Não altera projetos, submissões, resultados ou percentuais.

## Executar no Mac

O ambiente `ai/.venv` deve estar instalado com `requirements-train.lock` e o
pacote `ai` em modo editável, conforme o guia de treinamento. Na raiz:

```sh
AI_CHECKPOINT=ai/runs/multiple-evidence-v2-2026-11-11/best.pt \
AI_DEVICE=mps bun run dev:server
```

Esse checkpoint é um candidato local, não acompanha o Git e ainda **não passou**
no critério documental. O caminho acima habilita uma experiência explícita;
o checkpoint padrão do chat foi preservado.

Também é possível definir as variáveis em `apps/server/.env`:

- `AI_CHECKPOINT`: obrigatório para habilitar inferência; caminho local absoluto
  ou relativo à raiz. Somente pesos próprios de origem conhecida.
- `AI_PYTHON`: opcional; padrão `ai/.venv/bin/python` na raiz.
- `AI_DEVICE`: `auto`, `cpu` ou `mps`; padrão `auto`.

Sem checkpoint configurado, a chamada retorna `PRECONDITION_FAILED`.
O cliente autenticado pode executar:

```ts
const result = await trpc.ai.reviewText.mutate({
  text: "O projeto inclui testes.",
  requirements: ["O projeto deve incluir testes."],
});
```

Use a sessão/cookie já usada nas outras chamadas protegidas. O endpoint do
servidor é `/trpc/ai.reviewText`; texto e requisitos vão no corpo, nunca como
argumentos de shell. O cliente não escolhe checkpoint, dispositivo ou caminhos.

## Contrato e limites

Até 8.000 caracteres de texto e quatro requisitos de até 600 caracteres.
Uma análise por processo da API; concorrência retorna `TOO_MANY_REQUESTS`.
O subprocesso tem prazo de 120 segundos, limite de saída de 4 MiB e é encerrado
se exceder esses limites. Cada chamada carrega os pesos novamente.

O resultado é validado antes de sair da API. Saída inválida do modelo produz
insuficiência e `model_outputs_valid: false`; erro de processo ou protocolo
produz erro da API. Nenhum resultado parcial é persistido. O caminho privado
do checkpoint e as respostas brutas dos lotes não são expostos ao cliente.

Offsets são pontos de código Unicode dentro da unidade `text:1` do texto
original. Não use diretamente `String.slice` com esses offsets quando houver
emoji; `Array.from(text).slice(start, end).join("")` preserva essa unidade.
Cobertura significa passagem pelo texto, não entendimento ou extração de anexos.

Este recorte recebe texto, sem upload, OCR ou acesso a arquivos indicados pelo
cliente. A tela nativa e a persistência da conferência continuam pendentes.
O serviço depende do Mac/ambiente Python; não constitui um deploy em produção.

## Verificação de 11/11/2026

Inferência real com o candidato V2 no MPS, via serviço TypeScript e via adaptador
HTTP do tRPC em memória: sem sessão `401`, com sessão de teste injetada `200`.
O exemplo acima retornou `atendido`, uma citação válida e `experimental: true`.
O percurso pelo adaptador levou aproximadamente 3,8 s incluindo carregar pesos;
é uma única amostra, sem servidor de rede ou login real nesse teste.

Testes automatizados cobrem autenticação ausente, configuração ausente,
entrada inválida, concorrência, erro de processo, protocolo inválido e liberação
do bloqueio após falha. O [relatório do ciclo](MULTIPLE_EVIDENCE.md) registra os
limites de qualidade do candidato.


## Teste reproduzível com inferência real

Além dos testes comuns, há um teste opcional do adaptador HTTP com subprocesso
Python e pesos reais. Na raiz, com um checkpoint local explícito:

```sh
CONFERE_AI_LIVE_TEST=1 \
AI_CHECKPOINT=ai/runs/evidence-robustness-2026-11-11/best.pt \
AI_DEVICE=mps bun run test apps/api/src/modules/ai/live.test.ts
```

Ele cobre recusa sem sessão, respostas estruturadas, cobertura do texto,
citações com offsets Unicode, sinalização inconclusiva de conflitos detectados
e ausência de caminhos privados na resposta. Usa três textos autorais, incluindo
uma afirmação e uma negação na mesma unidade com emoji.

A sessão é sintética e o adaptador roda em memória, sem login, rede ou banco.
O teste verifica transporte e coerência das flags; não exige que o modelo
classifique corretamente os três textos. A qualidade semântica é medida nas
avaliações documentais. Sem `CONFERE_AI_LIVE_TEST=1`, esse teste fica desativado
e os testes comuns não carregam os pesos.
