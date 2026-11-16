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

- `AI_CHECKPOINT`: pesos para revisão documental; também servem de fallback para
  conversa. Caminho local absoluto ou relativo à raiz, de origem conhecida.
- `AI_CHAT_CHECKPOINT`: opcional; pesos exclusivos de conversa. Permite configurar
  o chat sem habilitar a revisão documental. Se ausente, usa `AI_CHECKPOINT`.
- `AI_PYTHON`: opcional; padrão `ai/.venv/bin/python` na raiz.
- `AI_DEVICE`: `auto`, `cpu` ou `mps`; padrão `auto`.

Sem checkpoint configurado para a tarefa, a chamada retorna `PRECONDITION_FAILED`.

A revisão aceita o decoder documental anterior e checkpoints com objetivo
`document_classifier`. Nesse último caso, o encoder aprende relevância e estados
por unidade; JSON e referências são construídos a partir dessas predições. Os
metadados retornam `inference_kind: unit_classifier` e `fragment_token_reserve`
(reserva do loteador), em vez de alegar tokens gerados pelo decoder. Essa
compatibilidade não aprova os pesos: os critérios de qualidade continuam exigidos.
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

## Conversa livre

`POST /trpc/ai.chat` usa a mesma sessão autenticada e configuração local de pesos
que `ai.reviewText`. Retorna texto gerado, sem exigir requisitos documentais.
Na raiz, para usar o último checkpoint do treinamento:

```sh
rtk proxy env AI_CHECKPOINT=ai/runs/api-batches-2026-11-11/last.pt AI_DEVICE=mps bun run dev:server
```

Com o cliente tRPC autenticado:

```ts
const first = await trpc.ai.chat.mutate({ message: "Olá, quem é você?" });
const next = await trpc.ai.chat.mutate({
  message: "Como você pode ajudar?",
  history: [{ message: "Olá, quem é você?", response: first.response }],
});
```

HTTP direto, usando o cookie de sessão de uma conta autenticada:

```sh
curl -X POST http://localhost:3000/trpc/ai.chat \
  -H 'Content-Type: application/json' \
  -b cookies.txt \
  --data '{"message":"Olá, quem é você?"}'
```

O arquivo `cookies.txt` deve conter a sessão obtida no login. A resposta tRPC
tem o formato `{"result":{"data":{"response":"...","experimental":true,"dropped_turns":0}}}`.
GET não executa geração; a rota é uma mutation POST.

A mensagem aceita até 2.000 caracteres e o histórico opcional até oito pares
`message`/`response`, cada texto com até 2.000 caracteres. O histórico é enviado
pelo cliente e não é salvo no servidor. Os turnos antigos que excedem o contexto
são descartados por inteiro, com a quantidade em `dropped_turns`. Uma mensagem
atual que não cabe no contexto retorna `BAD_REQUEST`, mesmo estando dentro do
limite de caracteres: o limite real depende da tokenização. A geração é limitada
a 128 tokens de saída e pode produzir texto vazio ou interrompido nesse limite.

Ambas as rotas compartilham uma única inferência por processo, timeout de
120 segundos e limite de saída de 4 MiB. Cada chamada recarrega os pesos. O
cliente não escolhe arquivos, dispositivo ou checkpoint. Sem `AI_CHECKPOINT`,
a API retorna `PRECONDITION_FAILED`.

Esse endpoint usa o formato de prompt da conversa no terminal. Ele não amplia a
capacidade dos pesos: o candidato recente foi treinado sobretudo para análise
documental e sua conversa geral não está validada. Nenhuma resposta é persistida.

Verificação de 13/11/2026: 139 testes Python, tipos TypeScript e oito testes
TypeScript com inferência real passaram. O teste HTTP em memória recusou sessão
ausente com 401 e GET com 405, gerou resposta com o último checkpoint
`api-batches-2026-11-11/last.pt` no MPS e aceitou uma segunda mensagem com
histórico. A sessão desse teste é sintética; não verifica login real, servidor
em rede ou qualidade semântica das respostas.
