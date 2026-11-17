# Teste local em documentos reais — 17/11/2026

Executado com a referência documental CI, sem treinamento, ajuste de limiares
ou alteração da API. Os pesos foram conferidos por SHA-256 antes e depois.

| Documento | Páginas | Critérios | Lotes | Status coincidentes | Conflitos |
|---|---:|---:|---:|---:|---:|
| [Gestão documental de RH](maua-2240.md) | 10 | 8 | 128 | 2 | 8 |
| [Práticas ágeis e arquitetura](maua-329.md) | 60 | 10 | 758 | 2 | 10 |

Todos os 18 resultados foram `evidencia_insuficiente`. Os quatro status
coincidentes já tinham esse rótulo no checklist; a coincidência não demonstra
que o modelo reconheceu corretamente o motivo. Houve zero aprovações finais
indevidas, mas foram retornadas evidências sem suporte semântico.

Exemplos observados:

- O requisito sobre percentuais de commits recebeu a folha de rosto como
  evidência positiva, embora ela não apresente esses percentuais.
- O requisito sobre proteção de acesso encontrou a citação esperada sobre 2FA,
  mas classificações opostas em outros trechos impediram a conclusão.
- No segundo PDF, a página física 2, visualmente confirmada em branco,
  permaneceu na entrada. O pipeline conservador bloqueou conclusões nas dez
  consultas. Também houve conflitos nesses casos; remover a página não
  provaria acerto semântico.

Auditoria mecânica: todo o texto extraído foi coberto por intervalos contíguos,
sem cortes ocultos, e todas as citações retornadas correspondem literalmente
às respectivas unidades. Isso confere rastreabilidade, não interpretação.
As citações editoriais esperadas foram localizadas após normalização de
espaços; nenhuma normalização foi aplicada à entrada da inferência.

Os arquivos já eram conhecidos e os critérios são experimentais do Confere,
não rubricas oficiais nem notas dos TCCs. As anotações ainda não têm revisão
humana independente. OCR pode conter erros; imagens e estrutura de tabelas
não são interpretadas. Não houve teste HTTP, publicação dos PDFs ou promoção.

Resumo rastreável: `output/ai/real-documents-cq-2026-11-17.json`.
Traços integrais permanecem privados em `ai/data/private/real-documents-cq-2026-11-17/`.
Verificação: dez testes focados aprovados, incluindo rejeição de citações
alteradas e falhas de cobertura.

```sh
rtk proxy env PYTHONPATH=ai/src ai/.venv/bin/python ai/scripts/evaluate_real_documents.py \
  --checkpoint ai/runs/evidence-calibrated-ci-2026-11-17/best.pt \
  --candidates ai/data/private/maua-2240/examples.candidate.jsonl \
    ai/data/private/maua-329/examples.candidate.jsonl \
  --output ai/data/private/real-documents-nova-execucao --device mps
```

A saída precisa ser nova para preservar a execução anterior.
