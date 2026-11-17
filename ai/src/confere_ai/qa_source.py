"""Proveniência e alinhamento do piloto público de leitura extrativa."""
SOURCE = 'https://huggingface.co/datasets/eraldoluis/faquad'
REVISION = '6ad978f20672bb41625b3b71fbe4a88b893d0a86'
TRAIN_SHA256 = '33d32022638a4ad252420d44544585279903a83637a974c6fe6c2e53f1c7b294'


def validate_provenance(row):
    p = row['provenance']
    if (p.get('source') != SOURCE or p.get('revision') != REVISION
            or p.get('train_sha256') != TRAIN_SHA256 or p.get('license') != 'CC-BY-4.0'):
        raise ValueError('Fonte extrativa não corresponde ao snapshot auditado.')
    context, start, answer = p.get('context'), p.get('answer_start'), row['target']
    if (not isinstance(context, str) or type(start) is not int or start < 0
            or context[start:start+len(answer)] != answer or not answer.strip()
            or row['prompt'] != f"Contexto: {context}\nPergunta: {p.get('question')}\nResponda usando apenas o texto."):
        raise ValueError('Resposta extrativa desalinhada ou contexto alterado.')
