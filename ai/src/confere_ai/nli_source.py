"""Contrato do único snapshot público de inferência admitido neste piloto."""
SOURCE = 'https://huggingface.co/datasets/hapaxlegomenon/InferBR'
REVISION = '304b2ca358c4f315a908b69d3d0c607ee101176d'
TRAIN_SHA256 = '49af0412e6e199c8b340dd2c596e7480afb89c60dedd7d982fbb278c23338ce6'
LICENSE_SHA256 = 'b1cd6172bf8061abcbd7917e9246ddeded210339482e802e3413b74cc7e0de38'
STATES = {0: 'nao_atendido', 1: 'atendido', 2: 'evidencia_insuficiente'}


def validate_provenance(row):
    p = row['provenance']
    if (row['split'] != 'train' or not row['task'].startswith('grounded_')
            or p.get('source') != SOURCE or p.get('revision') != REVISION
            or p.get('license') != 'MIT' or p.get('train_sha256') != TRAIN_SHA256
            or p.get('license_sha256') != LICENSE_SHA256 or not p.get('source_row_id')
            or type(p.get('source_label')) is not int or p['source_label'] not in STATES
            or row['task'] != 'grounded_' + STATES[p['source_label']]):
        raise ValueError('Proveniência ou rótulo de inferência pública inválido.')
