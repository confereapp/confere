"""Contrato de critérios editoriais locais sobre parágrafos públicos rastreáveis."""
import hashlib
from .qa_source import SOURCE, REVISION, TRAIN_SHA256
from .relation_partition import PROTOCOL


def validate_provenance(row):
    p = row['provenance']
    if (p.get('source') != SOURCE or p.get('revision') != REVISION
            or p.get('train_sha256') != TRAIN_SHA256 or p.get('license') != 'CC-BY-4.0'
            or p.get('partition_protocol') != PROTOCOL or p.get('annotation_kind') != 'local_editorial'
            or not p.get('rationale') or not row.get('source_document')
            or not row['task'].startswith('grounded_') or len(row['units']) != 1
            or hashlib.sha256(row['units'][0]['text'].encode()).hexdigest() != p.get('paragraph_sha256')):
        raise ValueError('Parágrafo ou anotação pública sem proveniência válida.')
    start, quote, text = p.get('anchor_start'), p.get('anchor'), row['units'][0]['text']
    if type(start) is not int or start < 0 or not isinstance(quote, str) or not quote.strip() or text[start:start+len(quote)] != quote:
        raise ValueError('Âncora editorial desalinhada.')
