"""Separação conservadora por documento e por frases normalizadas semelhantes."""
from collections import defaultdict
import hashlib
import re
import unicodedata

PROTOCOL = 'document_sentence_components_v1'


def words(text):
    text = unicodedata.normalize('NFKD', text.casefold())
    return re.findall(r'[a-z0-9]+', ''.join(c for c in text if not unicodedata.combining(c)))


def components(records):
    """Une documentos e qualquer frase dos dois lados com Jaccard de tokens >= .8."""
    parent = list(range(len(records)))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    def union(a, b):
        a, b = root(a), root(b)
        parent[max(a, b)] = min(a, b)
    documents = {}; exact = {}; inverted = defaultdict(set); texts = []
    for i, row in enumerate(records):
        document = row.get('source_document')
        if document:
            union(i, documents.setdefault(document, i))
        for text in [row['requirement']] + [u['text'] for u in row['units']]:
            tokens = frozenset(words(text))
            if not tokens: raise ValueError('Texto vazio após normalização.')
            if tokens in exact:
                union(i, exact[tokens]); continue
            candidates = set().union(*(inverted[t] for t in tokens))
            for j in candidates:
                other, owner = texts[j]
                if min(len(tokens), len(other)) / max(len(tokens), len(other)) < .8: continue
                if len(tokens & other) / len(tokens | other) >= .8: union(i, owner)
            j = len(texts); texts.append((tokens, i)); exact[tokens] = i
            for token in tokens: inverted[token].add(j)
    groups = defaultdict(list)
    for i in range(len(records)): groups[root(i)].append(i)
    result = {}
    for indices in groups.values():
        key = hashlib.sha256('\n'.join(sorted(records[i]['id'] for i in indices)).encode()).hexdigest()
        for i in indices: result[records[i]['id']] = key
    return result


def split_for(group):
    bucket = int(hashlib.sha256(('confere-relations-v1:' + group).encode()).hexdigest()[:8], 16) % 10
    return 'test' if bucket == 0 else 'validation' if bucket == 1 else 'train'


def verify_partition(rows):
    selected = [r for r in rows if r['provenance'].get('partition_protocol') == PROTOCOL]
    for group in _group_rows(selected).values():
        if len({r['split'] for r in group}) != 1:
            raise ValueError('Documento ou frase semelhante atravessa splits.')


def _group_rows(rows):
    groups = defaultdict(list)
    keys = components(rows)
    for row in rows: groups[keys[row['id']]].append(row)
    return groups
