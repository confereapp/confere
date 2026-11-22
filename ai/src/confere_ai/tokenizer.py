"""BPE didático sobre bytes UTF-8, treinado somente nos textos de treino."""

from collections import Counter, defaultdict
import heapq
import re

WHITESPACE = frozenset(b" \t\n\r\f\v")
CHUNKS = re.compile(r"[ \t\n\r\f\v]?[^ \t\n\r\f\v]*")


class ByteBPE:
    BOS, EOS, SEP, PAD = 256, 257, 258, 259

    def __init__(self, merges=()):
        self.merges = [tuple(pair) for pair in merges]
        self.vocab = {i: bytes([i]) for i in range(256)}
        for token, (left, right) in enumerate(self.merges, 260):
            self.vocab[token] = self.vocab[left] + self.vocab[right]
        self.ranks = {pair: token for token, pair in enumerate(self.merges, 260)}
        # Se nenhum token tem espaço ASCII fora da primeira posição, nenhuma fusão cruza o início de um
        # trecho " palavra"; codificar trecho a trecho (com cache) dá o mesmo resultado que o texto inteiro.
        self.chunked = all(not WHITESPACE.intersection(self.vocab[t][1:]) for t in range(260, self.vocab_size))
        self.cache = {}

    @property
    def vocab_size(self):
        return 260 + len(self.merges)

    @staticmethod
    def merge(ids, pair, token):
        result, index = [], 0
        while index < len(ids):
            if index + 1 < len(ids) and (ids[index], ids[index + 1]) == pair:
                result.append(token)
                index += 2
            else:
                result.append(ids[index])
                index += 1
        return result

    @classmethod
    def train(cls, texts, vocab_size=512):
        if vocab_size < 260:
            raise ValueError("Vocabulário precisa reservar 256 bytes e quatro tokens especiais.")
        sequences = [list(text.encode("utf-8")) for text in texts]
        merges = []
        while len(merges) + 260 < vocab_size:
            counts = Counter(pair for ids in sequences for pair in zip(ids, ids[1:]))
            if not counts:
                break
            pair = max(counts, key=lambda item: (counts[item], item))
            if counts[pair] < 2:
                break
            token = 260 + len(merges)
            sequences = [cls.merge(ids, pair, token) for ids in sequences]
            merges.append(pair)
        return cls(merges)

    @classmethod
    def train_words(cls, texts, vocab_size=8192):
        """BPE por trechos " palavra" com contagem incremental; escala para milhões de caracteres."""
        if vocab_size < 260:
            raise ValueError("Vocabulário precisa reservar 256 bytes e quatro tokens especiais.")
        frequency = Counter(chunk for text in texts for chunk in CHUNKS.findall(text) if chunk)
        words = [list(chunk.encode("utf-8")) for chunk in frequency]
        counts = list(frequency.values())
        pairs, where = Counter(), defaultdict(set)
        for index, (ids, count) in enumerate(zip(words, counts)):
            for pair in zip(ids, ids[1:]):
                pairs[pair] += count; where[pair].add(index)
        heap = [(-count, pair) for pair, count in pairs.items()]
        heapq.heapify(heap)
        merges = []
        while len(merges) + 260 < vocab_size and heap:
            negative, pair = heapq.heappop(heap)
            if pairs.get(pair, 0) != -negative:
                continue  # entrada desatualizada
            if -negative < 2:
                break
            token = 260 + len(merges); merges.append(pair)
            for index in where.pop(pair, ()):
                ids, count = words[index], counts[index]
                for old in zip(ids, ids[1:]):
                    pairs[old] -= count
                merged = cls.merge(ids, pair, token); words[index] = merged
                for new in zip(merged, merged[1:]):
                    pairs[new] += count; where[new].add(index)
                touched = set(zip(ids, ids[1:])) | set(zip(merged, merged[1:]))
                for changed in touched:
                    if pairs[changed] > 0 and changed != pair:
                        heapq.heappush(heap, (-pairs[changed], changed))
            pairs.pop(pair, None)
        # ponytail: empates resolvidos pela ordem do heap, não por (contagem, par) como em `train`.
        return cls(merges)

    def encode(self, text):
        if self.chunked and self.merges:
            result = []
            for chunk in CHUNKS.findall(text):
                if chunk:
                    ids = self.cache.get(chunk)
                    if ids is None:
                        ids = self.encode_whole(chunk)
                        if len(self.cache) < 1_000_000: self.cache[chunk] = ids
                    result.extend(ids)
            return result
        return self.encode_whole(text)

    def encode_whole(self, text):
        ids = list(text.encode("utf-8"))
        if not ids:
            return []
        previous = list(range(-1, len(ids) - 1))
        following = list(range(1, len(ids))) + [-1]
        alive = [True] * len(ids)
        pending = []

        def schedule(index):
            if index >= 0 and following[index] >= 0:
                rank = self.ranks.get((ids[index], ids[following[index]]))
                if rank is not None:
                    heapq.heappush(pending, (rank, index))

        for index in range(len(ids) - 1):
            schedule(index)
        while pending:
            rank, left = heapq.heappop(pending)
            right = following[left]
            if not alive[left] or right < 0 or self.ranks.get((ids[left], ids[right])) != rank:
                continue
            ids[left] = rank
            alive[right] = False
            following[left] = following[right]
            if following[right] >= 0:
                previous[following[right]] = left
            schedule(previous[left])
            schedule(left)
        return [token for token, active in zip(ids, alive) if active]

    def decode(self, ids):
        return b"".join(self.vocab[token] for token in ids if token not in {self.BOS, self.EOS, self.SEP, self.PAD}).decode("utf-8", errors="replace")
