// Extração determinística de itens de uma rubrica em texto. Sem IA: só regras
// de formatação de lista. O resultado é uma proposta que o usuário revisa.

const MARKER =
  /^\s*(?:\d{1,3}(?:\.\d{1,3})*(?:[.)]|\s*[-–:])\s+|\d{1,3}(?:\.\d{1,3})+\s+|[a-zA-Z][).]\s+|[•\-*–—▪●◦·](?:\s+|$)|\(\s*[xX]?\s*\)\s*|\[\s*[xX]?\s*\]\s*)/;

const SENTENCE_END = /[.;:!?]$/;

function clean(item: string): string {
  return item
    .replace(/\s+/g, " ")
    .replace(/[;,]\s*(?:e|ou)?$/i, "")
    .trim();
}

function isHeader(line: string): boolean {
  const words = line.split(/\s+/).length;
  return (
    line.endsWith(":") ||
    (words <= 6 && line === line.toUpperCase() && /[A-ZÀ-Ú]/.test(line))
  );
}

function fromMarkers(lines: string[]): string[] {
  const items: string[] = [];
  let current: string | null = null;
  const flush = () => {
    if (current !== null) items.push(current);
    current = null;
  };

  for (const line of lines) {
    const marker = line.match(MARKER);
    if (marker) {
      flush();
      current = line.slice(marker[0].length);
    } else if (!line) {
      flush();
    } else if (current !== null && !isHeader(line)) {
      current += ` ${line}`;
    } else {
      // Linha solta fora de um item: cabeçalho ou texto introdutório.
      flush();
    }
  }
  flush();
  return items;
}

function fromPlainText(lines: string[]): string[] {
  const chunks: string[] = [];
  let current = "";
  for (const line of lines) {
    if (!line) {
      if (current) chunks.push(current);
      current = "";
    } else if (
      current &&
      !SENTENCE_END.test(current) &&
      /^[a-zà-ú]/.test(line)
    ) {
      // Linha quebrada no meio da frase.
      current += ` ${line}`;
    } else {
      if (current) chunks.push(current);
      current = line;
    }
  }
  if (current) chunks.push(current);

  const items =
    chunks.length === 1
      ? (chunks[0] ?? "").split(/(?<=[.;!?])\s+(?=[A-ZÀ-Ú])/)
      : chunks;
  // Sem marcadores, exige um mínimo de texto para não virar requisito qualquer.
  return items.filter(
    (item) => item.split(/\s+/).length >= 3 && !isHeader(item.trim()),
  );
}

export function splitRubric(text: string): string[] {
  const lines = text.split(/\r?\n/).map((line) => line.trim());
  const hasMarkers = lines.some((line) => MARKER.test(line));
  const raw = hasMarkers ? fromMarkers(lines) : fromPlainText(lines);

  const seen = new Set<string>();
  const result: string[] = [];
  for (const item of raw) {
    const cleaned = clean(item);
    const key = cleaned.toLowerCase().replace(/[.!?]$/, "");
    if (cleaned.length < 3 || seen.has(key)) continue;
    seen.add(key);
    result.push(cleaned);
  }
  return result;
}
