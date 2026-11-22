import { describe, expect, test } from "vitest";
import { splitRubric } from "./rubric";

describe("splitRubric", () => {
  test("lista numerada com ponto, parêntese e hífen", () => {
    expect(
      splitRubric(
        "1. Introdução com objetivo\n2) Metodologia adotada\n3 - Resultados obtidos",
      ),
    ).toEqual([
      "Introdução com objetivo",
      "Metodologia adotada",
      "Resultados obtidos",
    ]);
  });

  test("numeração hierárquica", () => {
    expect(
      splitRubric("1.1 Objetivo geral\n1.2. Objetivos específicos"),
    ).toEqual(["Objetivo geral", "Objetivos específicos"]);
  });

  test("letras", () => {
    expect(splitRubric("a) Resumo\nb. Abstract\nC) Conclusão")).toEqual([
      "Resumo",
      "Abstract",
      "Conclusão",
    ]);
  });

  test("marcadores •, -, *, –", () => {
    expect(
      splitRubric("• Capa\n- Sumário\n* Referências ABNT\n– Apêndices"),
    ).toEqual(["Capa", "Sumário", "Referências ABNT", "Apêndices"]);
  });

  test("caixas de seleção ( ) e [ ]", () => {
    expect(
      splitRubric(
        "( ) Diagrama de arquitetura\n[x] Testes automatizados\n[ ] Manual",
      ),
    ).toEqual(["Diagrama de arquitetura", "Testes automatizados", "Manual"]);
  });

  test("junta linhas quebradas e descarta cabeçalhos", () => {
    const text = [
      "CRITÉRIOS DE AVALIAÇÃO",
      "O trabalho será avaliado conforme os itens abaixo:",
      "",
      "1. Descrever a arquitetura da solução,",
      "   incluindo diagramas de componentes;",
      "2. Apresentar testes com usuários reais; e",
      "",
      "Entrega final:",
      "3. Discutir limitações.",
    ].join("\n");
    expect(splitRubric(text)).toEqual([
      "Descrever a arquitetura da solução, incluindo diagramas de componentes",
      "Apresentar testes com usuários reais",
      "Discutir limitações.",
    ]);
  });

  test("remove duplicatas e itens vazios", () => {
    expect(
      splitRubric("- Resumo\n- \n- resumo\n-   Resumo.\n- Abstract"),
    ).toEqual(["Resumo", "Abstract"]);
  });

  test("sem marcadores: usa linhas (checklist-v1)", () => {
    const text = [
      "O trabalho deve declarar o objetivo geral do projeto.",
      "O trabalho deve descrever a arquitetura da solução.",
      "O trabalho deve propor trabalhos futuros.",
    ].join("\n");
    expect(splitRubric(text)).toEqual(text.split("\n"));
  });

  test("sem marcadores: junta linha quebrada no meio da frase", () => {
    expect(
      splitRubric(
        "Apresentar resultados quantitativos como\nmétricas e medições.\nTratar de acessibilidade no app.",
      ),
    ).toEqual([
      "Apresentar resultados quantitativos como métricas e medições.",
      "Tratar de acessibilidade no app.",
    ]);
  });

  test("sem marcadores: parágrafo único vira frases", () => {
    expect(
      splitRubric(
        "O aluno deve entregar o código-fonte. Deve haver documentação da API; Testes são obrigatórios.",
      ),
    ).toEqual([
      "O aluno deve entregar o código-fonte.",
      "Deve haver documentação da API",
      "Testes são obrigatórios.",
    ]);
  });

  test("texto vazio ou curto demais", () => {
    expect(splitRubric("")).toEqual([]);
    expect(splitRubric("Rubrica\n\nOk")).toEqual([]);
  });
});
