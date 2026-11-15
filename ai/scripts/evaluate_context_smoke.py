"""Diagnósticos autorais sem o preâmbulo de treino; saída literal não mede diálogo geral."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from confere_ai.chat import prepare_prompt
from confere_ai.model import GPT, GPTConfig
from confere_ai.tokenizer import ByteBPE

CASES = [
    dict(message="No projeto Horizonte, a responsável é Lia. Qual é o nome da responsável?", target="Lia."),
    dict(message="A entrega está prevista para daqui a 18 dias. Qual é o prazo informado?", target="18 dias."),
    dict(message="O projeto usa Python e terá testes automatizados. Qual linguagem ele usa?", target="Python."),
    dict(message="A responsável pelo projeto é Lia. Não há prazo descrito. Qual é o prazo de entrega?", target="Não tenho informação suficiente para responder."),
    dict(message="O relatório diz que há testes e depois diz que não há testes. Dá para confirmar se os testes existem?", target="Há informações conflitantes no texto."),
    dict(message="Qual nome devo usar agora?", history=[["Me chamo Lia.", "Prazer, Lia."], ["Corrigindo: meu nome é Rui.", "Prazer, Rui."]], target="Seu nome é Rui."),
    dict(message="Como você se chama?", history=[["Me chamo Lia.", "Prazer, Lia."], ["Corrigindo: meu nome é Rui.", "Prazer, Rui."]], target="Sou o Confere."),
    dict(message="O projeto não inclui testes, mas inclui documentação. A documentação foi incluída?", target="Sim."),
    dict(message="Equipe: Lia faz os testes; Rui escreve o manual. Quem faz os testes?", target="Lia."),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): parser.error("Saída deve ser nova.")
    torch.set_num_threads(4)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    tokenizer = ByteBPE(checkpoint["merges"])
    model = GPT(GPTConfig(**checkpoint["config"])).to("mps")
    model.load_state_dict(checkpoint["model"]); model.eval()
    outputs = []
    for case in CASES:
        ids, dropped = prepare_prompt(case.get("history", []), case["message"], tokenizer, model.config.block_size, 128, "conversation")
        assert dropped == 0
        generated = model.generate(torch.tensor([ids], device="mps"), 128, tokenizer.EOS)
        response = tokenizer.decode(generated[0, len(ids):].tolist())
        outputs.append(dict(**case, actual=response, literal_match=response.strip() == case["target"]))
    result = dict(checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(), step=checkpoint["step"],
                  literal_matches=sum(r["literal_match"] for r in outputs), total=len(outputs), samples=outputs,
                  adopted=False, note="Diagnósticos de desenvolvimento, formulados antes da conclusão do ciclo B; fora do corpus de treino. Igualdade literal penaliza paráfrases corretas; leitura semântica necessária. Não são teste cego de conversa geral.")
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__": main()
