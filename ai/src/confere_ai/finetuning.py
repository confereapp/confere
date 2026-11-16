"""Limita plasticidade sem alterar arquitetura ou formato dos checkpoints."""
import torch


def _frozen_input_gradient(module, inputs):
    value = inputs[0]
    if torch.is_grad_enabled() and value.device.type == "mps" and not value.requires_grad:
        # MPS LayerNorm pode gerar gradiente inválido com entrada congelada; esta folha não atualiza o encoder.
        return (value.detach().requires_grad_(True),)


def configure_trainable(model, last_blocks=None):
    if last_blocks is not None and (not isinstance(last_blocks, int) or not 1 <= last_blocks <= len(model.blocks)):
        raise ValueError("Número de blocos treináveis fora da arquitetura.")
    previous = getattr(model, "_frozen_input_hook", None)
    if previous is not None:
        previous.remove()
        del model._frozen_input_hook
    for parameter in model.parameters(): parameter.requires_grad_(last_blocks is None)
    if last_blocks is not None:
        for module in [*list(model.blocks)[-last_blocks:], model.norm, model.head]:
            for parameter in module.parameters(): parameter.requires_grad_(True)
        first = model.blocks[len(model.blocks)-last_blocks]
        model._frozen_input_hook = first.norm1.register_forward_pre_hook(_frozen_input_gradient)
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
