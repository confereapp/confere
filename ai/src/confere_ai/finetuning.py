"""Limita plasticidade sem alterar arquitetura ou formato dos checkpoints."""


def configure_trainable(model, last_blocks=None):
    if last_blocks is not None and (not isinstance(last_blocks, int) or not 1 <= last_blocks <= len(model.blocks)):
        raise ValueError("Número de blocos treináveis fora da arquitetura.")
    for parameter in model.parameters(): parameter.requires_grad_(last_blocks is None)
    if last_blocks is not None:
        for module in [*list(model.blocks)[-last_blocks:], model.norm, model.head]:
            for parameter in module.parameters(): parameter.requires_grad_(True)
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
