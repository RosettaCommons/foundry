def add_weight_decay(model, l2_coeff):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        #if len(param.shape) == 1 or name.endswith(".bias"):
        #if "norm" in name or name.endswith(".bias"):
        if "norm" in name: #fd let biases decay
            no_decay.append(param)
        else:
            decay.append(param)
    return [{'params': no_decay, 'weight_decay': 0.0}, {'params': decay, 'weight_decay': l2_coeff}]
