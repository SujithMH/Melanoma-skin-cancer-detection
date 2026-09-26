import numpy as np

def decode_vector(vector):
    """
    Decodes a 7D vector in [0, 1]^7 to a hyperparameter dictionary.
    """
    v = np.clip(vector, 0.0, 1.0)
    
    # 0: head_lr (log scale: 1e-5 to 1e-3)
    head_lr = 10 ** (np.log10(1e-5) + v[0] * (np.log10(1e-3) - np.log10(1e-5)))
    
    # 1: backbone_lr_mult (linear scale: 0.05 to 1.0)
    backbone_lr_mult = 0.05 + v[1] * (1.0 - 0.05)
    
    # 2: weight_decay (log scale: 1e-5 to 1e-2)
    weight_decay = 10 ** (np.log10(1e-5) + v[2] * (np.log10(1e-2) - np.log10(1e-5)))
    
    # 3: dropout (linear scale: 0.1 to 0.6)
    dropout = 0.1 + v[3] * (0.6 - 0.1)
    
    # 4: fc_hidden (step of 64: 128 to 1024)
    raw_fc = 128 + v[4] * (1024 - 128)
    fc_hidden = int(round(raw_fc / 64) * 64)
    fc_hidden = int(np.clip(fc_hidden, 128, 1024))
    
    # 5: fusion_type (categorical: concat vs add)
    fusion_options = ['concat', 'add']
    fusion_idx = min(int(np.floor(v[5] * len(fusion_options))), len(fusion_options) - 1)
    fusion_type = fusion_options[fusion_idx]
    
    # 6: sev_weight (linear scale: 0.1 to 1.0)
    sev_weight = 0.1 + v[6] * (1.0 - 0.1)
    
    return {
        'head_lr': float(head_lr),
        'backbone_lr_mult': float(backbone_lr_mult),
        'weight_decay': float(weight_decay),
        'dropout': float(dropout),
        'fc_hidden': fc_hidden,
        'fusion_type': fusion_type,
        'sev_weight': float(sev_weight)
    }