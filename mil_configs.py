# standard CLAM config
clam_config ={
    "clam_type": "SB", # "SB" or "MB"
    "atten_type": "3fc", # "2fc" is non-gated attention and "3fc" is gated attention
    "apply_max" : False,
    "drop_out" : 0.25,
    "k_sample" : 8,
    "n_classes": 3,
    "subtyping": False,
}

# standard ABMIL set "enable_clam": False in train_config
abmil_config ={
    "clam_type": "SB",
    "atten_type": "3fc",
    "apply_max" : False,
    "drop_out" : 0.0,
    "k_sample" : 8,
    "n_classes": 3,
    "subtyping": False,
}

# Scaled dot-product attention
sdp_config ={
    "clam_type": "SB",
    "atten_type": "sdp",
    "apply_max" : False,
    "drop_out" : 0.0,
    "k_sample" : 8,
    "n_classes": 3,
    "subtyping": False,
}

# standard average pooling MIL  set "enable_clam": False in train_config
avg_config ={
    "clam_type": "SB",
    "atten_type": "avg",
    "apply_max" : False,
    "drop_out" : 0.0,
    "k_sample" : 8,
    "n_classes": 3,
    "subtyping": False,
}