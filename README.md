# PatchCLAM
Rewritten CLAM (https://github.com/mahmoodlab/CLAM) for path-based datasets. 

## Strength
- Easy to understand and modify.
- Simple framework for [CLAM](https://github.com/mahmoodlab/CLAM), [ABMIL](https://github.com/AMLab-Amsterdam/AttentionDeepMIL), scaled dot-product attention, and average pooling.
- Square attention matrix implementation inspired by [TransMIL](https://proceedings.neurips.cc/paper/2021/file/10c272d06794d3e5785d5e7c5356e9ff-Paper.pdf).

## Prerequisites
- torch>=1.8.0
- torchvision (for MNIST demo)
- tqdm (for MNIST demo)

## Release Notes
-2024.11.06 Major bug fixed. Softmax to attention matrix is applied to diagonal elements for Attn_Net and Attn_Net_Gated. Incorrect AvgPool is fixed. 

## Benchmarks
Underway...

## How to Use
Load model and define criterion.
```
import mil_configs
from models import ClamWrapper
from utils import resnet50_baseline

model = ClamWrapper(mil_configs.clam_config, base_encoder=resnet50_baseline(pretrained=True))
criterion = nn.CrossEntropyLoss(reduction='mean')
inst_criterion = nn.BCEWithLogitsLoss(reduction='none')
```

Incert CLAM training snippet below. The ratio of bag_loss and inst_loss (default 7:3) can be a hyperparameter.
```
bag_logit, inst_logit, top_p_ids, top_n_ids  = model(images)
bag_loss = criterion(bag_logit, labels)
instance_target = torch.zeros_like(inst_logit).to(inst_logit.device)
instance_mask   = torch.zeros_like(inst_logit).to(inst_logit.device)
for p_index, n_index in zip(top_p_ids, top_n_ids):
    if p_index.dim() > 1: # CLAM-MB
        instance_target[p_index[labels.item()], labels] = 1.
        if mil_configs.clam_config['subtyping']:
            instance_mask[p_index[labels.item()], :] = 1. 
        else:
            instance_mask[p_index[labels.item(), :], labels] = 1. 
        instance_mask[n_index[labels.item()], labels] = 1.
    else: # CLAM-SB
        instance_target[p_index, labels] = 1.
        if mil_configs.clam_config['subtyping']:
            instance_mask[p_index] = 1.         
        else:
            instance_mask[p_index, labels] = 1. 
        
        instance_mask[n_index, labels] = 1.
inst_loss = inst_criterion(inst_logit.view(-1), instance_target.view(-1)) * instance_mask.view(-1)
inst_loss = inst_loss.mean()

loss = 0.7*bag_loss + 0.3*inst_loss
```

## MNIST example
MNIST_MIL.ipynb is provided as a small use code.
