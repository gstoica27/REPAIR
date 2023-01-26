import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn

match_tensors = None

def set_match_fn(fn):
    global match_tensors
    match_tensors = fn

def concat_mats(args, dim=0):
    return torch.cat(args, dim=dim)

def unconcat_mat(tensor, dim=0):
    return torch.chunk(tensor, chunks=2, dim=dim)

def check_convergence(old_transforms, current_transforms, eps=1e-5):
    if len(old_transforms) == 0: 
        return False, {}
    transform_norms = {}
    is_converged = True
    for key, old_transform in old_transforms.items():
        current_transform = current_transforms[key]
        is_close = torch.allclose(
            current_transform.output_align, 
            old_transform.output_align, 
            atol=eps
        )
        norm = torch.norm(current_transform.output_align - old_transform.output_align)
        if not is_close: 
            is_converged = False
        transform_norms[key] = torch.round(norm, decimals=3).cpu().numpy().round(3)
    return (is_converged, transform_norms)


# use the train loader with data augmentation as this gives better results
def reset_bn_stats(model, epochs=1, loader=None, DEVICE='cuda:0'):
    # resetting stats to baseline first as below is necessary for stability
    for m in model.modules():
        if type(m) == nn.BatchNorm2d:
            m.momentum = None # use simple average
            m.reset_running_stats()
    # run a single train epoch with augmentations to recalc stats
    model = model.train().to(DEVICE)
    for _ in range(epochs):
        with torch.no_grad(), torch.autocast("cuda"):
            for images, _ in loader:
                output = model(images.to(DEVICE).half())


def mix_weights(model, alpha, sd0, sd1):
    sd_alpha = {}
    for k in sd0.keys():
        param0 = sd0[k].to(DEVICE)
        param1 = sd1[k].to(DEVICE)
        sd_alpha[k] = (1 - alpha) * param0 + alpha * param1
    model.load_state_dict(sd_alpha)
    return

                
class LayerTransform(dict):
    def __init__(self, normalize_tensors=False, tensor_merge_type='concat'):
        super().__init__()
        self.output_align = None
        self.next_input_align = None
        self.normalize_tensors = normalize_tensors
        self.tensor_merge_type = tensor_merge_type
    
    def compute_transform(self):
        inputs = list(self.values())
        if self.normalize_tensors:
            for idx, inp in enumerate(inputs):
                inputs[idx] = F.normalize(inp, dim=-1)
        if self.tensor_merge_type == 'concat':
            match_input = concat_mats(inputs, dim=-1)
        elif self.tensor_merge_type == 'mean':
            match_input = torch.stack(inputs, dim=0).mean(0)
                
        self.output_align, self.next_input_align = match_tensors(match_input)
       

