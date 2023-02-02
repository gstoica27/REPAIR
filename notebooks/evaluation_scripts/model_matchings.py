import torch
import numpy as np
import scipy

from utils import *


def unflatten(x, k=3):
    O, IHW = x.shape
    return x.view(O, -1, k, k)

def merge_first_convs(state_dict, prefix, a_conv, b_conv, output_transform):
    flatten_conv = lambda x: x.flatten(1)
    a_w = flatten_conv(a_conv.weight)
    b_w = flatten_conv(b_conv.weight)
    ab_w = concat_mats((a_w, b_w), dim=0)
    output_transform[prefix] = ab_w
    output_transform.compute_transform()
    # merge_mat, unmerge_mat = match_tensors(ab_w)
    c_w = output_transform.output_align @ ab_w
    state_dict[prefix + '.weight'] = unflatten(c_w, a_conv.weight.shape[-1])
    return output_transform

def merge_bn(state_dict, prefix, a_bn, b_bn, output_transform):
    staterify = lambda bn: torch.stack((bn.weight, bn.bias, bn.running_mean), dim=1)
    unstaterify = lambda stats: stats.unbind(-1)
    
    a_stats = staterify(a_bn)
    b_stats = staterify(b_bn)
    ab_stats = concat_mats((a_stats, b_stats), dim=0)
    c_stats = output_transform.output_align @ ab_stats
    c_weight, c_bias, c_mean = unstaterify(c_stats)
    ab_var = concat_mats((a_bn.running_var[..., None], b_bn.running_var[...,None]))
    var_out_transform = output_transform.output_align.square()
    c_var = (var_out_transform @ ab_var).reshape(-1)
    state_dict[prefix + '.weight'] = c_weight
    state_dict[prefix + '.bias'] = c_bias
    state_dict[prefix + '.running_mean'] = c_mean
    state_dict[prefix + '.running_var'] = c_var
    pass

def block_diagonalize_tensors(tensor1, tensor2):
    zerooos = torch.zeros_like(tensor1)
    block_diagonal = concat_mats(
        (
            concat_mats((tensor1, zerooos), dim=1),
            concat_mats((zerooos, tensor2), dim=1),
        ),
        dim=0
    )
    return block_diagonal

def merge_hidden_conv(
    state_dict, prefix, a_conv, b_conv, 
    input_transform, output_transform, 
    recompute_output=False,
    only_input_align=False,
    ignore_mismatches=False
):
    O, I, H, W = a_conv.weight.shape
    get_I_by_O_by_HW = lambda x: x.permute(1, 0, 2, 3).flatten(2)
    
    a_I_by_O_by_HW = get_I_by_O_by_HW(a_conv.weight)
    b_I_by_O_by_HW = get_I_by_O_by_HW(b_conv.weight)
    
    dummy_zerooooo = torch.zeros_like(b_I_by_O_by_HW)
    ab_block_diago = concat_mats(
        (
            concat_mats((a_I_by_O_by_HW, dummy_zerooooo), dim=1),
            concat_mats((dummy_zerooooo, b_I_by_O_by_HW), dim=1)
        ),
        dim=0
    )
    
    # [I,2I]x[2I,2OHW]->[I,2OHW]
    ab_input_aligned = input_transform.next_input_align.T @ ab_block_diago.flatten(1)
    ab_input_aligned = ab_input_aligned.\
    reshape(ab_input_aligned.shape[0], -1, H*W).\
    transpose(1, 0).\
    flatten(1) # [I,2O,HW]->[2O,I,HW]->[2O,IHW]
    if only_input_align:
        return output_transform, ab_input_aligned
    
    output_transform[prefix] = ab_input_aligned
    if recompute_output:
        output_transform.compute_transform()
    
    c_flat = output_transform.output_align @ ab_input_aligned
    if ignore_mismatches:
        a_input_aligned, b_input_aligned = ab_input_aligned.chunk(2, dim=0)
        a_dims = a_input_aligned.any(dim=0, keepdim=True)
        b_dims = b_input_aligned.any(dim=0, keepdim=True)
        ab_dims = a_dims * b_dims
        ab_input_aligned_valid = ab_input_aligned * ab_dims
        
#         a_only_valid = ab_input_aligned * (~ab_dims)
#         remove_b = torch.ones((a_only_valid.shape[0], 1), device=a_dims.device) *2.
#         remove_b[remove_b.shape[0]//2:] = 0.
#         a_only_valid *= remove_b
#         ab_input_aligned_valid += a_only_valid
        
        c_a, c_b = output_transform.output_align.chunk(2, dim=-1)
        c_ab = ((c_a.sum(-1) * c_b.sum(-1)) != 0.)
        c_flat[c_ab] = (output_transform.output_align @ ab_input_aligned_valid)[c_ab]
#         print(f"{prefix} joint AB: {c_ab.sum()}/{c_ab.shape[0]}")
#         if prefix == 'linear':
#             import pdb; pdb.set_trace()
        
    state_dict[prefix + '.weight'] = unflatten(c_flat, a_conv.weight.shape[-1])
    
    output_block_diagonal_ab = block_diagonalize_tensors(
        a_conv.weight.flatten(2),
        b_conv.weight.flatten(2)
    )
    ab_output_aligned = output_transform.output_align @ output_block_diagonal_ab.flatten(1)
    ab_output_aligned = ab_output_aligned.reshape(ab_output_aligned.shape[0], -1, H*W).transpose(1, 0).flatten(1)
    input_transform[prefix] = ab_output_aligned
    
    return output_transform

def merge_linear(
    state_dict, prefix, a_linear, 
    b_linear, input_transform, 
    output_transform, 
    recompute_output=False,
    only_input_align=False,
    ignore_mismatches=False
):
    class conv_wrapper:
        def __init__(self, linear):
            self.weight = linear.weight[:, :, None, None]
    
    if only_input_align:
        output_transform, c_w = merge_hidden_conv(
            state_dict, prefix, conv_wrapper(a_linear), 
            conv_wrapper(b_linear), input_transform, 
            output_transform, recompute_output=recompute_output,
            only_input_align=only_input_align,
            ignore_mismatches=ignore_mismatches
        )
        state_dict[prefix + '.weight'] = c_w.flatten(1)
        state_dict[prefix + '.bias'] = concat_mats((a_linear.bias, b_linear.bias))
        
    else:
        output_transform = merge_hidden_conv(
            state_dict, prefix, conv_wrapper(a_linear), 
            conv_wrapper(b_linear), input_transform, 
            output_transform, recompute_output=recompute_output,
            only_input_align=only_input_align,
            ignore_mismatches=ignore_mismatches
        )
        state_dict[prefix + '.weight'] = state_dict[prefix + '.weight'][..., 0, 0]
        state_dict[prefix + '.bias'] = output_transform.output_align @ concat_mats((a_linear.bias, b_linear.bias))
    return output_transform
    
def merge_block(
    state_dict, prefix, a_block, b_block, 
    input_transform, intra_transform,
    output_transform=None, shortcut=False,
    ignore_mismatches=False
):
    conv1_transform = merge_hidden_conv(
        state_dict, prefix + '.conv1', a_block.conv1, b_block.conv1, 
        input_transform, intra_transform, recompute_output=True,
        ignore_mismatches=ignore_mismatches
    )
    merge_bn(state_dict, prefix + '.bn1', a_block.bn1, b_block.bn1, conv1_transform)
    
    
    conv2_transform = merge_hidden_conv(
        state_dict, 
        prefix + '.conv2', 
        a_block.conv2, 
        b_block.conv2, 
        conv1_transform,
        output_transform,
        recompute_output=shortcut,
        ignore_mismatches=ignore_mismatches
    )
    merge_bn(state_dict, prefix + '.bn2', a_block.bn2, b_block.bn2, conv2_transform)
    
    if shortcut:
        shortcut_transform = merge_hidden_conv(
            state_dict, 
            prefix + '.shortcut.0', 
            a_block.shortcut[0], 
            b_block.shortcut[0], 
            input_transform,
            output_transform=conv2_transform,
            ignore_mismatches=ignore_mismatches
        )
        merge_bn(
            state_dict, 
            prefix + '.shortcut.1', 
            a_block.shortcut[1], 
            b_block.shortcut[1], 
            shortcut_transform
        )
    
    return conv2_transform

hard_pass = lambda : None

def merge_resnet20(state_dict, a, b, transforms, concat_head=False, ignore_mismatches=False):
    transforms['conv1'] = merge_first_convs(
        state_dict, 'conv1', a.conv1, b.conv1, 
        output_transform=transforms['conv1']
    )
    merge_bn(state_dict, 'bn1', a.bn1, b.bn1, transforms['conv1'])
    
    for i in range(3):
        merge_block(
            state_dict, f'layer1.{i}', a.layer1[i], b.layer1[i], 
            input_transform=transforms['conv1'], 
            intra_transform=transforms[f'block1.{i}'],
            output_transform=transforms['conv1'],
            shortcut=False,
            ignore_mismatches=ignore_mismatches
        )
    
    transforms['block2'] = merge_block(
        state_dict, 'layer2.0', a.layer2[0], b.layer2[0], 
        input_transform=transforms['conv1'], 
        intra_transform=transforms[f'block2.0'],
        output_transform=transforms['block2'],
        shortcut=True,
        ignore_mismatches=ignore_mismatches
    )
    
    for i in range(1, 3):
        merge_block(
            state_dict, f'layer2.{i}', a.layer2[i], b.layer2[i], 
            input_transform=transforms['block2'], 
            intra_transform=transforms[f'block2.{i}'],
            output_transform=transforms['block2'],
            shortcut=False,
            ignore_mismatches=ignore_mismatches
        )
        
    transforms['block3'] = merge_block(
        state_dict, 'layer3.0', a.layer3[0], b.layer3[0], 
        input_transform=transforms['block2'], 
        intra_transform=transforms[f'block3.0'],
        output_transform=transforms['block3'],
        shortcut=True,
        ignore_mismatches=ignore_mismatches
    )
    for i in range(1, 3):
        merge_block(
            state_dict, f'layer3.{i}', a.layer3[i], b.layer3[i], 
            input_transform=transforms['block3'], 
            intra_transform=transforms[f'block3.{i}'],
            output_transform=transforms['block3'],
            shortcut=False,
            ignore_mismatches=ignore_mismatches
        )
        
    output_align_identity = torch.eye(a.linear.weight.shape[0], device=a.linear.weight.device)
    output_align_mat = torch.cat((output_align_identity/2, output_align_identity/2), dim=1)
    transforms['linear'].output_align = output_align_mat
    transforms['linear'] = merge_linear(
        state_dict, 'linear', a.linear, b.linear, 
        transforms['block3'], transforms['linear'],
        recompute_output=False,
        only_input_align=concat_head,
        ignore_mismatches=ignore_mismatches
    )
    
    return transforms
