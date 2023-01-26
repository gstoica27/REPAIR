import torch
from torch import nn
import torchvision
import numpy as np
import scipy


r = 0

def set_r(_rr):
    global r
    r = _rr

    
def match_tensors_exact_bipartite(
    hull_tensor,
    interleave=False,
    random_perm=False
):
    hull_normed = hull_tensor / hull_tensor.norm(dim=-1, keepdim=True)
    O = hull_tensor.shape[0]
    remainder = int(hull_tensor.shape[0] * (1-r))
    bound = O - remainder
    sims = hull_normed @ hull_normed.transpose(-1, -2)
    torch.diagonal(sims)[:] = -torch.inf
    permutation_matrix = torch.zeros((O, O - bound), device=sims.device)
    for i in range(bound):
        best_idx = sims.view(-1).argmax()
        row_idx = best_idx % sims.shape[1]
        col_idx = best_idx // sims.shape[1]
        permutation_matrix[row_idx, i] = 1
        permutation_matrix[col_idx, i] = 1
        sims[row_idx] = -torch.inf
        sims[col_idx] = -torch.inf
        sims[:, row_idx] = -torch.inf
        sims[:, col_idx] = -torch.inf
    
    unused = (sims.max(-1)[0] > -torch.inf).to(torch.int).nonzero().view(-1)
    for i in range(bound, O-bound):
        permutation_matrix[unused[i-bound], i] = 1
    merge = permutation_matrix / (permutation_matrix.sum(dim=0, keepdim=True) + 1e-5)
    unmerge = permutation_matrix
    return merge.T, unmerge

def do_nothing(x, mode=None):
    return x


def bipartite_soft_matching(
    metric: torch.Tensor,
    r: float,
    class_token: bool = False,
    distill_token: bool = False,
):
    """
    Applies ToMe with a balanced matching set (50%, 50%).
    Input size is [batch, tokens, channels].
    r indicates the ratio of tokens to remove (max 50% of tokens).
    Extra args:
     - class_token: Whether or not there's a class token.
     - distill_token: Whether or not there's also a distillation token.
    When enabled, the class token and distillation tokens won't get merged.
    """
    protected = 0
    if class_token:
        protected += 1
    if distill_token:
        protected += 1

    # We can only reduce by a maximum of 50% tokens
    t = metric.shape[1]
    r = t - int((1 - r) * t)
    r = min(r, (t - protected) // 2)

    if r <= 0:
        return do_nothing, do_nothing

    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = metric.chunk(2, dim=-2)
        scores = a @ b.transpose(-1, -2)

        if class_token:
            scores[..., 0, :] = -math.inf
        if distill_token:
            scores[..., :, 0] = -math.inf

        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
        src_idx = edge_idx[..., :r, :]  # Merged Tokens
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)

        if class_token:
            # Sort to ensure the class token is at the start
            unm_idx = unm_idx.sort(dim=1)[0]

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = x.chunk(2, dim=-2)
        n, t1, c = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)

        if distill_token:
            return torch.cat([unm[:, :1], dst[:, :1], unm[:, 1:], dst[:, 1:]], dim=1)
        else:
            return torch.cat([unm, dst], dim=1)

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        n, _, c = unm.shape

        src = dst.gather(dim=-2, index=dst_idx.expand(n, r, c))

        out = torch.zeros(n, metric.shape[1], c, device=x.device, dtype=x.dtype)

        out[..., dst.shape[-2]:, :] = dst
        out.scatter_(dim=-2, index=(unm_idx).expand(n, unm_len, c), src=unm)
        out.scatter_(dim=-2, index=(src_idx).expand(n, r, c), src=src)

        return out
    
    return merge, unmerge


def general_soft_matching(
    hull_tensor,
    interleave=False,
    random=False,
    r=.5
):  
    hull_tensor = hull_tensor[0]
    hull_normed = hull_tensor / hull_tensor.norm(dim=-1, keepdim=True)
    
    bound = int(hull_tensor.shape[0] * (1-r))
    
    sims = hull_normed @ hull_normed.transpose(-1, -2)
    uppertri_indices = torch.triu_indices(sims.shape[-2], sims.shape[-1], offset=0)
    sims[uppertri_indices[0], uppertri_indices[1]] = -torch.inf
    candidate_scores, candidate_indices = sims.max(-1)
    argsorted_scores = candidate_scores.argsort(descending=True)
    merge_indices = argsorted_scores[:bound]
    unmerge_indices = argsorted_scores[bound:]
    
    roots = torch.arange(sims.shape[0], device=sims.device)
    for _ in range(bound-1):
        roots[merge_indices] = roots[candidate_indices[merge_indices]]
    
    def merge(x, mode='mean'):
        x = x[0]
        merge_tensor = x.scatter_reduce(
            0, 
            roots[merge_indices][:, None].expand(bound, x.shape[1]),
            x[merge_indices], 
            reduce='mean'
        )
        unmerge_tensor = merge_tensor[unmerge_indices]
        return unmerge_tensor[None]
    
    def unmerge(x):
        x = x[0]
        import pdb; pdb.set_trace()
        out = torch.zeros((hull_tensor.shape[0], x.shape[1]), device=x.device)
        out.scatter_(
            0,
            index=unmerge_indices[:, None].expand(*x.shape),
            src=x
        )
        out = out.scatter(
            0,
            index=merge_indices[:, None].expand(*x.shape),
            src=out[roots[merge_indices]]
        )
        return out[None]
    
    return merge, unmerge

def concat_mats(args, dim=0):
    return torch.cat(args, dim=dim)

def unconcat_mat(tensor, dim=0):
    return torch.chunk(tensor, chunks=2, dim=dim)

def match_tensors_permute(
    hull_tensor, eps=1e-7, interleave=False, random_perm=False,
    backend_alg=bipartite_soft_matching
):
    """
    hull_tensor: [2O,I]
    """
    O, I = hull_tensor.shape
    O //= 2
    
    interleave_mat = torch.eye(2*O, device=hull_tensor.device)
    if interleave:
        A1, A2, B1, B2 = interleave_mat.chunk(4, dim=0)
        interleave_mat = torch.cat([A1, B1, A2, B2], dim=0)
        interleave_mat = interleave_mat.view(2, O, 2*O).transpose(0, 1).reshape(2*O, 2*O)
#         interleave_mat = interleave_mat[torch.randperm(2*O, device=hull_tensor.device)]
    
    hull_tensor = interleave_mat @ hull_tensor
    
    hull_tensor = hull_tensor / (hull_tensor.norm(dim=-1, keepdim=True) + eps)
    A, B = unconcat_mat(hull_tensor, dim=0)
    scores = -(A @ B.T)
    
    O_eye = torch.eye(O, device=hull_tensor.device)
    
    try:
        row_idx, col_idx = scipy.optimize.linear_sum_assignment(scores.cpu().numpy())
    except ValueError:
        pdb.set_trace()
    
    A_perm = O_eye[torch.from_numpy(row_idx)]#[perm]
    B_perm = O_eye[torch.from_numpy(col_idx)]#[perm]
    
    if random_perm:
        perm = torch.randperm(O, device=A.device)
        A_perm = A_perm[perm]
        B_perm = B_perm[perm]
    
    merge = (torch.cat((A_perm, B_perm), dim=1) / 2.) @ interleave_mat
    unmerge = interleave_mat.T @ (torch.cat((A_perm.T, B_perm.T), dim=0))
    return merge, unmerge


def match_tensors_tome(
    hull_tensor, eps=1e-7, interleave=False, 
    random_perm=False, backend_alg=bipartite_soft_matching
):
    """
    hull_tensor: [2O,I]
    """
    O, I = hull_tensor.shape
    O //= 2
    
    big_eye = torch.eye(2*O, device=hull_tensor.device)
    small_eye = torch.eye(O, device=hull_tensor.device)
    
    interleave_mat = big_eye
    if interleave:
        A1, A2, B1, B2 = interleave_mat.chunk(4, dim=0)
        interleave_mat = torch.cat([A1, B1, A2, B2], dim=0)
    
    
    hull_tensor = interleave_mat @ hull_tensor
    
    merge, unmerge = backend_alg(hull_tensor[None], 0.5)
    
    merge_mat = merge(big_eye[None])[0] @ interleave_mat
    unmerge_mat = interleave_mat.T @ unmerge(small_eye[None])[0]
    return merge_mat, unmerge_mat


def compress_tensors_tome(
    hull_tensor, eps=1e-7, interleave=False, 
    random_perm=False, backend_alg=bipartite_soft_matching
):
    """
    hull_tensor: [2O,I]
    """
    O, I = hull_tensor.shape
    O //= 2
    
    Oc = int(O * (1-r))
    
    big_eye = torch.eye(2*O, device=hull_tensor.device)
    small_eye = torch.eye(O, device=hull_tensor.device)
    
    interleave_mat = big_eye[:O, :]
    
    
    hull_tensor = interleave_mat @ hull_tensor
    
    merge, unmerge = backend_alg(hull_tensor[None], r)
    
    merge_mat = merge(big_eye[None, :O, :])[0] # @ interleave_mat
    unmerge_mat = interleave_mat.T @ unmerge(small_eye[None, :Oc, :Oc])[0]
    return merge_mat, unmerge_mat


def compress_tensors_exact_bipartite(
    hull_tensor,
    interleave=False,
    random_perm=False
):
    hull_normed = hull_tensor / hull_tensor.norm(dim=-1, keepdim=True)
    hull_normed = hull_normed.chunk(2, dim=0)[0]
    O = hull_tensor.shape[0] // 2
    remainder = int(O * (1-r))
    bound = O - remainder
    sims = hull_normed @ hull_normed.transpose(-1, -2)
    torch.diagonal(sims)[:] = -torch.inf
    permutation_matrix = torch.zeros((O, O - bound), device=sims.device)
    for i in range(bound):
        best_idx = sims.view(-1).argmax()
        row_idx = best_idx % sims.shape[1]
        col_idx = best_idx // sims.shape[1]
        permutation_matrix[row_idx, i] = 1
        permutation_matrix[col_idx, i] = 1
        sims[row_idx] = -torch.inf
        sims[col_idx] = -torch.inf
        sims[:, row_idx] = -torch.inf
        sims[:, col_idx] = -torch.inf
    
    unused = (sims.max(-1)[0] > -torch.inf).to(torch.int).nonzero().view(-1)
    for i in range(bound, O-bound):
        permutation_matrix[unused[i-bound], i] = 1
    zeros = torch.zeros_like(permutation_matrix)
    all_transform = concat_mats((permutation_matrix, zeros), dim=0)
    merge = all_transform / (all_transform.sum(dim=0, keepdim=True) + 1e-5)
    unmerge = all_transform
    return merge.T, unmerge


def expand_tensors_tome(
    hull_tensor, eps=1e-7, interleave=False, 
    random_perm=False, backend_alg=bipartite_soft_matching
):
    """
    hull_tensor: [2O,I]
    """
    O, I = hull_tensor.shape
    O //= 2
    
    Oc = int(2 * O * (1-r))
    
    big_eye = torch.eye(2*O, device=hull_tensor.device)
    small_eye = torch.eye(O, device=hull_tensor.device)
    
    interleave_mat = big_eye
    
    
    hull_tensor = interleave_mat @ hull_tensor
    
    merge, unmerge = backend_alg(hull_tensor[None], r)
    
    merge_mat = merge(big_eye[None])[0] # @ interleave_mat
    unmerge_mat = interleave_mat.T @ unmerge(big_eye[None, :Oc, :Oc])[0]
    return merge_mat, unmerge_mat


def get_procrustes(corr_mtx):
    U, _, Vh = torch.linalg.svd(corr_mtx)
    S = torch.eye(U.shape[0], device=U.device)
    S[-1, -1] = -1.
    return (U @ S) @ Vh

def match_tensors_procrustes(
    hull_tensor, eps=1e-7, 
    interleave=False, random_perm=False,
    backend_alg=bipartite_soft_matching
):
    """
    hull_tensor: [2O,I]
    """
    O, I = hull_tensor.shape
    O //= 2
    
    interleave_mat = torch.eye(2*O, device=hull_tensor.device)
    if interleave:
        A1, A2, B1, B2 = interleave_mat.chunk(4, dim=0)
        interleave_mat = torch.cat([A1, B1, A2, B2], dim=0)
        interleave_mat = interleave_mat.view(2, O, 2*O).transpose(0, 1).reshape(2*O, 2*O)
#         interleave_mat = interleave_mat[torch.randperm(2*O, device=hull_tensor.device)]
    
    hull_tensor = interleave_mat @ hull_tensor
    
    hull_tensor = hull_tensor / (hull_tensor.norm(dim=-1, keepdim=True) + eps)
    A, B = unconcat_mat(hull_tensor, dim=0)
    scores = (A @ B.T)
    
    P = get_procrustes(scores).T
    
    O_eye = torch.eye(O, device=hull_tensor.device)
    
    try:
        row_idx, col_idx = scipy.optimize.linear_sum_assignment(scores.cpu().numpy())
    except ValueError:
        pdb.set_trace()
    
    A_perm = O_eye[torch.from_numpy(row_idx)]#[perm]
    B_perm = P #O_eye[torch.from_numpy(col_idx)]#[perm]
    
    if random_perm:
        perm = torch.randperm(O, device=A.device)
        A_perm = A_perm[perm]
        B_perm = B_perm[perm]
    
    merge = (torch.cat((A_perm, B_perm), dim=1) / 2.) @ interleave_mat
    unmerge = interleave_mat.T @ (torch.cat((A_perm.T, B_perm.T), dim=0))
    return merge, unmerge



# def match_tensors_procrustes(hull_tensor, use_S=True, interleave=False, random_perm=False):
#     # We can only reduce by a maximum of 50% tokens
#     t = hull_tensor.shape[0]
#     r = int(.f5 * t)
#     with torch.no_grad():
#         A, B = unconcat_mat(hull_tensor, dim=0)
#         scores = -(A @ B.T)
#         U, S, V = torch.svd(scores)
        
# #         U[:, :rank] /= S[None, :]
#         U_r = U[:, :r]
#         U_r[:, rank:] = 0
#         S_r = torch.diag(S[:r]) if use_S else torch.eye(r, device=DEVICE)
# #         pdb.set_trace()
# #         V_r = V[:, :r]
#     merge_mat = U_r.T
#     unmerge_mat = U_r
#     return merge_mat, unmerge_mat

# match_tensors = match_tensors

def match_wrapper(fn, interleave=False, random_perm=False, backend_alg=bipartite_soft_matching):
    return lambda x: fn(x, interleave=interleave, random_perm=random_perm, backend_alg=bipartite_soft_matching)