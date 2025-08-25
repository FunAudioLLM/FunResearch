import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import einsum
from einops import rearrange
from collections import namedtuple
import math
import pdb

class SimVQ(nn.Module):
    """
    Improved version over VectorQuantizer, can be used as a drop-in replacement. Mostly
    avoids costly matrix multiplications and allows for post-hoc remapping of indices.
    """
    # NOTE: due to a bug the beta term was applied to the wrong term. for
    # backwards compatibility we use the buggy version by default, but you can
    # specify legacy=False to fix it.
    def __init__(self, n_e, e_dim, beta=0.25, remap=None, unknown_index="random",
                 sane_index_shape=True, legacy=False):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.legacy = legacy

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        nn.init.normal_(self.embedding.weight, mean=0, std=self.e_dim**-0.5)
        for p in self.embedding.parameters():
            p.requires_grad = False
        
        self.embedding_proj = nn.Linear(self.e_dim, self.e_dim)
    
        self.remap = remap
        if self.remap is not None:
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index # "random" or "extra" or integer
            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed+1
            print(f"Remapping {self.n_e} indices to {self.re_embed} indices. "
                  f"Using {self.unknown_index} for unknown indices.")
        else:
            self.re_embed = n_e

        self.sane_index_shape = sane_index_shape

    def remap_to_used(self, inds):
        ishape = inds.shape
        assert len(ishape)>1
        inds = inds.reshape(ishape[0],-1)
        used = self.used.to(inds)
        match = (inds[:,:,None]==used[None,None,...]).long()
        new = match.argmax(-1)
        unknown = match.sum(2)<1
        if self.unknown_index == "random":
            new[unknown]=torch.randint(0,self.re_embed,size=new[unknown].shape).to(device=new.device)
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)

    def unmap_to_all(self, inds):
        ishape = inds.shape
        assert len(ishape)>1
        inds = inds.reshape(ishape[0],-1)
        used = self.used.to(inds)
        if self.re_embed > self.used.shape[0]: # extra token
            inds[inds>=self.used.shape[0]] = 0 # simply set to zero
        back=torch.gather(used[None,:][inds.shape[0]*[0],:], 1, inds)
        return back.reshape(ishape)

    def forward(self, z, temp=None, rescale_logits=False, return_logits=False):
        assert temp is None or temp==1.0, "Only for interface compatible with Gumbel"
        assert rescale_logits==False, "Only for interface compatible with Gumbel"
        assert return_logits==False, "Only for interface compatible with Gumbel"
        # reshape z -> (batch, height, width, channel) and flatten
        z = rearrange(z, 'b c h w -> b h w c').contiguous()
        assert z.shape[-1] == self.e_dim
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        
        quant_codebook = self.embedding_proj(self.embedding.weight)

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(quant_codebook**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, rearrange(quant_codebook, 'n d -> d n'))

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = F.embedding(min_encoding_indices, quant_codebook).view(z.shape)
        perplexity = None
        min_encodings = None

        # compute loss for embedding
        if not self.legacy:
            commit_loss = self.beta * torch.mean((z_q.detach()-z)**2) + \
                   torch.mean((z_q - z.detach()) ** 2)
        else:
            commit_loss = torch.mean((z_q.detach()-z)**2) + self.beta * \
                   torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()

        if self.remap is not None:
            min_encoding_indices = min_encoding_indices.reshape(z.shape[0],-1) # add batch axis
            min_encoding_indices = self.remap_to_used(min_encoding_indices)
            min_encoding_indices = min_encoding_indices.reshape(-1,1) # flatten

        if self.sane_index_shape:
            min_encoding_indices = min_encoding_indices.reshape(
                z_q.shape[0], z_q.shape[2], z_q.shape[3])
            
        return (z_q, torch.tensor(0.0), min_encoding_indices), LossBreakdown(torch.tensor(0.0), torch.tensor(0.0), commit_loss, torch.tensor(0.0))

    def get_codebook_entry(self, indices, shape):
        # shape specifying (batch, height, width, channel)
        if self.remap is not None:
            indices = indices.reshape(shape[0],-1) # add batch axis
            indices = self.unmap_to_all(indices)
            indices = indices.reshape(-1) # flatten again

        # get quantized latent vectors
        z_q = self.embedding(indices)

        if shape is not None:
            z_q = z_q.view(shape)
            # reshape back to match original input shape
            z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q
    

class SimVQ1D(SimVQ):
    def forward(self, z, temp=None, rescale_logits=False, return_logits=False, mask=None):
        assert temp is None or temp==1.0, "Only for interface compatible with Gumbel"
        assert rescale_logits==False, "Only for interface compatible with Gumbel"
        assert return_logits==False, "Only for interface compatible with Gumbel"

        # reshape z -> (batch, height, width, channel) and flatten
        z = rearrange(z, 'b c h -> b h c').contiguous()
        assert z.shape[-1] == self.e_dim

        z_flattened = z.view(-1, self.e_dim)  # (B*H, C)
        if mask is not None:
            mask = rearrange(mask, 'b h -> (b h)')  # flatten mask to (B*H)
            valid_indices = mask.nonzero(as_tuple=False).squeeze(-1)  # indices where mask == True
            z_valid = z_flattened[valid_indices]  # only take valid points
        else:
            z_valid = z_flattened

        quant_codebook = self.embedding_proj(self.embedding.weight)  # projected codebook

        d = torch.sum(z_valid ** 2, dim=1, keepdim=True) + \
            torch.sum(quant_codebook ** 2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_valid, rearrange(quant_codebook, 'n d -> d n'))

        min_encoding_indices = torch.argmin(d, dim=1)  # (valid_points,)
        z_q_valid = F.embedding(min_encoding_indices, quant_codebook)  # (valid_points, C)

        # build full z_q filled with original z (for masked out areas)
        z_q_flattened = z_flattened.clone()
        z_q_flattened[valid_indices] = z_q_valid

        z_q = z_q_flattened.view(z.shape)  # reshape back to (B, H, C)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to (B, C, H)
        z_q = rearrange(z_q, 'b h c -> b c h').contiguous()

        # compute commitment loss ONLY on masked=True regions
        if mask is not None:
            z_for_loss = z_flattened[valid_indices]
            zq_for_loss = z_q_flattened[valid_indices]
        else:
            z_for_loss = z_flattened
            zq_for_loss = z_q_flattened

        if not self.legacy:
            commit_loss = self.beta * F.mse_loss(zq_for_loss.detach(), z_for_loss) + F.mse_loss(zq_for_loss, z_for_loss.detach())
        else:
            commit_loss = F.mse_loss(zq_for_loss.detach(), z_for_loss) + self.beta * F.mse_loss(zq_for_loss, z_for_loss.detach())

        # rebuild full min_encoding_indices with a placeholder (e.g., -1) where masked=False
        full_min_encoding_indices = torch.full((z_flattened.shape[0],), -1, dtype=torch.long, device=z.device)
        full_min_encoding_indices[valid_indices] = min_encoding_indices

        if self.remap is not None:
            full_min_encoding_indices = full_min_encoding_indices.reshape(z.shape[0], -1)  # add batch axis
            full_min_encoding_indices = self.remap_to_used(full_min_encoding_indices)
            full_min_encoding_indices = full_min_encoding_indices.reshape(-1, 1)  # flatten

        if self.sane_index_shape:
            full_min_encoding_indices = full_min_encoding_indices.reshape(z_q.shape[0], z_q.shape[2])
        # pdb.set_trace()

        return z_q, full_min_encoding_indices, commit_loss

    def compute_code_usage(self, embed_ind: torch.Tensor) -> torch.Tensor:
        """
        Compute per-code usage count from given indices.
        Args:
            embed_ind: [B, T] or [B*T]
        Returns:
            usage: [codebook_size] long tensor
        """
        flat = embed_ind.view(-1)
        flat = flat[flat >= 0]  # 忽略 padding 位（-1）
        usage = torch.bincount(flat, minlength=self.n_e)
        # pdb.set_trace()
        return usage

