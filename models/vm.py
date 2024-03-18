import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.vision_transformer import _cfg, PatchEmbed
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_, DropPath
from models.base_net import LinearML, LayerNormML

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = LinearML(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = LinearML(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = LinearML(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = LinearML(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_gradients = None
        self.attention_map = None
        
    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients
        
    def get_attn_gradients(self):
        return self.attn_gradients
    
    def save_attention_map(self, attention_map):
        self.attention_map = attention_map
        
    def get_attention_map(self):
        return self.attention_map
    
    def forward(self, x, text_embeds=None, text_atts=None, register_hook=False, image_atts=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        B_fusion, N_fusion, C_fusion = text_embeds.shape
        qkv_fusion = self.qkv(text_embeds).reshape(B_fusion, N_fusion, 3, self.num_heads,
                                                   C_fusion // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if image_atts is not None:
            attn += image_atts

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
                
        if register_hook:
            self.save_attention_map(attn)
            attn.register_hook(self.save_attn_gradients)        

        # attn: (bs, num_heads, num_patches, num_patches)
        # v: (bs, num_heads, num_patches, d)
        # attn @ v: (bs, num_heads, num_patches, d)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, text_embeds=None, text_atts=None, register_hook=False, image_atts=None):
        x = x + self.drop_path(self.attn(self.norm1(x), text_embeds, text_atts,
                                         register_hook=register_hook, image_atts=image_atts))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
    
    
class VmTransformer(nn.Module):
    """ Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`  -
        https://arxiv.org/abs/2010.11929
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, representation_size=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=None, local_attn_depth=0):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()
        
        self.fc = LinearML(1024, 768)
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

        self.num_patch_embed = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.num_pos_embed = self.num_patch_embed + 1
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_pos_embed, embed_dim))

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.depth = depth
        self.local_attn_depth = local_attn_depth  # do local attn from index=(depth - local_attn_depth)

        self.norm = norm_layer(embed_dim)

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, LinearML):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, LinearML) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward(self, x, text_embeds=None, text_atts=None, register_blk=-1, idx_to_group_img=None, image_atts=None):

        x = self.fc(x)
        B = x.shape[0]
#         x = self.patch_embed(x)

#         cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
#         x = torch.cat((cls_tokens, x), dim=1)
  
#         x = x + self.pos_embed[:,:x.size(1),:]
#         x = self.pos_drop(x)

        do_gather = True if idx_to_group_img is not None else False

        if do_gather and (image_atts is not None):
            full_atts = torch.ones(x.shape[:2], dtype=x.dtype).to(x.device)
            image_atts_blk = torch.cat([image_atts, full_atts], dim=0)

            image_atts_blk = image_atts_blk.unsqueeze(1).unsqueeze(2)
            image_atts_blk = (1.0 - image_atts_blk) * -10000.0
        else:
            image_atts_blk = None

        for i, blk in enumerate(self.blocks):
            if (self.local_attn_depth > 0) and (i >= self.depth-self.local_attn_depth):
                if do_gather:
                    do_gather = False

                    x_bs = torch.gather(x, dim=0, index=idx_to_group_img.view(-1, 1, 1).expand(-1, x.shape[1], x.shape[2]))
                    x = torch.cat([x_bs, x], dim=0)

                x = blk(x, text_embeds, text_atts, register_blk == i, image_atts=image_atts_blk)

            else:
                x = blk(x, register_blk==i, image_atts=None)

        x = self.norm(x)

        if idx_to_group_img is not None:
            bs = len(idx_to_group_img)
            x_bs, x_fullatts = torch.split(x, [bs, x.size(0)-bs])
            return x_bs, x_fullatts

        return x[:, 0,:]


def im_pos_embed(pos_embed_checkpoint, num_patches, num_extra_tokens=1):
    # num_patches = visual_encoder.num_patch_embed
    # num_extra_tokens = visual_encoder.num_pos_embed - visual_encoder.num_patch_embed

    # interpolate position embedding
    embedding_size = pos_embed_checkpoint.shape[-1]
    # height (== width) for the checkpoint position embedding
    orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
    # height (== width) for the new position embedding
    new_size = int(num_patches ** 0.5)

    if orig_size != new_size:
        # class_token and dist_token are kept unchanged
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        # only the position tokens are interpolated
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        print('reshape position embedding from %d to %d' % (orig_size ** 2, new_size ** 2))

        return new_pos_embed
    else:
        return pos_embed_checkpoint
