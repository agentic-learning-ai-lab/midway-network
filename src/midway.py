import copy
from functools import partial
import logging

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from src.vision_transformer import DINOHead, Block, trunc_normal_, Mlp, CrossBlock

log = logging.getLogger(__name__)


def get_2d_sincos_pos_embed(embed_dim, grid_size, n_cls_token=0):
    """
    grid_size: tuple (height, width) of the grid
    return:
    pos_embed: [grid_size[0]*grid_size[1], embed_dim] or [n_cls_token+grid_size[0]*grid_size[1], embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size[0], dtype=np.float32)
    grid_w = np.arange(grid_size[1], dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size[0], grid_size[1]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if n_cls_token>0:
        pos_embed = np.concatenate([np.zeros([n_cls_token, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

class GatingMlp(nn.Module):
    def __init__(self, dim, gating_dim, bias=None, tau=None, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm_layer = norm_layer(dim)
        self.mlp = Mlp(dim, dim, gating_dim)
        self.bias = bias
        self.tau = tau
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.norm_layer(x)
        x = self.mlp(x)
        if self.tau is not None:
            x = x / self.tau
        if self.bias is not None:
            x = x + self.bias
        return self.sigmoid(x)

class MidwayLoss(nn.Module):
    def __init__(self,
                 # dino head and loss parameters
                 embed_dim,
                 out_dim,
                 use_bn,
                 teacher_temp_schedule,
                 patch_grid_size=None,
                 student_temp=0.1,
                 center_momentum=0.9,
                 norm_last_layer=False,
                 # latent motion parameters
                 use_motion_crops=False,
                 use_pos_embed=True,
                 feature_level_idx=[11],
                 feature_block_type='cross',
                 feature_depth=1,
                 num_feature_heads=6,
                 shared_feature_block=False,
                 train_lateral=True,
                 use_cls_token=False,
                 motion_dim=192,
                 motion_depth=2,
                 num_motion_heads=6,
                 motion_tokens=10,
                 motion_agg_type='identity',
                 motion_pred_input=True,
                 shared_motion=True,
                 gating_type=None,
                 gating_bias=None,
                 gating_tau=None,
                 pred_type='self',
                 predictor_depth=4,
                 pred_tau=None,
                 num_pred_heads=6,
                 pos_embed_mode='learn',
                 shared_forward=True,
                 motion_head_type='linear',
                 target_type='backward',
                 patch_size=16,
                 # transformer parameters
                 mlp_ratio: float=4.,
                 qkv_bias: bool=False,
                 qk_scale: float=None,
                 drop_rate: float=0.,
                 attn_drop_rate: float=0.,
                 drop_path_rate: float=0.,
                 norm_layer=nn.LayerNorm,
                 num_cls_tokens=1,
                 **kwargs):
        super().__init__()
        self.num_cls_tokens = num_cls_tokens
        self.use_motion_crops = use_motion_crops
        if self.use_motion_crops:
            self.crop_pairs = [(0, 1), (1, 0)]
        else:
            self.crop_pairs = [(0, 3), (3, 0), (1, 2), (2, 1)]

        self.use_cls_token = use_cls_token
        num_pos = None
        if patch_grid_size is not None:
            num_pos = patch_grid_size[0] * patch_grid_size[1]
            if use_cls_token:
                num_pos += 1
        
        # HACK: for now, we assume that the number of patches is constant within and across training/inference
        self.use_pos_embed = use_pos_embed
        self.pos_embed_mode = pos_embed_mode
        if use_pos_embed:
            if num_pos is None:
                num_pos = 1
            if 'learn' in pos_embed_mode:
                self.pos_embed = nn.Parameter(torch.zeros(1, num_pos*2, motion_dim))
            elif pos_embed_mode == 'sincos':
                self.register_buffer('pos_embed', torch.from_numpy(get_2d_sincos_pos_embed(motion_dim, patch_grid_size, int(use_cls_token))).float())
                self.student_pos_offset = nn.Parameter(torch.randn(1, 1, motion_dim))
        
        self.feature_level_idx = feature_level_idx
        self.feature_block_type = feature_block_type
        self.shared_feature_block = shared_feature_block
        self.train_lateral = train_lateral
        if feature_block_type == 'cross':
            if shared_feature_block:
                self.student_feature_blocks = nn.ModuleList([
                    CrossBlock(
                        dim=embed_dim, num_heads=num_feature_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate, drop_path=0, norm_layer=norm_layer)
                    for _ in range(feature_depth)])
            else:
                for latent_level in range(len(feature_level_idx)-1):
                    student_feature_blocks = nn.ModuleList([
                        CrossBlock(
                            dim=embed_dim, num_heads=num_feature_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=0, norm_layer=norm_layer)
                        for _ in range(feature_depth)])
                    setattr(self, f"student_feature_blocks_{latent_level}", student_feature_blocks)

        self.student_embed = nn.Linear(embed_dim, motion_dim)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, motion_depth)]  # stochastic depth decay rule
        
        self.shared_motion = shared_motion
        self.motion_pred_input = motion_pred_input
        if shared_motion:
            self.motion_blocks = nn.ModuleList([
                Block(
                    dim=motion_dim, num_heads=num_motion_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[j], norm_layer=norm_layer)
                for j in range(motion_depth)])
        else:
            for latent_level in range(len(feature_level_idx)-1):
                motion_blocks = nn.ModuleList([
                    Block(
                        dim=motion_dim, num_heads=num_motion_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[j], norm_layer=norm_layer)
                    for j in range(motion_depth)])
                setattr(self, f"motion_blocks_{latent_level}", motion_blocks)
        
        self.motion_tokens = nn.Parameter(torch.zeros(1, motion_tokens, motion_dim))
        self.motion_agg_type = motion_agg_type  # options are ['identity', 'add', 'concat']
        self.motion_proj = nn.Linear(motion_dim, embed_dim)

        self.gating_type = None
        if gating_type is not None:
            gating_type, gating_out = gating_type.split('-')
            if gating_out == 'vector':
                gating_dim = embed_dim
            elif gating_out == 'scalar':
                gating_dim = 1
            else:
                raise NotImplementedError(f"Gating type {gating_type} not implemented.")
            
            self.gating_type = gating_type
            for latent_level in range(len(feature_level_idx)-1):
                if gating_type in ['initial', 'all']:
                    initial_gating_mlp = GatingMlp(motion_dim, gating_dim, gating_bias, gating_tau)
                    setattr(self, f"initial_gating_mlp_{latent_level}", initial_gating_mlp)
                if gating_type in ['pred', 'all']:
                    gating_mlps = nn.ModuleList([
                        GatingMlp(embed_dim, gating_dim, gating_bias, gating_tau) for _ in range(predictor_depth - 1)])
                    setattr(self, f"gating_mlps_{latent_level}", gating_mlps)
                if gating_type == 'gating_blk':
                    gating_blks = nn.ModuleList([
                        Block(
                            dim=embed_dim, num_heads=num_motion_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=0, norm_layer=norm_layer)
                    for _ in range(predictor_depth)])
                    setattr(self, f"gating_blks_{latent_level}", gating_blks)
                    gating_mlps = nn.ModuleList([
                        GatingMlp(embed_dim, gating_dim, gating_bias, gating_tau) for _ in range(predictor_depth)])
                    setattr(self, f"gating_mlps_{latent_level}", gating_mlps)
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, predictor_depth)]  # stochastic depth decay rule
        self.predictor_depth = predictor_depth
        
        self.shared_forward = shared_forward
        self.pred_type = pred_type
        if pred_type != 'self':
            self.pred_tokens = nn.Parameter(torch.zeros(1, 1, embed_dim))
            if pos_embed_mode == 'learn':
                self.pred_pos_embed = nn.Parameter(torch.zeros(1, num_pos, embed_dim))
            elif pos_embed_mode == 'learn-levels':
                self.pred_pos_embed = nn.Parameter(torch.zeros(len(feature_level_idx)-1, num_pos, embed_dim))
            elif pos_embed_mode == 'sincos':
                self.register_buffer('pred_pos_embed', torch.from_numpy(get_2d_sincos_pos_embed(embed_dim, patch_grid_size, int(use_cls_token))).float())

        if pred_type == 'self':
            pred_block = Block
        elif pred_type in ['cross', 'cross-query']:
            pred_block = partial(CrossBlock, tau=pred_tau)
        else:
            raise NotImplementedError(f"Predictor type {pred_type} not implemented.")
        if shared_forward:
            self.forward_predictor = nn.ModuleList([
                pred_block(
                    dim=embed_dim, num_heads=num_pred_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[j], norm_layer=norm_layer)
                for j in range(self.predictor_depth)])
        else:
            for latent_level in range(len(feature_level_idx)-1):
                forward_predictor = nn.ModuleList([
                    pred_block(
                        dim=embed_dim, num_heads=num_pred_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[j], norm_layer=norm_layer)
                    for j in range(self.predictor_depth)])
                setattr(self, f"forward_predictor_{latent_level}", forward_predictor)

        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.teacher_temp_schedule = teacher_temp_schedule

        self.motion_head_type = motion_head_type
        if motion_head_type == 'identity':
            self.student_head = nn.Identity()
        elif motion_head_type == 'linear':
            self.student_head = nn.Linear(embed_dim, embed_dim)
        elif motion_head_type == 'mlp':
            self.student_head = Mlp(embed_dim, embed_dim, embed_dim)
        elif motion_head_type == 'dino':
            self.student_head = DINOHead(
                embed_dim,
                out_dim,
                use_bn=use_bn,
                norm_last_layer=norm_last_layer,
            )
            self.teacher_head = DINOHead(
                embed_dim,
                out_dim,
                use_bn=use_bn,
            )
            for i in range(len(feature_level_idx)-1):
                self.register_buffer(f"center_{i}", torch.zeros(1, 1, out_dim))
        else:
            raise NotImplementedError(f"Head type {motion_head_type} not implemented.")
        
        self.apply(self._init_weights)

        self.teacher_embed = copy.deepcopy(self.student_embed)
        
        if self.motion_head_type == 'dino':
            self.teacher_head.load_state_dict(self.student_head.state_dict(), strict=False)
        else:
            self.teacher_head = copy.deepcopy(self.student_head)
        
        if self.feature_block_type == 'cross':
            if self.shared_feature_block:
                self.teacher_feature_blocks = copy.deepcopy(self.student_feature_blocks)
            else:
                for latent_level in range(len(feature_level_idx)-1):
                    setattr(self, f"teacher_feature_blocks_{latent_level}", copy.deepcopy(getattr(self, f"student_feature_blocks_{latent_level}")))
        
        self.target_type = target_type
        self.patch_size = patch_size 
        
        for p in self.teacher_parameters:
            p.requires_grad = False
            
    @property
    def student_parameters(self):
        for module in [self.student_embed, self.student_head]:
            for p in module.parameters():
                yield p
        if self.feature_block_type == 'cross':
            if self.shared_feature_block:
                for p in self.student_feature_blocks.parameters():
                    yield p
            else:
                for latent_level in range(len(self.feature_level_idx)-1):
                    for p in getattr(self, f"student_feature_blocks_{latent_level}").parameters():
                        yield p

    @property
    def teacher_parameters(self):
        for module in [self.teacher_embed, self.teacher_head]:
            for p in module.parameters():
                yield p
        if self.feature_block_type == 'cross':
            if self.shared_feature_block:
                for p in self.teacher_feature_blocks.parameters():
                    yield p
            else:
                for latent_level in range(len(self.feature_level_idx)-1):
                    for p in getattr(self, f"teacher_feature_blocks_{latent_level}").parameters():
                        yield p

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, student_output, teacher_output, midway_weights, epoch_iter, loss_masks=None):
        metric_dict = {}
        motion_loss = 0
        losses = []
        student_temp = self.student_temp
        teacher_temp = self.teacher_temp_schedule[epoch_iter]
        ncrops = len(self.crop_pairs)
        student_features = []
        for i, feat in enumerate(student_output):
            if i in self.feature_level_idx:
                feat = feat.chunk(ncrops)
                feat = torch.cat([feat[v] for v, iq in self.crop_pairs], dim=0)
                if not self.use_cls_token:
                    feat = feat[:, self.num_cls_tokens:]
                student_features.append(feat)
        teacher_features = []
        for i, feat in enumerate(teacher_output):
            if i in self.feature_level_idx:
                feat = feat.chunk(ncrops)
                feat = torch.cat([feat[iq] for v, iq in self.crop_pairs], dim=0)
                if not self.use_cls_token:
                    feat = feat[:, self.num_cls_tokens:]
                teacher_features.append(feat)
        if loss_masks is not None:
            loss_masks = loss_masks.chunk(ncrops)
            loss_masks = torch.cat([loss_masks[iq] for v, iq in self.crop_pairs], dim=0)
            loss_masks = F.interpolate(loss_masks, scale_factor=(1. / self.patch_size), mode='bilinear')
            loss_masks = (loss_masks > 0.5).float()
            loss_masks = loss_masks.flatten(1)
            if self.use_cls_token:
                ones = torch.ones((loss_masks.shape[0], 1), device=loss_masks.device)
                loss_masks = torch.cat([ones, loss_masks], dim=1)
        
        x1 = student_features[-1]
        _x1 = x1 # predicted features
        if not self.train_lateral:
            _x1 = _x1.detach()
        x2 = teacher_features[-1]
        B, L, D = x1.shape
        m = self.motion_tokens.expand(B, -1, -1)

        for i in range(len(self.feature_level_idx)-2, -1, -1):
            if self.motion_pred_input:
                _x1 = self.student_embed(_x1)
            else:
                _x1 = self.student_embed(x1)
            _x2 = self.teacher_embed(x2).detach()

            if self.use_pos_embed:
                if 'learn' in self.pos_embed_mode:
                    if self.pos_embed.shape[1] == 2:
                        _x1 = _x1 + self.pos_embed[:, 0:1].expand(B, L, -1)
                        _x2 = _x2 + self.pos_embed[:, 1:2].expand(B, L, -1)
                    else:
                        _x1 = _x1 + self.pos_embed[:, :L].expand(B, -1, -1)
                        _x2 = _x2 + self.pos_embed[:, L:].expand(B, -1, -1)
                elif self.pos_embed_mode == 'sincos':
                    _x1 = _x1 + self.pos_embed.unsqueeze(0).expand(B, -1, -1)
                    _x1 = _x1 + self.student_pos_offset
                    _x2 = _x2 + self.pos_embed.unsqueeze(0).expand(B, -1, -1)
            
            if self.shared_motion:
                motion_blocks = self.motion_blocks
            else:
                motion_blocks = getattr(self, f"motion_blocks_{i}")
            
            old_m = m.clone()
            motion_input = torch.cat([m, _x1, _x2], dim=1)
            for blk in motion_blocks:
                motion_input = blk(motion_input)
            m = motion_input[:, :m.shape[1]]

            if i != len(self.feature_level_idx)-2:
                if self.motion_agg_type == 'identity':
                    pass
                elif self.motion_agg_type == 'add':
                    m = m + old_m
                elif self.motion_agg_type == 'concat':
                    m = torch.cat([old_m, m], dim=1)
            
            lower_x1 = student_features[i]
            lower_x2 = teacher_features[i]
            if not self.train_lateral:
                lower_x1 = lower_x1.detach()
                lower_x2 = lower_x2.detach()
            if self.feature_block_type == 'cross':
                if self.shared_feature_block:
                    student_feature_blocks = self.student_feature_blocks
                    teacher_feature_blocks = self.teacher_feature_blocks
                else:
                    student_feature_blocks = getattr(self, f"student_feature_blocks_{i}")
                    teacher_feature_blocks = getattr(self, f"teacher_feature_blocks_{i}")
                
                for blk in student_feature_blocks:
                    lower_x1, x1 = blk(lower_x1, x1)
                
                with torch.no_grad():
                    for blk in teacher_feature_blocks:
                        lower_x2, x2 = blk(lower_x2, x2)

            if self.shared_forward:
                forward_predictor = self.forward_predictor
            else:
                forward_predictor = getattr(self, f"forward_predictor_{i}")

            if self.pred_type == 'self':
                pred_tokens = lower_x1
                _x1 = torch.cat([self.motion_proj(m), pred_tokens], dim=1)
                M = m.shape[1]
                gating = None
                if self.gating_type in ['initial', 'all']:
                    gating_mlp = getattr(self, f"initial_gating_mlp_{i}")
                    gating = gating_mlp(motion_input[:, M:M+L])
                    metric_dict[f'init-gating-lvl{i}-mean'] = gating.mean().detach()
                    metric_dict[f'init-gating-lvl{i}-var'] = gating.var().detach()
                    D = gating.shape[2]
                    ones = torch.ones((B, M, D), device=_x1.device)
                    gating = torch.cat([ones, gating], dim=1)
                gating_mlps = []
                if self.gating_type in ['pred', 'all']:
                    gating_mlps = getattr(self, f"gating_mlps_{i}")
                if self.gating_type == 'gating_blk':
                    gating_x1 = _x1
                    b = 0
                    for gating_blk, gating_mlp, blk in zip(getattr(self, f"gating_blks_{i}"), getattr(self, f"gating_mlps_{i}"), forward_predictor):
                        gating_x1 = gating_blk(gating_x1)
                        next_gating = gating_mlp(gating_x1[:, M:M+L])
                        metric_dict[f'gating-lvl{i}-blk{b}-mean'] = next_gating.mean().detach()
                        metric_dict[f'gating-lvl{i}-blk{b}-var'] = next_gating.var().detach()
                        D = next_gating.shape[2]
                        ones = torch.ones((B, M, D), device=_x1.device)
                        next_gating = torch.cat([ones, next_gating], dim=1)
                        _x1 = blk(_x1, gating=next_gating)
                        b += 1
                else:
                    for b, blk in enumerate(forward_predictor):
                        next_gating = None
                        if b < len(gating_mlps):
                            next_gating = gating_mlps[b](_x1[:, M:M+L])
                            metric_dict[f'pred-gating-lvl{i}-blk{b}-mean'] = next_gating.mean().detach()
                            metric_dict[f'pred-gating-lvl{i}-blk{b}-var'] = next_gating.var().detach()
                            D = next_gating.shape[2]
                            ones = torch.ones((B, M, D), device=_x1.device)
                            next_gating = torch.cat([ones, next_gating], dim=1)
                        _x1 = blk(_x1, gating=gating)
                        gating = next_gating
                _x1 = _x1[:, M:]
            elif self.pred_type in ['cross', 'cross-query']:
                pred_tokens = self.pred_tokens.expand(B, L, -1)
                if self.pos_embed_mode == 'learn':
                    pred_pos_embed = self.pred_pos_embed
                elif self.pos_embed_mode == 'learn-levels':
                    pred_pos_embed = self.pred_pos_embed[i].unsqueeze(0)
                elif self.pos_embed_mode == 'sincos':
                    pred_pos_embed = self.pred_pos_embed.squeeze(0)
                pred_tokens = pred_tokens + pred_pos_embed.expand(B, -1, -1)
                m_proj = self.motion_proj(m)
                _x1 = torch.cat([m_proj, pred_tokens], dim=1)
                q_tokens = None
                if self.pred_type =='cross-query':
                    q_tokens = torch.cat([m_proj, lower_x1], dim=1)
                M = m.shape[1]
                for blk in forward_predictor:
                    _x1, lower_x1 = blk(_x1, lower_x1, q_tokens)
                _x1 = _x1[:, M:]
            s1 = self.student_head(_x1)
            with torch.no_grad():
                if self.target_type == 'backward':
                    s2 = self.teacher_head(lower_x2)
                elif self.target_type == 'forward':
                    s2 = self.teacher_head(teacher_features[i])
                else:
                    raise NotImplementedError(f"Target type {self.target_type} not implemented.")
            if self.motion_head_type == 'dino':
                s1 = s1 / student_temp
                with torch.no_grad():
                    center = getattr(self, f"center_{i}")
                    self.update_center(s2, i)
                    s2 = F.softmax((s2 - center) / teacher_temp, dim=-1)
                loss = torch.sum(-s2.detach() * F.log_softmax(s1, dim=-1), dim=-1)
                if loss_masks is not None:
                    loss = loss * loss_masks
                    loss = loss.sum() / loss_masks.sum()
                else:
                    loss = loss.mean()
            else:
                s1 = F.normalize(s1, dim=-1)
                with torch.no_grad():
                    s2 = F.normalize(s2, dim=-1)
                loss = 2 - 2 * (s1 * s2.detach()).sum(dim=-1)
                if loss_masks is not None:
                    loss = loss * loss_masks
                    loss = loss.sum() / loss_masks.sum()
                else:
                    loss = loss.mean()
            motion_loss += (midway_weights[i] * loss)
            losses.insert(0, loss.detach().cpu())
            del loss
            torch.cuda.empty_cache()
            x1 = lower_x1
            x2 = lower_x2 
        return motion_loss, losses, metric_dict

    @torch.no_grad()
    def update_center(self, teacher_output, center_level):
        batch_center = torch.sum(teacher_output, dim=(0, 1), keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (teacher_output.shape[0] * teacher_output.shape[1] * dist.get_world_size())

        # ema update
        setattr(self, f"center_{center_level}", getattr(self, f"center_{center_level}") * self.center_momentum + batch_center * (1 - self.center_momentum))
