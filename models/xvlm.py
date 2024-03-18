# Multi-Grained Vision Language Pre-Training: Aligning Texts with Visual Concepts (https://arxiv.org/abs/2111.08276)
# Github: https://github.com/zengyan-97/X-VLM
# Copyright (c) 2022, ByteDance Inc.
# All rights reserved.

import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from models.base_net import LinearML, LayerNormML
from models.clip_vit import CLIPVisionTransformer
from models.swin_transformer import SwinTransformer, interpolate_relative_pos_embed
from models.vit import VisionTransformer, interpolate_pos_embed
from models.vm import VmTransformer
from models.xbert import BertConfig, BertForMaskedLM, BertModel
from models.xroberta import RobertaConfig, RobertaForMaskedLM, RobertaModel
from utils import read_json


def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


class ContrastiveLoss(nn.Module):
    def __init__(self, batch_size=21, temperature=0.1):
        super().__init__()
        self.batch_size = batch_size
        self.delta_pos = 0.1  # delta_pos
        self.delta_neg = 0.5  # delta_neg
        self.zeros = 0

        if torch.cuda.is_available():
            self.register_buffer("temperature", torch.tensor(temperature).cuda())  # 超参数 温度
            self.register_buffer("negatives_mask", (
                ~torch.eye(batch_size * 2, batch_size * 2, dtype=bool).cuda()).float())  # 主对角线为0，其余位置全为1的mask矩阵

    def forward(self, emb_i, emb_j):  # emb_i, emb_j 是来自同一图像的两种不同的预处理方法得到
        self.batch_size = len(emb_i)
        z_i = F.normalize(emb_i, dim=1)  # (bs, dim)  --->  (bs, dim)
        z_j = F.normalize(emb_j, dim=1)  # (bs, dim)  --->  (bs, dim)

        representations = torch.cat([z_i, z_j], dim=0)  # repre: (2*bs, dim)
        if torch.cuda.is_available():
            similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0),
                                                    dim=2).cuda()  # simi_mat: (2*bs, 2*bs)

        sim_ij = torch.diag(similarity_matrix, len(emb_i))  # bs
        sim_ji = torch.diag(similarity_matrix, -len(emb_i))  # bs
        positives = torch.cat([sim_ij, sim_ji], dim=0)  # 2*bs

        nominator = torch.exp(positives / self.temperature)  # 2*bs
        denominator = self.negatives_mask[:len(emb_i) * 2, :len(emb_i) * 2] * torch.exp(
            similarity_matrix / self.temperature)  # 2*bs, 2*bs

        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))  # 2*bs
        loss = torch.sum(loss_partial) / (2 * self.batch_size)

        # =================cal regu loss=======================
        scores = torch.matmul(emb_i, emb_i.T)
        dis = torch.abs(torch.min(torch.mul((scores - self.delta_pos), (scores - self.delta_neg)),
                                  torch.zeros(scores.shape[0], scores.shape[0]).cuda()))
        dis = torch.sum(dis, dim=1)
        # print(scores,torch.mul((scores-self.delta_pos),(scores-self.delta_neg)), dis)
        return loss, dis.sum()


class AllGather(torch.autograd.Function):
    """An autograd function that performs allgather on a tensor."""

    @staticmethod
    def forward(ctx, tensor, rank, world_size):
        output = [torch.empty_like(tensor) for _ in range(world_size)]
        dist.all_gather(output, tensor)
        ctx.rank = rank
        ctx.batch_size = tensor.shape[0]
        return torch.cat(output, 0)

    @staticmethod
    def backward(ctx, grad_output):
        return (
            grad_output[ctx.batch_size * ctx.rank: ctx.batch_size * (ctx.rank + 1)],
            None,
            None
        )


allgather = AllGather.apply


def build_vision_encoder(config, load_params=False):
    """
    Args:
        load_params: False when building fine-tuning models
    """
    num_patches = (config['image_res'] // config['patch_size']) ** 2

    if config['use_clip_vit']:  # good performance, but only base model available
        vision_config = read_json(config['vision_config'])
        assert config['patch_size'] == vision_config['patch_size']
        vision_width = vision_config['vision_width']

        vision_encoder = CLIPVisionTransformer(image_size=config['image_res'], patch_size=vision_config['patch_size'],
                                               hidden_size=vision_config['vision_width'],
                                               hidden_act=vision_config['hidden_act'],
                                               num_attention_heads=vision_config['num_attention_heads'],
                                               attention_dropout=vision_config['attention_dropout'],
                                               intermediate_size=vision_config['intermediate_size'],
                                               num_hidden_layers=vision_config['num_hidden_layers'],
                                               local_attn_depth=vision_config['local_attn_depth'])

        if load_params:
            # download from https://huggingface.co/openai/clip-vit-base-patch16/tree/main
            state_dict_orig = torch.load(vision_config['ckpt'], map_location="cpu")
            state_dict = {}
            for k, v in state_dict_orig.items():
                if k.startswith('vision_model.'):
                    k = k[13:]
                    if k.startswith('embeddings.'):
                        k = k[11:]
                        k = k.replace('patch_embedding.weight', 'patch_embed.weight')
                        k = k.replace('position_embedding.weight', 'pos_embed.weight')

                    if k != 'position_ids':
                        state_dict[k] = v

            pos_embed_reshaped = interpolate_pos_embed(state_dict['pos_embed.weight'].unsqueeze(dim=0),
                                                       num_patches=num_patches, num_extra_tokens=1)
            state_dict['pos_embed.weight'] = pos_embed_reshaped.squeeze(dim=0)

    elif config['use_swin']:
        vision_config = read_json(config['vision_config'])
        assert config['image_res'] == vision_config['image_res']
        assert config['patch_size'] == 32
        vision_width = vision_config['vision_width']

        vision_encoder = SwinTransformer(img_size=vision_config['image_res'],
                                         patch_size=4,
                                         in_chans=3,
                                         embed_dim=vision_config['embed_dim'],
                                         depths=vision_config['depths'],
                                         num_heads=vision_config['num_heads'],
                                         window_size=vision_config['window_size'],
                                         mlp_ratio=4.,
                                         qkv_bias=True,
                                         drop_rate=0.0,
                                         drop_path_rate=0.1,
                                         ape=False,
                                         patch_norm=True,
                                         use_checkpoint=False)

        if load_params:
            # download from https://github.com/microsoft/Swin-Transformer
            state_dict = torch.load(vision_config['ckpt'], map_location="cpu")['model']

            for k in list(state_dict.keys()):
                if 'relative_position_bias_table' in k:
                    dst_num_pos = (2 * vision_config['window_size'] - 1) ** 2
                    state_dict[k] = interpolate_relative_pos_embed(state_dict[k], dst_num_pos, param_name=k)
                elif ('relative_position_index' in k) or ('attn_mask' in k):
                    del state_dict[k]

    else:  # deit, worse than clip-vit/swin...
        assert config['patch_size'] == 16
        vision_width = 768

        vision_encoder = VisionTransformer(
            img_size=config['image_res'], patch_size=config['patch_size'], embed_dim=768, depth=12, num_heads=12,
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(LayerNormML, eps=1e-6),
            local_attn_depth=4)

        if load_params:
            # download from https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth
            state_dict = torch.load("data/deit_base_patch16_224-b5f2ef4d.pth", map_location="cpu")["model"]
            pos_embed_reshaped = interpolate_pos_embed(state_dict['pos_embed'], num_patches=num_patches,
                                                       num_extra_tokens=1)
            state_dict['pos_embed'] = pos_embed_reshaped

    if load_params:
        print("### Load ViT: ", flush=True)
        msg = vision_encoder.load_state_dict(state_dict, strict=False)
        print("missing_keys: ", msg.missing_keys)
        print("unexpected_keys: ", msg.unexpected_keys)

    return vision_encoder, vision_width


def build_vm_encoder(config, load_params=False):
    """
    Args:
        load_params: False when building fine-tuning models
    """
    num_patches = (config['image_res'] // config['patch_size']) ** 2

    # deit, worse than clip-vit/swin...
    #     assert config['patch_size'] == 16
    vision_width = 768

    vision_encoder = VmTransformer(
        img_size=config['image_res'], patch_size=config['patch_size'], embed_dim=768, depth=4, num_heads=12,
        mlp_ratio=4, qkv_bias=True, norm_layer=partial(LayerNormML, eps=1e-6),
        local_attn_depth=4)

    # if load_params:
    #     # download from https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth
    #     state_dict = torch.load("data/deit_base_patch16_224-b5f2ef4d.pth", map_location="cpu")["model"]
    #     pos_embed_reshaped = im_pos_embed(state_dict['pos_embed'], num_patches=num_patches, num_extra_tokens=1)
    #     state_dict['pos_embed'] = pos_embed_reshaped

    # if load_params:
    #     print("### Load ViT: ", flush=True)
    #     msg = vision_encoder.load_state_dict(state_dict, strict=False)
    #     print("missing_keys: ", msg.missing_keys)
    #     print("unexpected_keys: ", msg.unexpected_keys)

    return vision_encoder, vision_width


def build_text_encoder(config, vision_width, load_text_params=False, use_mlm_loss=False, config_text=None):
    init_params = []  # train from scratch with larger lr

    if config_text is None:
        config_text = RobertaConfig.from_json_file(config['text_config']) \
            if config['use_roberta'] else BertConfig.from_json_file(config['text_config'])

    config_text.encoder_width = vision_width

    if use_mlm_loss:  # for pre-training, load_text_params by default (otherwise notimplemented)
        assert load_text_params is True
        if ('accelerator' in config.keys()) and (config['accelerator']['FP16_OPT_LEVEL'] != 'O0'):
            config_text.fp16 = True  # will use some operations to avoid gradient overflow

        if config['use_roberta']:
            text_encoder, msg = RobertaForMaskedLM.from_pretrained(config['text_encoder'], config=config_text,
                                                                   output_loading_info=True)
        else:
            text_encoder, msg = BertForMaskedLM.from_pretrained(config['text_encoder'], config=config_text,
                                                                output_loading_info=True)

        print("### Load BERT: ")
        for k, v in msg.items():
            print(f"{k}: {sorted(v)}")

        init_params.extend(['text_encoder.' + n for n in msg['missing_keys']])  # of cross attention

        if ('load_bertL_by_sep' in config.keys()) and config['load_bertL_by_sep']:
            state_dict = torch.load(os.path.join(config['text_encoder'], 'pytorch_model.bin'))
            for idx, i_layer in enumerate([13, 15, 17, 19, 21, 23]):
                state_dict_i = {k[22:]: v for k, v in state_dict.items() if f'layer.{i_layer}' in k}
                if config['use_roberta']:
                    msg = text_encoder.roberta.encoder.layer[config_text.fusion_layer + idx].load_state_dict(
                        state_dict_i, strict=False)
                else:
                    msg = text_encoder.bert.encoder.layer[config_text.fusion_layer + idx].load_state_dict(
                        state_dict_i, strict=False)
                print(f"### Load {i_layer} to {config_text.fusion_layer + idx}-layer: {msg}")

    else:  # for fine-tuning, not load_text_params by default
        # assert load_text_params is False

        if config['use_roberta']:
            text_encoder = RobertaModel.from_pretrained(config['text_encoder'], config=config_text,
                                                        add_pooling_layer=False)
        else:
            text_encoder = BertModel.from_pretrained(config['text_encoder'], config=config_text,
                                                     add_pooling_layer=False)

    return text_encoder, init_params


def build_mlp(input_dim, output_dim):
    return nn.Sequential(
        LinearML(input_dim, input_dim * 2),
        LayerNormML(input_dim * 2),
        nn.GELU(),
        LinearML(input_dim * 2, output_dim)
    )


def load_pretrained(ckpt_rpath, config, is_eval=False, load_text=False):
    checkpoint = torch.load(ckpt_rpath, map_location='cpu')
    state_dict = checkpoint['model'] if 'model' in checkpoint.keys() else checkpoint

    if is_eval:
        return state_dict

    num_patches = (config['image_res'] // config['patch_size']) ** 2

    print("### Loading pretrained vision encoder", flush=True)
    if config['use_clip_vit']:
        del state_dict['vision_encoder.position_ids']
        pos_embed_reshaped = interpolate_pos_embed(state_dict['vision_encoder.pos_embed.weight'].unsqueeze(dim=0),
                                                   num_patches=num_patches, num_extra_tokens=1)
        state_dict['vision_encoder.pos_embed.weight'] = pos_embed_reshaped.squeeze(dim=0)

    elif config['use_swin']:

        window_size = read_json(config['vision_config'])['window_size']

        for k in list(state_dict.keys()):
            if 'relative_position_bias_table' in k:
                dst_num_pos = (2 * window_size - 1) ** 2
                state_dict[k] = interpolate_relative_pos_embed(state_dict[k], dst_num_pos, param_name=k)
            elif ('relative_position_index' in k) or ('attn_mask' in k):
                del state_dict[k]

    else:
        pos_embed_reshaped = interpolate_pos_embed(state_dict['vision_encoder.pos_embed'],
                                                   num_patches=num_patches, num_extra_tokens=1)
        state_dict['vision_encoder.pos_embed'] = pos_embed_reshaped

    if load_text:
        print("### Loading pretrained text encoder", flush=True)
        for key in list(state_dict.keys()):
            if 'text_encoder.' in key:
                if config['use_roberta']:
                    if 'roberta.' in key:
                        encoder_key = key.replace('roberta.', '')
                        state_dict[encoder_key] = state_dict[key]
                        del state_dict[key]

                else:
                    if 'bert.' in key:
                        encoder_key = key.replace('bert.', '')
    #                         state_dict[encoder_key] = state_dict[key]
    #                         del state_dict[key]

    return state_dict


class XVLMBase(nn.Module):
    def __init__(self, config=None, load_vision_params=False, load_text_params=False,
                 use_contrastive_loss=False, use_matching_loss=False, use_mlm_loss=False, use_bbox_loss=False,
                 config_text=None):
        super().__init__()
        self.init_params = []  # train from scratch with larger lr

        self.vision_encoder, vision_width = build_vision_encoder(config, load_params=load_vision_params)
        self.vm_encoder, vm_width = build_vm_encoder(config, load_params=load_vision_params)
        self.fc = LinearML(vm_width, vm_width)

        self.text_encoder, init_params = build_text_encoder(config, vision_width=vision_width,
                                                            load_text_params=load_text_params,
                                                            use_mlm_loss=use_mlm_loss,
                                                            config_text=config_text)  # text & cross-modal
        self.init_params.extend(init_params)

        self.vision_width = vision_width
        self.text_width = self.text_encoder.config.hidden_size  # i.e. cross_width
        self.softmax = nn.Softmax(dim=-1)

        self.loss_cons = ContrastiveLoss()

        if use_contrastive_loss:
            self.embed_dim = config['embed_dim']
            self.vision_proj = LinearML(self.vision_width, self.embed_dim)
            self.text_proj = LinearML(self.text_width, self.embed_dim)
            self.init_params.extend(['vision_proj.' + n for n, _ in self.vision_proj.named_parameters()])
            self.init_params.extend(['text_proj.' + n for n, _ in self.text_proj.named_parameters()])

            self.temp = nn.Parameter(torch.ones([]) * config['temp'])
            # self.temp_match = nn.Parameter(torch.ones([]) * config['temp'])
            self.init_params.extend(['temp'])

        if use_matching_loss:
            self.itm_head = build_mlp(input_dim=self.text_width, output_dim=1)
            self.init_params.extend(['itm_head.' + n for n, _ in self.itm_head.named_parameters()])

        if use_bbox_loss:
            self.bbox_head = build_mlp(input_dim=self.text_width, output_dim=4)
            self.init_params.extend(['bbox_head.' + n for n, _ in self.bbox_head.named_parameters()])

    def load_pretrained(self, ckpt_rpath, config, is_eval=False):
        state_dict = load_pretrained(ckpt_rpath, config, is_eval=is_eval, load_text=True)
        msg = self.load_state_dict(state_dict, strict=False)
        print('load checkpoint from %s' % ckpt_rpath)
        print("missing_keys: ", [p for p in msg.missing_keys if 'vision_encoder' not in p])
        print("unexpected_keys: ", msg.unexpected_keys)

    def get_vision_embeds(self, image, image_atts=None, idx_to_group_img=None):
        """
        vision_embeds: cls + patch embeds
        """
        if idx_to_group_img is None:
            image_embeds = self.vision_encoder(image)
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
            return image_embeds, image_atts

        else:
            if image_atts is None:
                image_embeds_fullatts = self.vision_encoder(image)
                image_embeds_fullatts = torch.gather(image_embeds_fullatts, dim=0,
                                                     index=idx_to_group_img.view(-1, 1, 1).expand(
                                                         -1, image_embeds_fullatts.shape[1],
                                                         image_embeds_fullatts.shape[2]))

                image_atts = torch.ones(image_embeds_fullatts.size()[:-1], dtype=torch.long).to(image.device)

                return image_embeds_fullatts, image_atts

            else:
                assert image_atts.size(0) == idx_to_group_img.size(0)  # bsz
                image_embeds, image_embeds_fullatts = \
                    self.vision_encoder(image, idx_to_group_img=idx_to_group_img, image_atts=image_atts)

                image_embeds_fullatts = torch.gather(image_embeds_fullatts, dim=0,
                                                     index=idx_to_group_img.view(-1, 1, 1).expand(
                                                         -1, image_embeds_fullatts.shape[1],
                                                         image_embeds_fullatts.shape[2]))

                return image_embeds, image_atts, image_embeds_fullatts

    def get_text_embeds(self, text_ids, text_atts):
        encoder = self.text_encoder.bert if hasattr(self.text_encoder, 'bert') else self.text_encoder
        return encoder(text_ids, attention_mask=text_atts, return_dict=True, mode='text').last_hidden_state

    def get_cross_embeds(self, image_embeds, image_atts, text_ids=None, text_embeds=None, text_atts=None):
        assert text_atts is not None

        encoder = self.text_encoder.bert if hasattr(self.text_encoder, 'bert') else self.text_encoder

        if text_embeds is not None:
            return encoder(encoder_embeds=text_embeds,
                           attention_mask=text_atts,
                           encoder_hidden_states=image_embeds,
                           encoder_attention_mask=image_atts,
                           return_dict=True,
                           mode='fusion',
                           ).last_hidden_state
        elif text_ids is not None:
            return encoder(text_ids,
                           attention_mask=text_atts,
                           encoder_hidden_states=image_embeds,
                           encoder_attention_mask=image_atts,
                           return_dict=True,
                           ).last_hidden_state
        else:
            raise ValueError

    def get_features(self, image_embeds=None, text_embeds=None, text_embeds_pos=None):
        if image_embeds is None:
            return F.normalize(self.text_proj(text_embeds[:, 0, :]), dim=-1)
        elif text_embeds is None:
            return F.normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1)
        else:
            return F.normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1), \
                F.normalize(self.text_proj(text_embeds[:, 0, :]), dim=-1), \
                F.normalize(self.text_proj(text_embeds_pos[:, 0, :]), dim=-1)

    def get_contrastive_loss(self, image_feat, text_feat, idx=None):
        """
        Args:
            image_feat, text_feat: normalized

        Returns: contrastive loss

        """
        assert image_feat.size(-1) == self.embed_dim
        assert text_feat.size(-1) == self.embed_dim

        image_feat_all = allgather(image_feat, torch.distributed.get_rank(), torch.distributed.get_world_size())
        text_feat_all = allgather(text_feat, torch.distributed.get_rank(), torch.distributed.get_world_size())
        logits = image_feat_all @ text_feat_all.t() / self.temp

        bsz = image_feat_all.shape[0]

        if idx is None:
            labels = torch.arange(bsz, device=image_feat.device)
            loss_i2t = F.cross_entropy(logits, labels)
            loss_t2i = F.cross_entropy(logits.t(), labels)

        else:
            idx = idx.view(-1, 1)
            assert idx.size(0) == image_feat.size(0)
            idx_all = allgather(idx, torch.distributed.get_rank(), torch.distributed.get_world_size())
            pos_idx = torch.eq(idx_all, idx_all.t()).float()
            labels = pos_idx / pos_idx.sum(1, keepdim=True)

            loss_i2t = -torch.sum(F.log_softmax(logits, dim=1) * labels, dim=1).mean()
            loss_t2i = -torch.sum(F.log_softmax(logits.t(), dim=1) * labels, dim=1).mean()

        return (loss_i2t + loss_t2i) / 2

    def get_matching_loss(self, image_embeds, image_atts, image_feat, text_embeds, text_atts, text_feat,
                          text_embeds_pos, text_atts_pos, text_feat_pos, idx=None, single_text_embed=None):
        """
        Matching Loss with hard negatives
        """

        bs = image_embeds.size(0)
        negative_sample = 2

        with torch.no_grad():
            sim_i2t = image_feat @ text_feat.t() / self.temp
            sim_t2i = text_feat @ image_feat.t() / self.temp
            weights_i2t = F.softmax(sim_i2t, dim=1) + 1e-5
            weights_t2i = F.softmax(sim_t2i, dim=1) + 1e-5
            weights_i2t.fill_diagonal_(0)
            weights_t2i.fill_diagonal_(0)

        sims_t2i_fusion = torch.full((bs, bs), 0.0).to(image_embeds.device)
        # sims_t2i_fusion = torch.zeros((bs,bs)).to(image_embeds.device)
        sims_t2i_hard = torch.zeros((bs, negative_sample + 1)).to(image_embeds.device)
        labels_t2i = torch.arange(bs, device=image_feat.device)
        cross_pos = self.get_cross_embeds(image_embeds, image_atts, text_embeds=text_embeds, text_atts=text_atts)[:, 0,
                    :]
        output_pos = self.itm_head(cross_pos)
        output_pos = torch.squeeze(output_pos)
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2i[b], negative_sample)  # .item()
            input_text_embeds = text_embeds[b].repeat(negative_sample, 1, 1)
            input_text_atts = text_atts[b].repeat(negative_sample, 1)
            cross_embed = self.get_cross_embeds(image_embeds[neg_idx], image_atts[neg_idx],
                                                text_embeds=input_text_embeds,
                                                text_atts=input_text_atts)[:, 0, :]
            output = self.itm_head(cross_embed)
            output = torch.squeeze(output)
            sims_t2i_fusion[b, neg_idx] = output
            sims_t2i_fusion[b, b] = output_pos[b]
            # num = 0
            # for index in range(len(sims_t2i_fusion[b])):
            #     if sims_t2i_fusion[b, index] > -10:
            #         sims_t2i_hard[b,num] = sims_t2i_fusion[b, index]
            #         if index == b:
            #             labels_t2i[b] = num
            #         num += 1

        sims_i2t_fusion = torch.full((bs, bs), 0.0).to(image_embeds.device)
        sims_i2t_hard = torch.zeros((bs, negative_sample + 1)).to(image_embeds.device)
        labels_i2t = torch.arange(bs, device=image_feat.device)
        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2t[b], negative_sample)  # .item()
            input_image_embeds = image_embeds[b].repeat(negative_sample, 1, 1)
            input_image_atts = image_atts[b].repeat(negative_sample, 1)
            cross_embed = self.get_cross_embeds(input_image_embeds, input_image_atts,
                                                text_embeds=text_embeds[neg_idx],
                                                text_atts=text_atts[neg_idx])[:, 0, :]
            output = self.itm_head(cross_embed)
            output = torch.squeeze(output)
            sims_i2t_fusion[b, neg_idx] = output
            sims_i2t_fusion[b, b] = output_pos[b]
            # num = 0
            # for index in range(len(sims_i2t_fusion[b])):
            #     if sims_i2t_fusion[b, index] > -10:
            #         sims_i2t_hard[b,num] = sims_i2t_fusion[b, index]
            #         if index == b:
            #             labels_i2t[b] = num
            #         num += 1
        # print("sims_t2i_fusion : ", sims_t2i_hard)
        loss_t2i = F.cross_entropy(sims_t2i_fusion, labels_t2i)
        loss_i2t = F.cross_entropy(sims_i2t_fusion, labels_i2t)

        # # ====================cal relation KD========================
        fusion_text_embed = l2norm(cross_pos, dim=-1) + 10 ** -6
        single_text_embed = l2norm(single_text_embed, dim=-1) + 10 ** -6
        text_fusion_sims = torch.ones((bs, bs)).to(image_embeds.device)
        text_single_sims = torch.ones((bs, bs)).to(image_embeds.device)
        for i in range(bs):
            for j in range(i + 1, bs):
                text_fusion_sims[i][j] = torch.sqrt(torch.sum((fusion_text_embed[i] - fusion_text_embed[j]) ** 2))
                text_fusion_sims[j][i] = text_fusion_sims[i][j]
                text_single_sims[i][j] = torch.sqrt(torch.sum((single_text_embed[i] - single_text_embed[j]) ** 2))
                text_single_sims[j][i] = text_single_sims[i][j]
        text_fusion_sims = text_fusion_sims.reshape(-1)
        text_single_sims = text_single_sims.reshape(-1)
        text_fusion_sims = self.softmax(text_fusion_sims) + 10 ** -6
        text_single_sims = self.softmax(text_single_sims) + 10 ** -6
        loss_text_KD = torch.sum(text_single_sims * torch.log(text_single_sims / text_fusion_sims)) / bs
        #
        # ====================cal relation KD========================
        cross_text_cons = self.get_cross_embeds(image_embeds, image_atts,
                                                text_embeds=text_embeds_pos, text_atts=text_atts_pos)[:, 0, :]
        loss_c, loss_regu = self.loss_cons(cross_pos, cross_text_cons)
        loss_regu = loss_regu * 0.001
        loss_text_cons = loss_c + loss_regu

        #         #====================cal fusion text itm========================
        #         sims_t2i_fusion_asym = torch.full((bs, bs), 0.0).to(image_embeds.device)
        #         labels_t2i_asym = torch.arange(bs, device=image_feat.device)
        #         cross_pos_asym = self.vm_encoder(image_embeds, text_embeds=text_embeds,
        #                                          text_atts=text_atts,  image_atts=image_atts)
        #         output_pos_asym = self.itm_head(cross_pos_asym)
        #         output_pos_asym = torch.squeeze(output_pos_asym)
        #         for b in range(bs):
        #             neg_idx = torch.multinomial(weights_t2i[b], negative_sample)#.item()
        #             input_text_embeds=text_embeds[b].repeat(negative_sample,1,1)
        #             input_text_atts=text_atts[b].repeat(negative_sample,1)
        #             cross_embed_asym = self.vm_encoder(image_embeds[neg_idx], text_embeds=input_text_embeds,
        #                                               text_atts=input_text_atts, image_atts=image_atts[neg_idx])
        #             output_asym = self.itm_head(cross_embed_asym)
        #             output_asym = torch.squeeze(output_asym)
        #             sims_t2i_fusion_asym[b, neg_idx] = output_asym
        #             sims_t2i_fusion_asym[b, b] = output_pos_asym[b]

        #         sims_i2t_fusion_asym = torch.full((bs, bs), 0.0).to(image_embeds.device)
        #         labels_i2t_asym = torch.arange(bs, device=image_feat.device)
        #         for b in range(bs):
        #             neg_idx = torch.multinomial(weights_i2t[b], negative_sample)#.item()
        #             input_image_embeds=image_embeds[b].repeat(negative_sample,1,1)
        #             input_image_atts=image_atts[b].repeat(negative_sample,1)
        #             cross_embed_asym = self.vm_encoder(input_image_embeds, text_embeds=text_embeds[neg_idx],
        #                                               text_atts=text_atts[neg_idx], image_atts=input_image_atts)
        #             output_asym = self.itm_head(cross_embed_asym)
        #             output_asym = torch.squeeze(output_asym)
        #             sims_i2t_fusion_asym[b, neg_idx] = output_asym
        #             sims_i2t_fusion_asym[b, b] = output_pos_asym[b]

        #         loss_t2i_asym = F.cross_entropy(sims_t2i_fusion_asym , labels_t2i_asym)
        #         loss_i2t_asym = F.cross_entropy(sims_i2t_fusion_asym , labels_i2t_asym)

        return (loss_t2i + loss_i2t), loss_text_cons+loss_text_KD

    def get_mlm_loss(self, text_ids_masked, text_atts, image_embeds, image_atts, masked_pos, masked_ids):
        return self.text_encoder(text_ids_masked,
                                 attention_mask=text_atts,
                                 encoder_hidden_states=image_embeds,
                                 encoder_attention_mask=image_atts,
                                 return_dict=True,
                                 labels=masked_ids,
                                 masked_pos=masked_pos).loss

    def get_mpr_loss(self, image_embeds, image_atts, text_embeds, text_atts, mpr_ids):
        bs = image_embeds.size(0)
        cross_pos_asym = self.vm_encoder(image_embeds, text_embeds=text_embeds, text_atts=text_atts,
                                         image_atts=image_atts)
        cross_pos_asym = self.fc(cross_pos_asym)
        mpr_matric = torch.full((image_embeds.size(0), image_embeds.size(1), image_embeds.size(2)), 1.0)
        for b in range(bs):
            mpr_matric[b][mpr_ids[b]] = mpr_matric[b][mpr_ids[b]] - 1
        mpr_matric = mpr_matric.to(image_embeds.device)
        mpr_embed = image_embeds * mpr_matric
        mpr_pos_asym = self.vm_encoder(mpr_embed, text_embeds=text_embeds, text_atts=text_atts)
        mpr_pos_asym = self.fc(mpr_pos_asym)
        loss_mpr = 0.0
        for b in range(bs):
            loss_mpr += F.mse_loss(mpr_pos_asym[b][mpr_ids[b]], cross_pos_asym[b][mpr_ids[b]])

        return loss_mpr
