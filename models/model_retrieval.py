import torch
from models import XVLMBase, load_pretrained


class XVLM(XVLMBase):
    def __init__(self, config):
        super().__init__(config, load_vision_params=True, load_text_params=True,
                         use_contrastive_loss=True, use_matching_loss=True, use_mlm_loss=True, use_bbox_loss=False)

        self.num_attention_heads = self.text_encoder.config.num_attention_heads
        self.init_params = []

    def load_pretrained(self, ckpt_rpath, config, is_eval=False):
        state_dict = load_pretrained(ckpt_rpath, config, is_eval=is_eval, load_text=True)
        msg = self.load_state_dict(state_dict, strict=False)
        print('load checkpoint from %s' % ckpt_rpath)
        print("missing_keys: ", [p for p in msg.missing_keys if 'vision_encoder' not in p])
        print("unexpected_keys: ", msg.unexpected_keys)

    def forward(self, image, text_ids, text_atts, text_ids_pos, text_atts_pos,
                text_atts_mask=None, text_ids_masked=None, masked_pos=None, masked_ids=None,
                idx=None, mpr_ids=None, single_text_embed=None):
        image_embeds, image_atts = self.get_vision_embeds(image)
        text_embeds = self.get_text_embeds(text_ids, text_atts)
        text_embeds_pos = self.get_text_embeds(text_ids_pos, text_atts_pos)

        image_feat, text_feat, text_feat_pos = self.get_features(image_embeds, text_embeds, text_embeds_pos)
        loss_itc = self.get_contrastive_loss(image_feat, text_feat, idx=idx)
        loss_itm, loss_cons = self.get_matching_loss(image_embeds, image_atts, image_feat, text_embeds, text_atts,
                                                     text_feat, text_embeds_pos, text_atts_pos,
                                                     text_feat_pos, idx=idx, single_text_embed=single_text_embed)

        loss_mlm = self.get_mlm_loss(text_ids_masked, text_atts_mask, image_embeds, image_atts, masked_pos, masked_ids)
        loss_mpr = self.get_mpr_loss(image_embeds, image_atts, text_embeds, text_atts, mpr_ids)

        return loss_itc, loss_itm, loss_cons * 10, loss_mlm * 0.4, loss_mpr * 10.0
