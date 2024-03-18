import collections

import json
import os
import numpy as np
import copy
import torch
import random

from random import randint, shuffle
from random import random as rand

from torch.utils.data import Dataset
from transformers import BertTokenizer, RobertaTokenizer

from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

from dataset.utils import pre_caption


class TextMaskingGenerator:
    def __init__(self, tokenizer, mask_prob, mask_max, skipgram_prb=0.2, skipgram_size=3, mask_whole_word=True,
                 use_roberta=False):
        self.id2token = {i: w for w, i in tokenizer.get_vocab().items()}
        print("len(tokenizer.id2token), ", len(self.id2token), flush=True)

        self.use_roberta = use_roberta

        for i in range(len(self.id2token)):
            assert i in self.id2token.keys()  # check

        self.cls_token = tokenizer.cls_token
        self.mask_token = tokenizer.mask_token
        print("mask_generator.cls_token, ", self.cls_token, flush=True)
        print("mask_generator.mask_token, ", self.mask_token, flush=True)

        self.mask_max = mask_max
        self.mask_prob = mask_prob

        self.skipgram_prb = skipgram_prb
        self.skipgram_size = skipgram_size
        self.mask_whole_word = mask_whole_word

    def get_random_word(self):
        i = randint(0, len(self.id2token) - 1)
        return self.id2token[i]

    def __call__(self, tokens: list):  # tokens: [CLS] + ...
        n_pred = min(self.mask_max, max(
            1, int(round(len(tokens) * self.mask_prob))))

        # candidate positions of masked tokens
        assert tokens[0] == self.cls_token
        special_pos = set([0])  # will not be masked
        cand_pos = list(range(1, len(tokens)))

        shuffle(cand_pos)
        masked_pos = set()
        max_cand_pos = max(cand_pos)
        for pos in cand_pos:
            if len(masked_pos) >= n_pred:
                break
            if pos in masked_pos:
                continue

            def _expand_whole_word(st, end):
                new_st, new_end = st, end

                if self.use_roberta:
                    while (new_st > 1) and (tokens[new_st][0] != 'Ġ'):
                        new_st -= 1
                    while (new_end < len(tokens)) and (tokens[new_end][0] != 'Ġ'):
                        new_end += 1
                else:
                    # bert, WordPiece
                    while (new_st >= 0) and tokens[new_st].startswith('##'):
                        new_st -= 1
                    while (new_end < len(tokens)) and tokens[new_end].startswith('##'):
                        new_end += 1

                return new_st, new_end

            if (self.skipgram_prb > 0) and (self.skipgram_size >= 2) and (rand() < self.skipgram_prb):
                # ngram
                cur_skipgram_size = randint(2, self.skipgram_size)
                if self.mask_whole_word:
                    st_pos, end_pos = _expand_whole_word(
                        pos, pos + cur_skipgram_size)
                else:
                    st_pos, end_pos = pos, pos + cur_skipgram_size
            else:
                if self.mask_whole_word:
                    st_pos, end_pos = _expand_whole_word(pos, pos + 1)
                else:
                    st_pos, end_pos = pos, pos + 1

            for mp in range(st_pos, end_pos):
                if (0 < mp <= max_cand_pos) and (mp not in special_pos):
                    masked_pos.add(mp)
                else:
                    break

        masked_pos = list(masked_pos)
        n_real_pred = len(masked_pos)
        if n_real_pred > n_pred:
            shuffle(masked_pos)
            masked_pos = masked_pos[:n_pred]

        for pos in masked_pos:
            if rand() < 0.8:  # 80%
                tokens[pos] = self.mask_token
            elif rand() < 0.5:  # 10%
                tokens[pos] = self.get_random_word()

        return tokens, masked_pos


class re_train_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=60):

        self.ann = []
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.img_ids = {}
        self.train_text_embed = np.load('/opt/data/private/wuxy/data/f30k_precomp/regu/train_text_embed_regu.npy')

        if len(ann_file) > 1:
            ann_tmp = json.load(open(ann_file[0], 'r'))
            id2num = collections.defaultdict(int)
            for d in ann_tmp:
                id2num[d['image']] += 1
                if id2num[d['image']] > 5:
                    continue
                self.ann.append(d)

            del ann_tmp, id2num

            self.val_ann = json.load(open(ann_file[1], 'r'))
            val_st_id = 1000000001
            for val in self.val_ann:
                caps = val['caption']
                for c in caps[:5]:
                    self.ann.append({
                        'caption': c,
                        'image': val['image'],
                        'image_id': val_st_id + 1
                    })
                val_st_id + 1
        else:
            for f in ann_file:
                self.ann += json.load(open(f, 'r'))

        n = 0
        for ann in self.ann:
            img_id = ann['image_id']
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1

        # self.text_embed = np.load(os.path.join(self.image_root,'regu/train_text_embed_regu_coco_0.npy'))
        # print("data : ", len(self.ann), self.text_embed.shape)

        # ===============mask=================
        self.add_eos = False
        self.tokenized = False
        self.tokenizer = BertTokenizer.from_pretrained('data/bert-base-uncased')
        self.cls_token = self.tokenizer.cls_token
        self.eos_token = self.tokenizer.sep_token
        self.pad_token_id = self.tokenizer.pad_token_id
        self.mask_token_id = self.tokenizer.mask_token_id

        print("dataset.cls_token, ", self.cls_token, flush=True)
        print("dataset.eos_token, ", self.eos_token, flush=True)
        print("dataset.pad_token_id, ", self.pad_token_id, flush=True)
        print("dataset.mask_token_id, ", self.mask_token_id, flush=True)

        #         self.mask_generator = TextMaskingGenerator(self.tokenizer, config['mask_prob'],
        #                                                    config['max_masks'], config['skipgram_prb'],
        #                                                    config['skipgram_size'], config['mask_whole_word'])
        self.mask_generator = TextMaskingGenerator(self.tokenizer, 0.25, 6, 0.2, 3, True)

        self.PAD_mask = -100  # loss will ignore this
        self.max_tokens = 60  # config['max_tokens']
        self.max_masks = 6  # config['max_masks']

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):

        ann = self.ann[index]
        # print(self.image_root)

        single_text_embed = self.train_text_embed[0]

        image_path = os.path.join(self.image_root, ann['image'])
        image = Image.open(image_path).convert('RGB')
        # image = Image.open('/test/whc/').convert('RGB')

        image = self.transform(image)

        caption = pre_caption(ann['caption'], self.max_words)
        #         single_text_embed = self.text_embed[index]

        # ===============positive text=================
        rand = random.randint(1, 4)
        if index % 5 == 0 and index + rand < len(self.ann) - 1:
            captions_pos = pre_caption(self.ann[index + rand]['caption'], self.max_words)
        else:
            captions_pos = pre_caption(self.ann[index - rand]['caption'], self.max_words)

        # ===============mask=================
        text = ann['caption']
        text = pre_caption(text, self.max_words)  # be careful, if text is '', it will cause error
        tokens = self.tokenizer.tokenize(text)

        tokens = [self.cls_token] + tokens[:self.max_tokens - 1]

        if self.add_eos:
            tokens = tokens[:self.max_tokens - 1]
            tokens += [self.eos_token]

        n_tokens = len(tokens)
        assert n_tokens >= 2, "len(word tokens) < 2"

        text_ids = self.tokenizer.convert_tokens_to_ids(tokens)  # list of int

        tokens_masked, masked_pos = self.mask_generator(copy.deepcopy(tokens))
        text_ids_masked = self.tokenizer.convert_tokens_to_ids(tokens_masked)  # list of int
        masked_ids = [text_ids[p] for p in masked_pos]

        # pad
        n_pad = self.max_tokens - n_tokens
        text_ids = text_ids + [self.pad_token_id] * n_pad
        text_atts = [1] * n_tokens + [0] * n_pad

        text_ids_masked = text_ids_masked + [self.pad_token_id] * n_pad
        n_pad = self.max_masks - len(masked_ids)
        masked_pos = masked_pos + [0] * n_pad
        masked_ids = masked_ids + [self.PAD_mask] * n_pad

        text_ids_masked = torch.tensor(text_ids_masked, dtype=torch.long)
        text_atts = torch.tensor(text_atts, dtype=torch.long)
        masked_pos = torch.tensor(masked_pos, dtype=torch.long)
        masked_ids = torch.tensor(masked_ids, dtype=torch.long)

        # ===============mask mpr=================
        mpr_ids = []
        while (len(mpr_ids) < 10):
            index = random.randint(1, 143)
            if index not in mpr_ids:
                mpr_ids.append(index)
        mpr_ids = torch.tensor(mpr_ids, dtype=torch.long)

        return image, caption, self.img_ids[ann[
            'image_id']], captions_pos, text_atts, text_ids_masked, masked_pos, masked_ids, mpr_ids, single_text_embed


class re_eval_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=60):
        self.ann = json.load(open(ann_file, 'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words

        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}

        txt_id = 0
        for img_id, ann in enumerate(self.ann):
            self.image.append(ann['image'])
            self.img2txt[img_id] = []
            for i, caption in enumerate(ann['caption'][:5]):
                self.text.append(pre_caption(caption, self.max_words))
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id] = img_id
                txt_id += 1
        print("data : ", len(self.text))

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):

        image_path = os.path.join(self.image_root, self.ann[index]['image'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        return image, index
