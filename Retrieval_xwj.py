import argparse
import datetime
import json
from pathlib import Path
import os

import math
import yaml
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.functional as F

import utils
from dataset import create_dataset, create_sampler, create_loader
from embed.NDCG_coco import *
from models.model_retrieval import XVLM
from models.tokenization_bert import BertTokenizer
from models.tokenization_roberta import RobertaTokenizer
from optim import create_optimizer
from scheduler import create_scheduler


@torch.no_grad()
def evaluation(model, data_loader, tokenizer, device, config):
    model.eval()

    # text1 = ["A basket ball player is posing in front of a basket.", "A person with a basketball stands in front of a goal.", "A giraffe is eating out of a basket.", "A tiger lay laying on a blanket in a cage."]
    # text2 = ["A woman standing over a pan filled with food in a kitchen.", "A woman wearing shorts bends over a frying pan on a stove.", "A woman is cooking on a large black stove.", "A woman cooking on an old fashioned stove."]
    # text3 = ["A young woman holding a giant tennis racquet.", "The girl is smiling, holding a gigantic tennis racket.", "A young girl is holding a tennis racket.", "A woman is holding a tennis racket and a towel."]

    text1 = ["A basket ball player is posing in front of a basket.", "A person with a basketball stands in front of a goal.", "A uniformed boy is holding a basketball with his back to the hoop.", "A basketball player holds a basketball for a picture."]
    text2 = ["A woman standing over a pan filled with food in a kitchen.", "a person standing in front of a counter top and a tall pile of food.", "A woman smiling while she prepares a plate of food.", "a person next to a horse on a beach."]
    text3 = ["A young woman holding a giant tennis racquet.", "A pretty young woman holding a giant tennis racquet.", "A girl is holding a novelty tennis racket in a very large size.", "The girl is smiling, holding a gigantic tennis racket."]


    text_input = tokenizer(text1, padding='max_length', truncation=True, max_length=config['max_tokens'],
                        return_tensors="pt").to(device)
    text_output = model.text_encoder.bert(text_input.input_ids, attention_mask=text_input.attention_mask,
                                            mode='text')
    text_feat = text_output.last_hidden_state
    print("---------example1")
    for i in range(1, 4):
        mse768 = torch.mean((text_feat[0] - text_feat[i]) ** 2)
        print("768", mse768)


    text_input = tokenizer(text2, padding='max_length', truncation=True, max_length=config['max_tokens'],
                        return_tensors="pt").to(device)
    text_output = model.text_encoder.bert(text_input.input_ids, attention_mask=text_input.attention_mask,
                                            mode='text')
    text_feat = text_output.last_hidden_state
    print("---------example2")
    for i in range(1, 4):
        mse768 = torch.mean((text_feat[0] - text_feat[i]) ** 2)
        print("768", mse768)


    text_input = tokenizer(text3, padding='max_length', truncation=True, max_length=config['max_tokens'],
                        return_tensors="pt").to(device)
    text_output = model.text_encoder.bert(text_input.input_ids, attention_mask=text_input.attention_mask,
                                            mode='text')
    text_feat = text_output.last_hidden_state
    print("---------example3")
    for i in range(1, 4):
        mse768 = torch.mean((text_feat[0] - text_feat[i]) ** 2)
        print("768", mse768)

    exit()



def main(args, config):
    utils.init_distributed_mode(args)
    device = torch.device(args.device)

    world_size = utils.get_world_size()

    if args.bs > 0:
        config['batch_size_train'] = args.bs // world_size

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    train_dataset, val_dataset, test_dataset = create_dataset('re', config)


    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        samplers = create_sampler([train_dataset], [True], num_tasks, global_rank) + [None, None]
    else:
        samplers = [None, None, None]

    train_loader, val_loader, test_loader = create_loader([train_dataset, val_dataset, test_dataset], samplers,
                                                          batch_size=[config['batch_size_train']] + [
                                                              config['batch_size_test']] * 2,
                                                          num_workers=[8, 4, 4],
                                                          is_trains=[True, False, False],
                                                          collate_fns=[None, None, None])

    model = XVLM(config=config)

    # model.load_pretrained("output/coco-distill/checkpoint_best.pth", config, is_eval=args.evaluate)     # UMT
    model.load_pretrained("output_new/colvr-coco/checkpoint_best.pth", config, is_eval=args.evaluate)   # CoLVR

    model = model.to(device)
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    if config['use_roberta']:
        tokenizer = RobertaTokenizer.from_pretrained(config['text_encoder'])
    else:
        tokenizer = BertTokenizer.from_pretrained(config['text_encoder'])


    evaluation(model_without_ddp, test_loader, tokenizer, device, config)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--config', type=str, default='configs/Retrieval_flickr.yaml')
    parser.add_argument('--output_dir', type=str,
                        default="output/flickr_CE_contras_10_0.4_10")  # this script works for both mscoco and flickr30k
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='tcp://127.0.0.1:5222', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, action='store_false')

    parser.add_argument('--bs', default=4, type=int, help="for each gpu, batch_size = bs // num_gpus")
    parser.add_argument('--evaluate', action='store_true')

    args = parser.parse_args()
    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))
    main(args, config)


'''
CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --nproc_per_node=1 Retrieval_xwj.py --output_dir="output/tmp" --dataname="coco" --evaluate


'''