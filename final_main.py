import os
import torch
import random
import argparse
import numpy as np
from final_run import Run

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def MMDFND():
        parser = argparse.ArgumentParser()
        parser.add_argument('--path',default='./data/')
        parser.add_argument('--max_len', type=int, default=224)
        parser.add_argument('--batch_size', type=int, default=16)
        parser.add_argument('--vocab_file',default='./pretrained_model/chinese_roberta_wwm_base_ext_pytorch/vocab.txt')
        # parser.add_argument('--vocab_file', default='princeton-nlp/unsup-simcse-roberta-base')
        parser.add_argument('--num_workers', type=int, default=0)
        parser.add_argument('--model_name', default='princeton-nlp/unsup-simcse-roberta-base')
        parser.add_argument('--bert', default='./pretrained_model/chinese_roberta_wwm_base_ext_pytorch')
        parser.add_argument('--mlp_dims', type=int,default=128)
        parser.add_argument('--emb_dim', type=int, default=768)
        parser.add_argument('--dropout', type=float, default=0.2)
        parser.add_argument('--save_dir', default= './runs')
        parser.add_argument('--name',default='MMDFND')
        parser.add_argument("--lr", type=float, default=3e-4, help="learning rate")
        parser.add_argument("--beta1", type=float, default=0.9, help="beta1")
        parser.add_argument("--beta2", type=float, default=0.98, help="beta2")#0.98
        parser.add_argument("--weight_decay", type=float, default=0.001, help="weight decay")#0.0001
        parser.add_argument("--eps", type=float, default=1e-6, help="eps")
        parser.add_argument("--warmup_steps", type=int, default=1000, help="Warmup steps")
        parser.add_argument('--total_epoch', type=int, default=1)
        parser.add_argument('--gpu', default='0')
        parser.add_argument('--seed', type=int, default=3074)
        args = parser.parse_args()
        return args


args = MMDFND()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.enabled = True

if __name__ == '__main__':
    Run(args = args).main()
