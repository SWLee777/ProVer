import os
import torch
import pickle
import numpy as np
import pandas as pd
import cn_clip.clip as clip
import jieba.posseg as pseg
from transformers import BertTokenizer
from transformers import AutoTokenizer
from torch.utils.data import TensorDataset, DataLoader


def process_content(content: np.ndarray, max_nouns: int = 30):
    processed = []
    for text in content:
        # 提取名词
        nouns = [word for word, flag in pseg.cut(text) if flag.startswith('n')][:max_nouns]
        new_text = f"{text} {' '.join(nouns)}"
        processed.append(new_text)

    return processed

def word2input(texts,vocab_file,max_len):
    tokenizer = BertTokenizer(vocab_file=vocab_file)
    token_ids =[]
    for i,text in enumerate(texts):
        token_ids.append(tokenizer.encode(text, max_length=max_len, add_special_tokens=True, padding='max_length',
                             truncation=True))
    token_ids = torch.tensor(token_ids)
    masks = torch.zeros(token_ids.size())
    for i,token in enumerate(token_ids):
        masks[i] = (token != 0)
    return token_ids,masks


def _init_fn(worker_id):
    np.random.seed(2024)

class weibo_data():
    def __init__(self,max_len, batch_size, model_name,vocab_file, num_workers):
        self.max_len = max_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.model_name = model_name
        self.vocab_file = vocab_file
        # self.category_dict = category_dict
#content,label,category,{}_losder.pkl,{}_clip_loader.pkl
    def load_jsonl_to_tensor(self,path):
        df = pd.read_json(path, lines=True, encoding='utf-8')
        content = df['text'].astype('str').to_numpy()  # 确保是字符串
        label = torch.tensor(df['final'].astype(int).to_numpy())  # 转换为 int 再转 tensor
        return content, label
    def load_data(self,path,imagepath,clipimagepath,shuffle):#(json)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)  # 根据模型名称匹配分词器。
        content, label = self.load_jsonl_to_tensor(path)
        token_ids, masks = word2input(content, self.vocab_file, self.max_len)
        ordered_image = pickle.load(open(imagepath,'rb'))
        clip_image = pickle.load(open(clipimagepath, 'rb'))
        # print(len(clip_image))
        # print(clip_image[0].size())
        print(type(clip_image))
        print(clip_image.size())
        clip_text = clip.tokenize(content)

        encoded_batch = tokenizer.batch_encode_plus(
            content.tolist(),
            return_tensors="pt",  # 返回pytorch张量
            max_length=self.max_len,
            padding='max_length',  # 填充为512
            truncation=True,  # 超出就截断
        )
        input_ids = encoded_batch['input_ids']
        attention_mask = encoded_batch['attention_mask']

        datasets =TensorDataset(
            input_ids,
            attention_mask,
            label,
            ordered_image,
            clip_image,
            clip_text,
            token_ids,
            masks,
        )
        dataloader = DataLoader(
            dataset = datasets,
            batch_size = self.batch_size,
            num_workers = self.num_workers,
            pin_memory = True,#内存优化技术
            shuffle = shuffle,
            worker_init_fn = _init_fn,
        )
        return dataloader

def clipdata2gpu(batch):
    batch_data = {
        'input_ids': batch[0].cuda(),
        'attention_mask': batch[1].cuda(),
        'label': batch[2].cuda(),
        'image':batch[3].cuda(),
        'clip_image':batch[4].cuda(),
        'clip_text': batch[5].cuda(),
        'token_ids':batch[6].cuda(),
        'masks':batch[7].cuda(),
    }
    return batch_data



class Averager():
    def __init__(self):
        self.n = 0
        self.v = 0
    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1
    def item(self):
        return self.v