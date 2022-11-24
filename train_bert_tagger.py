from bert_mrc_pytorch.bert_mrc_global_pointer import BertGlobalPointer
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertConfig
from bert_mrc_pytorch.bert_tagger import BertNER, BertTagger
import json
import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as F
from einops import rearrange


tokenizer = BertTokenizer.from_pretrained(r'D:\PekingInfoResearch\pretrain_models\bert-base-chinese')
maxlen = 300
BATCH_SIZE = 4
categories = set()
def load_data(filename):
    """加载数据
    单条格式：[text, (start, end, label), (start, end, label), ...]，
              意味着text[start:end + 1]是类型为label的实体。
    """
    D = []
    d = {'text': [], 'label': []}
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            if line == '\n':
                D.append(d)
                d = {'text': [], 'label': []}
                continue
            line = line.split()
            d['text'].append(line[0])
            d['label'].append(line[1])
            categories.add(line[1])
    return D


train_data = load_data(r'D:\open_data\ner\zh_msra\train.char.bmes')
valid_data = load_data(r'D:\open_data\ner\zh_msra\dev.char.bmes')
categories = list(sorted(categories))

class TextSamplerDataset(Dataset):
    def __init__(self, data, seq_len, mode='train'):
        super().__init__()
        self.data = data
        self.seq_len = seq_len
        self.mode = mode

    def __getitem__(self, index):
        d = self.data[index]
        token_ids = tokenizer.encode(d['text'])
        segment_ids = self.pad([0] * len(token_ids))
        mask = self.pad([1] * len(token_ids))
        token_ids = self.pad(token_ids)
        labels = self.pad([categories.index(x)+1 for x in d['label']])
        label_mask = self.pad([1 if x != 'O' else 0 for x in d['label']])

        return token_ids, segment_ids, mask, labels, label_mask

    def pad(self, lst, value=0, max_length=None):
        max_length = max_length or self.seq_len
        if len(lst) > max_length:
            truncate_len = len(lst) - max_length
            lst = lst[:-truncate_len-1] + lst[-1:]
        else:
            add_len = max_length - len(lst)
            lst = lst + [value] * add_len
        return lst

    def __len__(self):
        return len(self.data)


def collate_fn(batch):
    token_ids, segment_ids, mask, labels, label_mask = list(zip(*batch))

    token_ids = torch.LongTensor(token_ids)
    segment_ids = torch.LongTensor(segment_ids)
    labels = torch.LongTensor(labels)
    mask = torch.FloatTensor(mask)
    label_mask = torch.FloatTensor(label_mask)

    return token_ids, segment_ids, mask, labels, label_mask

train_dataset = TextSamplerDataset(train_data, maxlen)
valid_dataset = TextSamplerDataset(valid_data, maxlen)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

model = BertNER.from_pretrained(r'D:\PekingInfoResearch\pretrain_models\bert-base-chinese',
                                                      num_labels=len(categories)+1)
# model = BertTagger()
optim = torch.optim.Adam(model.parameters(), lr=2e-5)

def calculate_acc(y_true, y_pred, label_mask):
    """给GlobalPointer设计的F1
    """
    y_pred = torch.argmax(y_pred, dim=-1)
    acc = ((y_true == y_pred) * label_mask).sum() / torch.sum(label_mask).clamp(min=1e-9)
    return acc


for _ in range(100):
    model.cuda()
    model.train()
    pbar = tqdm(train_loader)
    for batch in pbar:
        batch = [x.cuda() for x in batch]
        outputs = model(batch[0], attention_mask=batch[2], token_type_ids=batch[1], labels=batch[3])
        loss = outputs.loss
        acc = calculate_acc(batch[3], outputs.logits, batch[4])
        loss.backward()
        optim.step()
        optim.zero_grad()
        pbar.set_description(f"loss {loss} acc {acc}")



