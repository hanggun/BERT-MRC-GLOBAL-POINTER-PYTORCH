from bert_mrc_pytorch.bert_mrc_global_pointer import BertGlobalPointer, BertModel
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizerFast, BertConfig
import json
import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as F
from einops import rearrange
from data_loader import DataMaker
import argparse

torch.manual_seed(2333)  # pytorch random seed
torch.backends.cudnn.deterministic = True
tokenizer = BertTokenizerFast.from_pretrained(r'D:\PekingInfoResearch\pretrain_models\bert-base-chinese')
maxlen = 256
BATCH_SIZE = 4
categories = set()
def load_data(filename):
    """加载数据
    单条格式：[text, (start, end, label), (start, end, label), ...]，
              意味着text[start:end + 1]是类型为label的实体。
    """
    D = []
    for d in json.load(open(filename, encoding='utf8')):
        D.append([d['text']])
        for e in d['entities']:
            start, end, label = e['start_idx'], e['end_idx'], e['type']
            if start <= end:
                D[-1].append((start, end+1, label))
            categories.add(label)
    return D


def load_bid_data(filename):
    """加载数据
    单条格式：[text, (start, end, label), (start, end, label), ...]，
              意味着text[start:end + 1]是类型为label的实体。
    """
    D = []
    with open(filename, encoding='utf-8') as f:
        f = f.read()
        for l in f.split('\n \n'):
            if not l:
                continue
            d = ['']
            for i, c in enumerate(l.split('\n')):
                char, flag = c.split('\t')
                d[0] += char
                if flag[0] == 'B':
                    d.append([i, i, flag[2:]])
                    categories.add(flag[2:])
                elif flag[0] == 'E':
                    d[-1][1] = i+1
            D.append(d)
    return D


def load_people_data(filename):
    """加载数据
    单条格式：[text, (start, end, label), (start, end, label), ...]，
              意味着text[start:end + 1]是类型为label的实体。
    """
    D = []
    with open(filename, encoding='utf-8') as f:
        f = f.read()
        for l in f.split('\n\n'):
            if not l:
                continue
            d = ['']
            for i, c in enumerate(l.split('\n')):
                char, flag = c.split(' ')
                d[0] += char
                if flag[0] == 'B':
                    d.append([i, i, flag[2:]])
                    categories.add(flag[2:])
                elif flag[0] == 'I':
                    d[-1][1] = i
            D.append(d)
    return D

def load_clue_data(filename):
    """加载数据
    单条格式：[text, (start, end, label), (start, end, label), ...]，
              意味着text[start:end + 1]是类型为label的实体。
    """
    D = []
    with open(filename, encoding='utf-8') as f:
        for l in f:
            l = json.loads(l)
            d = [l['text']]
            for k, v in l['label'].items():
                categories.add(k)
                for spans in v.values():
                    for start, end in spans:
                        d.append((start, end, k))
            D.append(d)
    return D
# train_data = load_bid_data(r'D:\PekingInfoResearch\entity_extractor_by_ner\data\delete_table_datasets\train_for_crf.txt')
# valid_data = load_bid_data(r'D:\PekingInfoResearch\entity_extractor_by_ner\data\delete_table_datasets\test_for_crf.txt')
# train_data = load_people_data(r'D:\open_data\ner\china-people-daily-ner-corpus\example.train')
# valid_data = load_people_data(r'D:\open_data\ner\china-people-daily-ner-corpus\example.dev')
train_data = load_clue_data(r'D:\open_data\ner\cluener_public/train.json')[:1000]
valid_data = load_clue_data(r'D:\open_data\ner\cluener_public/dev.json')[:100]
categories = list(sorted(categories))
config = BertConfig.from_pretrained(r'D:\PekingInfoResearch\pretrain_models\bert-base-chinese')
config.__setattr__('head', len(categories))
config.__setattr__('head_size', 64)
model = BertGlobalPointer.from_pretrained(r'D:\PekingInfoResearch\pretrain_models\bert-base-chinese', config=config)

def sequence_padding(inputs, length=None, value=0, seq_dims=1, mode='post'):
    """Numpy函数，将序列padding到同一长度
    """
    if length is None:
        length = np.max([np.shape(x)[:seq_dims] for x in inputs], axis=0)
    elif not hasattr(length, '__getitem__'):
        length = [length]

    slices = [np.s_[:length[i]] for i in range(seq_dims)]
    slices = tuple(slices) if len(slices) > 1 else slices[0]
    pad_width = [(0, 0) for _ in np.shape(inputs[0])]

    outputs = []
    for x in inputs:
        x = x[slices]
        for i in range(seq_dims):
            if mode == 'post':
                pad_width[i] = (0, length[i] - np.shape(x)[i])
            elif mode == 'pre':
                pad_width[i] = (length[i] - np.shape(x)[i], 0)
            else:
                raise ValueError('"mode" argument must be "post" or "pre".')
        x = np.pad(x, pad_width, 'constant', constant_values=value)
        outputs.append(x)

    return np.array(outputs)


def get_map(t, offsets):
    """map的主要目的是将一个片段的token的开始和结尾映射到tokenize后的位置"""
    start_map, end_map = {}, {}
    for i, token in enumerate(t):
        if token in tokenizer.all_special_tokens:
            continue
        start, end = offsets[i]
        start_map[start] = i
        end_map[end] = i
    return start_map, end_map


class TextSamplerDataset(Dataset):
    def __init__(self, data, seq_len, mode='train'):
        super().__init__()
        self.data = data
        self.seq_len = seq_len
        self.mode = mode

    def __getitem__(self, index):
        d = self.data[index]
        outputs = tokenizer(d[0], return_offsets_mapping=True, truncation=True, max_length=self.seq_len)
        tokens = tokenizer.convert_ids_to_tokens(outputs['input_ids'])
        start_map, end_map = get_map(tokens, outputs['offset_mapping'])
        token_ids = outputs['input_ids']
        segment_ids = outputs['token_type_ids']
        mask = outputs['attention_mask']
        labels = np.zeros((len(categories), self.seq_len, self.seq_len))
        for start, end, label in d[1:]:
            if start == end:
                continue
            if start in start_map and end in end_map:
                start = start_map[start]
                end = end_map[end]
                label = categories.index(label)
                labels[label, start, end] = 1

        return d, token_ids, mask, segment_ids, labels[:, :len(token_ids), :len(token_ids)]

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
    d, token_ids, mask, segment_ids, labels = list(zip(*batch))

    token_ids = torch.LongTensor(sequence_padding(token_ids))
    segment_ids = torch.LongTensor(sequence_padding(segment_ids))
    labels = torch.LongTensor(sequence_padding(labels, seq_dims=3))
    mask = torch.LongTensor(sequence_padding(mask))

    return d, token_ids, mask, segment_ids, labels

train_dataset = TextSamplerDataset(train_data, maxlen)
valid_dataset = TextSamplerDataset(valid_data, maxlen)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)


optim = torch.optim.Adam(model.parameters(), lr=2e-5)
loss_fn = torch.nn.BCEWithLogitsLoss(reduction='none')

def multi_label_crossentropy(y_true, y_pred):
    y_pred = (1 - 2 * y_true) * y_pred
    y_neg = y_pred - y_true * 1e12
    y_pos = y_pred - (1 - y_true) * 1e12
    zeros = torch.zeros_like(y_pred[..., :1])
    y_neg = torch.concat([y_neg, zeros], dim=-1)
    y_pos = torch.concat([y_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pos, dim=-1)
    return neg_loss + pos_loss

def calculate_loss(y_true, logits, mask):
    y_true = rearrange(y_true, 'b h m n -> (b h) (m n)')
    logits = rearrange(logits, 'b h m n -> (b h) (m n)')
    loss = multi_label_crossentropy(y_true, logits)
    return torch.mean(loss)

def global_pointer_f1_score(y_true, y_pred):
    """给GlobalPointer设计的F1
    """
    y_pred = torch.greater(y_pred, 0).float()
    return 2 * torch.sum(y_true * y_pred) / torch.sum(y_true + y_pred).clamp(min=1e-9)

def train():
    best_f1 = 0.
    for _ in range(10):
        model.cuda()
        model.train()
        pbar = tqdm(enumerate(train_loader))
        total_loss, total_f1 = 0, 0
        for batch_ind, batch in pbar:
            texts = batch[0]
            batch = [x.cuda() for x in batch[1:]]
            logits = model(*batch[:3])
            # logits = model(batch[0], batch[2], batch[1])
            loss = calculate_loss(batch[3], logits, batch[2])
            f1 = global_pointer_f1_score(batch[3], logits)
            loss.backward()
            optim.step()
            optim.zero_grad()
            total_f1 += f1.item()
            total_loss += loss.item()
            avg_loss = total_loss / (batch_ind + 1)
            avg_f1 = total_f1 / (batch_ind + 1)
            pbar.set_description(f"loss {np.mean(avg_loss)} f1 {np.mean(avg_f1)} lr {optim.param_groups[0]['lr']}")

        model.eval()
        pbar = tqdm(enumerate(valid_loader))
        total_f1 = 0
        for batch_ind, batch in pbar:
            texts = batch[0]
            batch = [x.cuda() for x in batch[1:]]
            logits = model(*batch[:3])
            f1 = global_pointer_f1_score(batch[3], logits)
            total_f1 += f1.item()
            avg_f1 = total_f1 / (batch_ind + 1)
            pbar.set_description(f"f1 {np.mean(avg_f1)}")

        if avg_f1 > best_f1:
            torch.save(model.state_dict(), 'best_model.pt')
            print('best model saved with f1 %f' % avg_f1)
            best_f1 = avg_f1


def evaluate():
    model.load_state_dict(torch.load('best_model.pt'))
    model.cuda()
    model.eval()
    pbar = tqdm(enumerate(valid_loader))
    total_f1 = 0
    for batch_ind, batch in pbar:
        texts = batch[0]
        batch = [x.cuda() for x in batch[1:]]
        logits = model(*batch[:3])
        f1 = global_pointer_f1_score(batch[3], logits)
        total_f1 += f1.item()
        avg_f1 = total_f1 / (batch_ind + 1)
        pbar.set_description(f"f1 {np.mean(avg_f1)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='gau')
    parser.add_argument('-m',
                        type=str,
                        default='test',
                        help='train or eval')
    parser.add_argument('-r',
                        type=str,
                        default='part',
                        help='part or all')
    parser.add_argument('--eval_num',
                        type=int,
                        default=10,
                        help='eval number')
    args = parser.parse_args()
    if args.m == 'train':
        train()
    else:
        evaluate()