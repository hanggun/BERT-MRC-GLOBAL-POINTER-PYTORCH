from bert_mrc_pytorch.bert_mrc_global_pointer import BertGlobalPointer, BertModel
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizerFast, BertConfig, AdamW, get_linear_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup
import json
import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as F
from einops import rearrange
from data_loader import DataMaker
from config import config
import argparse

torch.manual_seed(2333)  # pytorch random seed
torch.backends.cudnn.deterministic = True
tokenizer = BertTokenizerFast.from_pretrained(r'D:\PekingInfoResearch\pretrain_models\bert-base-chinese')
maxlen = 256
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
            # if d[1:]:
            #     D.append(d)
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


def load_msra_bmes_data(filename):
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
                if flag[0] == 'S':
                    d.append([i, i, flag[2:]])
                    categories.add(flag[2:])
                if flag[0] == 'B':
                    d.append([i, i, flag[2:]])
                    categories.add(flag[2:])
                elif flag[0] == 'E':
                    d[-1][1] = i
            if d[1:]:
                D.append(d)
    return D


def load_msra_data(filename):
    """加载数据
    单条格式：[text, (start, end, label), (start, end, label), ...]，
              意味着text[start:end + 1]是类型为label的实体。
    """
    D = []
    with open(filename, encoding='utf-8') as f:
        l = json.load(f)
    for line in l:
        if not line['impossible']:
            D.append(line)
            categories.add(line['entity_label'])
    return D


# train_data = load_bid_data(r'D:\PekingInfoResearch\entity_extractor_by_ner\data\delete_table_datasets\train_for_crf.txt')
# valid_data = load_bid_data(r'D:\PekingInfoResearch\entity_extractor_by_ner\data\delete_table_datasets\test_for_crf.txt')
# train_data = load_people_data(r'D:\open_data\ner\china-people-daily-ner-corpus\example.train')
# valid_data = load_people_data(r'D:\open_data\ner\china-people-daily-ner-corpus\example.dev')
# train_data = load_clue_data(r'D:\open_data\ner\cluener_public/train.json')[:1000]
# valid_data = load_clue_data(r'D:\open_data\ner\cluener_public/dev.json')[:100]
train_data = load_msra_data(r'D:\open_data\ner\zh_msra/mrc-ner.train')
valid_data = load_msra_data(r'D:\open_data\ner\zh_msra/mrc-ner.dev')
test_data = load_msra_data(r'D:\open_data\ner\zh_msra/mrc-ner.test')
# train_data = load_msra_bmes_data(r'D:\open_data\ner\zh_msra\train.char.bmes')[:1000]
# valid_data = load_msra_bmes_data(r'D:\open_data\ner\zh_msra\dev.char.bmes')[:100]
categories = list(sorted(categories))
bert_config = BertConfig.from_pretrained(r'D:\PekingInfoResearch\pretrain_models\bert-base-chinese')
bert_config.__setattr__('head', len(categories))
bert_config.__setattr__('head_size', 64)
model = BertGlobalPointer.from_pretrained(r'D:\PekingInfoResearch\pretrain_models\bert-base-chinese', config=bert_config)


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


def get_map(t, offsets, token_type_ids):
    """map的主要目的是将一个片段的token的开始和结尾映射到tokenize后的位置"""
    start_map, end_map = {}, {}
    for i, (token, type) in enumerate(zip(t, token_type_ids)):
        if token in tokenizer.all_special_tokens:
            continue
        if type == 0:
            continue
        start, end = offsets[i]
        start_map[start] = i
        end_map[end-1] = i
    return start_map, end_map


# def get_map(t, offsets, *args, **kwargs):
#     """map的主要目的是将一个片段的token的开始和结尾映射到tokenize后的位置"""
#     start_map, end_map = {}, {}
#     for i, token in enumerate(t):
#         if token in tokenizer.all_special_tokens:
#             continue
#         start, end = offsets[i]
#         start_map[start] = i
#         end_map[end-1] = i
#     return start_map, end_map


class TextSamplerDataset(Dataset):
    def __init__(self, data, seq_len, mode='train'):
        super().__init__()
        self.data = data
        self.seq_len = seq_len
        self.mode = mode

    def __getitem__(self, index):
        d = self.data[index]
        d['query'] = ''.join(d['query'].split())
        d['context'] = ''.join(d['context'].split())
        outputs = tokenizer(d['query'], d['context'], return_offsets_mapping=True, truncation=True, max_length=self.seq_len)
        tokens = tokenizer.convert_ids_to_tokens(outputs['input_ids'])
        start_map, end_map = get_map(tokens, outputs['offset_mapping'], outputs['token_type_ids'])
        token_ids = outputs['input_ids']
        segment_ids = outputs['token_type_ids']
        mask = outputs['attention_mask']
        labels = np.zeros((len(categories), self.seq_len, self.seq_len))

        for start, end in zip(d['start_position'], d['end_position']):
            label = d['entity_label']
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


# class TextSamplerDataset(Dataset):
#     def __init__(self, data, seq_len, mode='train'):
#         super().__init__()
#         self.data = data
#         self.seq_len = seq_len
#         self.mode = mode
#
#     def __getitem__(self, index):
#         d = self.data[index]
#         outputs = tokenizer(d[0], return_offsets_mapping=True, truncation=True, max_length=self.seq_len)
#         tokens = tokenizer.convert_ids_to_tokens(outputs['input_ids'])
#         start_map, end_map = get_map(tokens, outputs['offset_mapping'])
#         token_ids = outputs['input_ids']
#         segment_ids = outputs['token_type_ids']
#         mask = outputs['attention_mask']
#         labels = np.zeros((len(categories), self.seq_len, self.seq_len))
#         for start, end, label in d[1:]:
#             # if start == end:
#             #     continue
#             if start in start_map and end in end_map:
#                 start = start_map[start]
#                 end = end_map[end]
#                 label = categories.index(label)
#                 labels[label, start, end] = 1
#
#         return d, token_ids, mask, segment_ids, labels[:, :len(token_ids), :len(token_ids)]
#
#     def pad(self, lst, value=0, max_length=None):
#         max_length = max_length or self.seq_len
#         if len(lst) > max_length:
#             truncate_len = len(lst) - max_length
#             lst = lst[:-truncate_len-1] + lst[-1:]
#         else:
#             add_len = max_length - len(lst)
#             lst = lst + [value] * add_len
#         return lst
#
#     def __len__(self):
#         return len(self.data)


def collate_fn(batch):
    d, token_ids, mask, segment_ids, labels = list(zip(*batch))

    token_ids = torch.LongTensor(sequence_padding(token_ids))
    segment_ids = torch.LongTensor(sequence_padding(segment_ids))
    labels = torch.LongTensor(sequence_padding(labels, seq_dims=3))
    mask = torch.LongTensor(sequence_padding(mask))

    return d, token_ids, mask, segment_ids, labels

train_dataset = TextSamplerDataset(train_data, maxlen)
valid_dataset = TextSamplerDataset(valid_data, maxlen)
test_dataset = TextSamplerDataset(test_data, maxlen)
train_loader = DataLoader(train_dataset, batch_size=config.batch_size, collate_fn=collate_fn, shuffle=False)
valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size, collate_fn=collate_fn, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn, shuffle=False)

def build_optimizer_and_scheduler(model, config, total_steps):

    module = (model.module if hasattr(model, "module") else model)
    model_param = list(module.named_parameters())

    bert_param = []
    other_param = []
    for name, param in model_param:
        if name.split('.')[0] == 'bert':
            bert_param.append((name, param))
        else:
            other_param.append((name, param))

    no_decay = ["bias", "LayerNorm.weight", 'layer_norm']
    optimizer_grouped_parameters = [
        # bert module
        {'params': [p for n, p in bert_param if not any(nd in n for nd in no_decay)],
         'weight_decay': config.weight_decay, 'lr': config.lr},
        {'params': [p for n, p in bert_param if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0, 'lr': config.lr},

        # other module
        {'params': [p for n, p in other_param if not any(nd in n for nd in no_decay)],
         'weight_decay': config.weight_decay, 'lr': config.other_lr},
        {'params': [p for n, p in other_param if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0, 'lr': config.other_lr}
    ]

    warmup_steps = int(config.warmup_proportion * total_steps)
    optimizer = AdamW(optimizer_grouped_parameters, lr=config.lr, eps=config.adam_epsilon)
    scheduler = get_polynomial_decay_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    return optimizer, scheduler


total_steps = len(train_loader) * config.max_epoches // config.gradient_accumulation_steps
optimizer, scheduler = build_optimizer_and_scheduler(model, config, total_steps)
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
    for _ in range(config.max_epoches):
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

            loss = loss / config.gradient_accumulation_steps
            loss.backward()
            # optimizer.step()
            # optimizer.zero_grad()
            if (batch_ind + 1) % config.gradient_accumulation_steps == 0:
                optimizer.step()  # update parameters of net
                scheduler.step()  # update learning rate schedule
                optimizer.zero_grad()  # reset gradients

            total_f1 += f1.item()
            total_loss += loss.item()
            avg_loss = total_loss / (batch_ind + 1)
            avg_f1 = total_f1 / (batch_ind + 1)
            pbar.set_description(f"loss {np.mean(avg_loss)} f1 {np.mean(avg_f1)} lr {optimizer.param_groups[0]['lr']}")

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


class NamedEntityRecognizer(object):
    """命名实体识别器
    """
    def recognize(self, d, model, threshold=0):
        d['query'] = ''.join(d['query'].split())
        d['context'] = ''.join(d['context'].split())
        outputs = tokenizer(d['query'], d['context'], return_offsets_mapping=True, truncation=True,
                            max_length=maxlen)
        tokens = tokenizer.convert_ids_to_tokens(outputs['input_ids'])
        start_map, end_map = get_map(tokens, outputs['offset_mapping'], outputs['token_type_ids'])
        start_map_inv, end_map_inv = {y:x for x,y in start_map.items()}, {y:x for x,y in end_map.items()}
        token_ids = outputs['input_ids']
        segment_ids = outputs['token_type_ids']
        mask = outputs['attention_mask']
        token_ids = torch.LongTensor(token_ids).cuda()[None, ...]
        segment_ids = torch.LongTensor(segment_ids).cuda()[None, ...]
        mask = torch.LongTensor(mask).cuda()[None, ...]
        scores = model(token_ids, mask=mask, token_type_ids=segment_ids)[0]
        scores[:, [0, -1]] -= np.inf
        scores[:, :, [0, -1]] -= np.inf
        entities = []
        for l, start, end in zip(*torch.where(scores > threshold)):
            entities.append(
                (start_map_inv[start.item()], end_map_inv[end.item()], categories[l.item()])
            )
        return entities


NER = NamedEntityRecognizer()

def evaluate(*args):
    model.load_state_dict(torch.load('best_model.pt'))
    model.cuda()
    model.eval()
    pbar = tqdm(enumerate(test_loader))
    total_f1 = 0
    for batch_ind, batch in pbar:
        texts = batch[0]
        batch = [x.cuda() for x in batch[1:]]
        logits = model(*batch[:3])
        f1 = global_pointer_f1_score(batch[3], logits)
        total_f1 += f1.item()
        avg_f1 = total_f1 / (batch_ind + 1)
        pbar.set_description(f"f1 {np.mean(avg_f1)}")


def evaluate_entity(data):
    """评测函数
    """
    X, Y, Z = 1e-10, 1e-10, 1e-10
    model.load_state_dict(torch.load('best_model.pt'))
    model.cuda()
    model.eval()
    pbar = tqdm(enumerate(data), ncols=100)
    for i, d in pbar:
        R = set(NER.recognize(d, model))
        true_labels = []
        for start, end in zip(d['start_position'], d['end_position']):
            true_labels.append([start, end, d['entity_label']])
        T = set([tuple(i) for i in true_labels])
        if i == 664:
            print(i)
        X += len(R & T)
        Y += len(R)
        Z += len(T)

        pbar.set_description('f1 %f' % (2 * X / (Y + Z)))
    f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
    return f1, precision, recall

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='gau')
    parser.add_argument('-m',
                        type=str,
                        default='eval',
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
        print(evaluate_entity(test_data))