import re
from random import randrange, shuffle, randint
import random
import math

import torch
from torch import nn
from torch import optim
import numpy as np

text = (
    'Hello, how are you? I am Romeo.\n'
    'Hello, Romeo My name is Juliet. Nice to meet you.\n'
    'Nice meet you too. How are you today?\n'
    'Great. My baseball team won the competition.\n'
    'Oh Congratulations, Juliet\n'
    'Thanks you Romeo'
)

max_len = 30
batch_size = 6
max_pred = 5
d_model = 768
lays = 6
heads = 12
d_k = 64
d_ff = 4 * 768

sentences = re.sub("[.,!?\\-]", '', text.lower()).split('\n')
word_list = list(set(" ".join(sentences).split(" ")))
word_index = {'[PAD]': 0, '[CLS]': 1, '[SEP]': 2, '[MASK]': 3}
for i, w in enumerate(word_list):
    word_index[w] = i + 4
index_word = {i: w for i, w in enumerate(word_index)}
vocab_size = len(index_word)


# 随机获得两个句子
def get_2_sentences():
    sentence_a_index, sentence_b_index = randrange(len(sentences)), randrange(len(sentences))
    sentence_a, sentence_b = sentences[sentence_a_index], sentences[sentence_b_index]
    # 将sentence中的单词转化为数字
    token_a, token_b = [word_index[w] for w in sentence_a.split()], [word_index[w] for w in sentence_b.split()]
    return token_a, token_b, sentence_a_index, sentence_b_index


def get_bath():
    batch = []
    positive_size = 0
    negitive_size = 0
    while positive_size < batch_size / 2 or negitive_size < batch_size / 2:
        token_a, token_b, sentence_a_index, sentence_b_index = get_2_sentences()
        input_token = [word_index['[CLS]']] + token_a + [word_index['[SEP]']] + token_b + [word_index['[SEP]']]
        segment_token = [0] * (1 + len(token_a) + 1) + [1] * (len(token_b) + 1)
        cand_mask_pos = [i for i, token in enumerate(input_token) if
                         token != word_index['[CLS]'] and token != word_index['[SEP]']]
        pre_num = min(max_pred, max(1, round(len(cand_mask_pos) * 0.15)))
        shuffle(cand_mask_pos)
        mask_pose = []
        mask_token = []
        # 得到mask的单词
        for i in cand_mask_pos[:pre_num]:
            mask_pose.append(i)
            mask_token.append(input_token[i])
            # 再随机一次，如果小于0.8则换成[mask] 如果小于0.5则换成词典中的任意单词中的任意单词
            if random.random() < 0.8:
                input_token[i] = word_index['[MASK]']
            elif random.random() < 0.5:
                index = randint(0, vocab_size - 1)
                input_token[i] = index
        # 保持每个句子等长
        if max_len > len(input_token):
            n_pad = max_len - len(input_token)
            input_token.extend([0] * n_pad)
            segment_token.extend([0] * n_pad)

        if max_pred > len(mask_pose):
            n_pad = max_pred - len(mask_pose)
            mask_pose.extend([0] * n_pad)
            mask_token.extend([0] * n_pad)

        # 分别获得一半批次数量的正样本和负样本
        if sentence_a_index + 1 == sentence_b_index and positive_size < batch_size / 2:
            # 正样本采样
            batch.append([input_token, mask_token, mask_pose, segment_token, True])
            positive_size += 1
        elif sentence_a_index + 1 != sentence_b_index and negitive_size < batch_size / 2:
            # 负样本采样
            batch.append([input_token, mask_token, mask_pose, segment_token, False])
            negitive_size += 1
    return batch


# 定义类Embedding
class Embedding(torch.nn.Module):
    def __init__(self):
        super(Embedding, self).__init__()
        self.input_embed = nn.Embedding(vocab_size, d_model)
        self.seg_embed = nn.Embedding(2, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, input_token, pos_token, segment_token):
        # 相加后记得要加权平均
        embedding = self.input_embed(input_token) + self.pos_embed(pos_token) + self.seg_embed(segment_token)
        return self.norm(embedding)


def get_attn_pad_mask(input_token):
    b_size, q_size = input_token.size()
    attn_pad_mask = input_token.data.eq(0).unsqueeze(1)
    return attn_pad_mask.expand(b_size, q_size, q_size)


class CalAttn(nn.Module):
    def __init__(self):
        super(CalAttn, self).__init__()

    def forward(self, q, k, v, attn_pad_mask):
        k = k.transpose(-1, -2)
        scores = torch.matmul(q, k) / np.sqrt(d_k)
        attn_pad_mask = attn_pad_mask.unsqueeze(1).repeat(1, heads, 1, 1)
        scores.masked_fill_(attn_pad_mask, -1e9)
        attn = nn.Softmax(dim=-1)(scores)
        out_put = torch.matmul(attn, v)
        return out_put, attn


# 定义self_attention
class SelfAttention(nn.Module):
    def __init__(self):
        super(SelfAttention, self).__init__()
        self.w_q = nn.Linear(d_model, heads * d_k)
        self.w_k = nn.Linear(d_model, heads * d_k)
        self.w_v = nn.Linear(d_model, heads * d_k)
        self.cal_attn = CalAttn()

    def forward(self, q, k, v, attn_pad_mask):
        residual = v
        q = self.w_q(q).unsqueeze(2).view(batch_size, -1, heads, d_k).transpose(1, 2)
        k = self.w_q(k).unsqueeze(2).view(batch_size, -1, heads, d_k).transpose(1, 2)
        v = self.w_q(v).unsqueeze(2).view(batch_size, -1, heads, d_k).transpose(1, 2)
        out_put, attn = self.cal_attn(q, k, v, attn_pad_mask)
        # transpose不能直接接view
        out_put = out_put.transpose(1, 2).contiguous().view(batch_size, -1, heads * d_k)
        out_put = nn.Linear(heads * d_k, d_model)(out_put)
        out_put = nn.LayerNorm(d_model)(out_put + residual)
        return out_put, attn


def gelu(x):
    "Implementation of the gelu activation function by Hugging Face"
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


# 定义transformer fedforward层
class FedForward(nn.Module):
    def __init__(self):
        super(FedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, output):
        return self.fc2(gelu(self.fc1(output)))


# 定义transformar网络层
class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.self_attn = SelfAttention()
        self.fed_ward = FedForward()

    def forward(self, out_put, attn_pad_mask):
        out_put, attn = self.self_attn(out_put, out_put, out_put, attn_pad_mask)
        out_put = self.fed_ward(out_put)
        return out_put, attn


# 定义bert网络
class BERT_NN(torch.nn.Module):
    def __init__(self):
        super(BERT_NN, self).__init__()
        self.embed = Embedding()
        embed_weight = self.embed.input_embed.weight
        n_vocab, n_dim = embed_weight.size()
        self.transformer = Transformer()
        self.classier = nn.Linear(d_model, 2)
        self.linear = nn.Linear(d_model, d_model)
        self.activ = gelu
        self.norm = nn.LayerNorm(d_model)
        self.decoder = nn.Linear(d_model, n_vocab)
        self.decoder_bias = nn.Parameter(torch.zeros(n_vocab))

    def forward(self, input_token, mask_pose, segment_token):
        pos_token = torch.arange(input_token.size(1), dtype=torch.long)
        pos_token = pos_token.unsqueeze(0).expand_as(input_token)
        out_put = self.embed(input_token, pos_token, segment_token)
        attn_pad_mask = get_attn_pad_mask(input_token)
        # 多层transformer
        for lay in range(lays):
            out_put, attn = self.transformer(out_put, attn_pad_mask)
        # 取出每个批次的第一个单词既[CLS]
        h_pooled = out_put[:, 0]
        logits_cls = self.classier(h_pooled)

        # 将mask_pose缩放到batch*max_pred*d_model
        mask_pose = mask_pose.unsqueeze(2).expand(-1, -1, d_model)
        h_masked = torch.gather(out_put, 1, mask_pose)
        h_masked = self.norm(self.activ(self.linear(h_masked)))
        logits_lm = self.decoder(h_masked) + self.decoder_bias
        return logits_cls, logits_lm


epoch = 100
model = BERT_NN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
for i in range(epoch):
    batch = get_bath()
    input_token, mask_token, mask_pose, segment_token, is_next = zip(*batch)
    input_token, mask_token, mask_pose, segment_token, is_next = torch.LongTensor(input_token), torch.LongTensor(
        mask_token), torch.LongTensor(mask_pose), torch.LongTensor(segment_token), torch.LongTensor(is_next)
    logits_cls, logits_lm = model(input_token, mask_pose, segment_token)
    # 分别计算mask损失及cls损失
    loss_ml = criterion(logits_lm.transpose(1, 2), mask_token)
    loss_cls = criterion(logits_cls, is_next)
    loss = loss_cls + loss_ml
    if (i % 10 == 0):
        print(f'loss:{loss}')
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
