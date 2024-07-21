import os
import random
from pathlib import Path

import torch
import math
import torch.nn.functional as F
from salo_spattn import matchingStatic_Block


# LOG_LOAD_BALANCE = os.getenv('LOG_LOAD_BALANCE', False)
LOG_LOAD_BALANCE = False

if LOG_LOAD_BALANCE:
    csv_path = Path('load_balance.csv')
    assert not csv_path.exists(), f'{csv_path} already exists.'
    csv_file = csv_path.open('w')
    csv_file.write('50%-no-skip,50%-skip,25%-no-skip,25%-skip,overall-sparsity\n')


def _eval_load_balance(sparsity_mask, attn_mask, num_ports=64, num_pes=16, no_skip=False):
    # sparsity_mask: bool, [batch_size, num_heads, seq_len, seq_len]
    # attn_mask: bool, [batch_size, 1, seq_len, seq_len]
    batch_size, num_heads, seq_len, seq_len = sparsity_mask.shape
    assert seq_len % num_ports == 0

    # split sparsity mask into `num_ports`-dim vectors
    # sparsity_mask: [batch_size, num_heads, seq_len * seq_len / num_ports, num_ports]
    sparsity_mask = sparsity_mask.view(batch_size, num_heads, -1, num_ports)

    # count nonzeros in each vector
    # num_nonzero: [batch_size, num_heads, seq_len * seq_len / num_ports]
    num_nonzero = sparsity_mask.sum(dim=-1)
    
    # split attention mask into `num_ports`-dim vectors
    # attn_mask: [batch_size, 1, seq_len * seq_len / num_ports, num_ports]
    attn_mask = attn_mask.view(batch_size, 1, -1, num_ports)

    # vector-wise attention mask: mask out vectors that are completely covered by the original attention mask 
    # attn_mask: bool, [batch_size, 1, seq_len * seq_len / num_ports]
    attn_mask = attn_mask.sum(dim=-1).ne(0)
    
    # filter out masked vectors from num_nonzero
    # num_nonzero: 1-D vector
    num_nonzero = torch.masked_select(num_nonzero, attn_mask)
    
    # count and skip all-zero vectors
    skip_mask = num_nonzero.ne(0)
    num_skips = skip_mask.sum()

    # filter out skipped vectors from num_nonzero
    num_nonzero = torch.masked_select(num_nonzero, skip_mask)
    
    # split non-empty vectors into segments with nnz no greater than num_pes
    # assuming num_pes = 3, a vector of length 10 can be divided into four segments [3, 3, 3, 1]
    # in this case, there are three full segments (where all pes are occupied) and one unfull remnant
    num_splits = num_nonzero / num_pes
    num_full_splits = num_splits.floor().sum()
    num_all_splits = num_splits.ceil().sum()
    
    # a full segment leads to a pe utilization of 100%
    # while pe util of a remnant segment is calculated as num-occupied-pes / num-pes
    acc_full_split_utils = num_full_splits * 1.0
    acc_remn_split_utils = num_splits.frac().sum()
    # accumulated pe utilization of all segments
    acc_all_split_utils = acc_full_split_utils + acc_remn_split_utils

    if no_skip:
        pe_util = acc_all_split_utils / (num_all_splits + num_skips)
    else:
        pe_util = acc_all_split_utils / num_all_splits

    return pe_util.item()


def _eval_overall_sparsity(sparsity_mask, attn_mask):
    # sparsity_mask: bool, [batch_size, num_heads, seq_len, seq_len]
    # attn_mask: bool, [batch_size, 1, seq_len, seq_len]
    scaling_factor = attn_mask.mean(dim=(1, 2, 3))
    sparsity_per_seq = (sparsity_mask * attn_mask).mean(dim=(1, 2, 3))
    overall_sparsity = (sparsity_per_seq / scaling_factor).mean().item()
    return overall_sparsity


def _eval_load_balance2(sparsity_mask, attn_mask, num_ports=64, num_pes=16, no_skip=False):
    attn_mask = (attn_mask > -1).float()
    attn_mask = attn_mask * (attn_mask.permute(0, 1, 3, 2))
    # sparsity_mask: bool, [batch_size, num_heads, seq_len, seq_len]
    # attn_mask: bool, [batch_size, 1, seq_len, seq_len]
    batch_size, num_heads, seq_len, seq_len = sparsity_mask.shape


    # # cloth专用
    # if (seq_len % num_ports != 0):      
    #     a_padding = (0, num_ports-(seq_len%num_ports))
    #     s_padding = (0, num_ports-(seq_len%num_ports)) 
    #     attn_mask = torch.nn.functional.pad(attn_mask, a_padding, mode='constant', value=0.0)
    #     sparsity_mask = torch.nn.functional.pad(sparsity_mask, s_padding, mode='constant', value=0.0) 
    # seq_len = sparsity_mask.shape[-1]


    assert seq_len % num_ports == 0

    # split sparsity mask into `num_ports`-dim vectors
    # sparsity_mask: [batch_size, num_heads, seq_len * seq_len / num_ports, num_ports]
    sparsity_mask = sparsity_mask.view(batch_size, num_heads, -1, num_ports)

    # count nonzeros in each vector
    # num_nonzero: [batch_size, num_heads, seq_len * seq_len / num_ports]
    num_nonzero = sparsity_mask.sum(dim=-1)
    
    # split attention mask into `num_ports`-dim vectors
    # attn_mask: [batch_size, 1, seq_len * seq_len / num_ports, num_ports]
    attn_mask = attn_mask.view(batch_size, 1, -1, num_ports)

    # vector-wise attention mask: mask out vectors that are completely covered by the original attention mask 
    # attn_mask: bool, [batch_size, 1, seq_len * seq_len / num_ports]
    attn_mask = attn_mask.sum(dim=-1).ne(0)
    
    # filter out masked vectors from num_nonzero
    # num_nonzero: 1-D vector
    num_nonzero = torch.masked_select(num_nonzero, attn_mask)
    
    # count and skip all-zero vectors
    skip_mask = num_nonzero.ne(0)
    num_skips = skip_mask.sum()

    # filter out skipped vectors from num_nonzero
    num_nonzero = torch.masked_select(num_nonzero, skip_mask)
    
    # split non-empty vectors into segments with nnz no greater than num_pes
    # assuming num_pes = 3, a vector of length 10 can be divided into four segments [3, 3, 3, 1]
    # in this case, there are three full segments (where all pes are occupied) and one unfull remnant
    num_splits = num_nonzero / num_pes
    num_full_splits = num_splits.floor().sum()
    num_all_splits = num_splits.ceil().sum()
    
    # a full segment leads to a pe utilization of 100%
    # while pe util of a remnant segment is calculated as num-occupied-pes / num-pes
    acc_full_split_utils = num_full_splits * 1.0
    acc_remn_split_utils = num_splits.frac().sum()
    # accumulated pe utilization of all segments
    acc_all_split_utils = acc_full_split_utils + acc_remn_split_utils

    if no_skip:
        pe_util = acc_all_split_utils / (num_all_splits + num_skips)
    else:
        pe_util = acc_all_split_utils / num_all_splits

    return pe_util.item()


def _eval_overall_sparsity2(sparsity_mask, attn_mask):
    # sparsity_mask: bool, [batch_size, num_heads, seq_len, seq_len]
    # attn_mask: bool, [batch_size, 1, 1, seq_len]
    # attn_mask = torch.where(attn_mask != 0, torch.tensor(0.0), torch.tensor(1.0))
    attn_mask = (attn_mask > -1).float()
    attn_mask = attn_mask * (attn_mask.permute(0, 1, 3, 2))
    scaling_factor = attn_mask.mean(dim=(1, 2, 3))
    sparsity_per_seq = (sparsity_mask.float() * attn_mask).mean(dim=(1, 2, 3))
    overall_sparsity = (sparsity_per_seq / scaling_factor).mean().item()
    return overall_sparsity


# 计算不需要取入的QV比例
def _eval_overall_dontQV(sparsity_mask, attn_mask):
    # sparsity_mask: bool, [batch_size, num_heads, seq_len, seq_len]
    # attn_mask: bool, [batch_size, 1, 1, seq_len]
    attn_mask = (attn_mask > -1).float()
    attn_mask = attn_mask * (attn_mask.permute(0, 1, 3, 2))
    sparsity_mask  = torch.where(attn_mask > 0, sparsity_mask, 0)
    sparsity_mask2  = torch.where(attn_mask > 0, 1, 0)
    col_sum_s = torch.sum(sparsity_mask, -2)
    col_sum_s = torch.where(col_sum_s > 0, 1, 0)
    col_sum_s2 = torch.sum(sparsity_mask2, -2)
    col_sum_s2 = torch.where(col_sum_s2 > 0, 1, 0)
    overall_dontQV = 1.0 - (torch.sum(col_sum_s, dim=-1) / torch.sum(col_sum_s2, dim=-1)).mean().item()
    return overall_dontQV


w_count = 0
mean_len = 0
all_sparisity = 0
all_util = 0
all_util2 = 0
all_dontQK = 0


def gen_sparsity_mask_salo(threshold, attention_scores, attn_mask, token_mask):
    global w_count
    global mean_len
    global all_sparisity
    global all_util
    global all_dontQK
    # attention_scores: [batch_size, num_attention_heads, seq_len, seq_len]  attn_mask: 0 OR -10000

    # 原文无需softmax
    # attention_scores = F.softmax(attention_scores + attn_mask, dim=-1)

    match_size = 64
    pe_size = 8
    global_nums = 1
    random_nums = 3
    dilation = 1

    attention_scores_salo = attention_scores
    attn_mask_salo = attn_mask

    # # cloth专用
    # pad_width = (match_size - attention_scores.shape[-1] % match_size) % match_size
    # attention_scores_salo = F.pad(attention_scores, (0, pad_width, 0, pad_width), mode='constant', value=0) 
    # attn_mask_salo = F.pad(attn_mask, (0, pad_width), mode='constant', value=-10000) 
    
    sparsity_mask = matchingStatic_Block(attention_scores_salo,attn_mask_salo, match_size, pe_size, global_nums, random_nums, dilation) # size: [batch_size, num_attention_heads, seq_len, seq_len]
    
    # # cloth专用
    # s_len = attention_scores.shape[-1]
    # sparsity_mask = sparsity_mask[:, :, :s_len, :s_len]

    w_count += 1
    all_sparisity += _eval_overall_sparsity2(sparsity_mask, attn_mask)
    all_util += _eval_load_balance2(sparsity_mask, attn_mask, num_ports=64, num_pes=16, no_skip=False)
    # all_util += _eval_load_balance2(sparsity_mask, attn_mask, num_ports=32, num_pes=8, no_skip=False)
    # all_util += _eval_load_balance2(sparsity_mask, attn_mask, num_ports=32, num_pes=4, no_skip=False)
    # all_dontQK += _eval_overall_dontQV(sparsity_mask, attn_mask)
    mean_len += torch.mean(torch.sum((attn_mask > -1).float(), dim=-1))
    if(w_count == 1) :
        with open('mask_output/sparsity_mask.txt', 'w') as txt:
            txt.writelines('eval_sparsity:\n')
    if(w_count%100 == 1) :
        with open('mask_output/sparsity_mask.txt', 'a') as txt:
            txt.writelines('mean_sparisity: {:<10.5f} mean_util: {:<10.5f} mean_dontQK: {:<10.5f} mean_len: {:<10.5f} w_count: {:<10}\n'.format(round(all_sparisity/w_count, 5), round(all_util/w_count, 5), round(all_dontQK/w_count, 5), round(mean_len.item()/w_count, 5), w_count))
            # txt.writelines('mean_sparisity: {:<10.5f} mean_util: {:<10.5f} mean_len: {:<10.5f} w_count: {:<10}\n'.format(round(all_sparisity/w_count, 5), round(all_util/w_count, 5), round(mean_len.item()/w_count, 5), w_count))

    
    sparsity_mask = sparsity_mask.type_as(attention_scores)


    # Sanger随机评估
    if LOG_LOAD_BALANCE and random.random() < 3e-2:
        attn_mask = (attn_mask > -1).float()
        attn_mask = attn_mask * attn_mask.permute(0, 1, 3, 2)
        logs = [
            _eval_load_balance(sparsity_mask, attn_mask, num_pes=32, no_skip=True), 
            _eval_load_balance(sparsity_mask, attn_mask, num_pes=32, no_skip=False), 
            _eval_load_balance(sparsity_mask, attn_mask, num_pes=16, no_skip=True), 
            _eval_load_balance(sparsity_mask, attn_mask, num_pes=16, no_skip=False),
            _eval_overall_sparsity(sparsity_mask, attn_mask)
        ]
        csv_file.write(','.join([f'{stat:.6f}' for stat in logs]) + '\n')
    sparsity_mask = (1.0 - sparsity_mask) * -10000.0
    
    return sparsity_mask.detach()


def gen_sparsity_mask_sanger(threshold, attention_scores, attn_mask, token_mask):
    global w_count
    global mean_len
    global all_sparisity
    global all_util
    global all_util2
    global all_dontQK
    # attention_scores: [batch_size, num_attention_heads, seq_len, seq_len]
    # attn_mask: 0 OR -10000
    attention_scores = F.softmax(attention_scores + attn_mask, dim=-1)

    sparsity_mask = attention_scores > 2e-3
    # sparsity_mask = attention_scores > threshold

    # # cloth专用
    # s_len = attention_scores.shape[-1]
    # sparsity_mask = sparsity_mask[:, :, :s_len, :s_len]


    w_count += 1
    all_sparisity += _eval_overall_sparsity2(sparsity_mask, attn_mask)
    all_util += _eval_load_balance2(sparsity_mask, attn_mask, num_ports=64, num_pes=16, no_skip=False)
    all_util2 += _eval_load_balance2(sparsity_mask, attn_mask, num_ports=32, num_pes=8, no_skip=False)
    # all_util += _eval_load_balance2(sparsity_mask, attn_mask, num_ports=32, num_pes=4, no_skip=False)
    # all_dontQK += _eval_overall_dontQV(sparsity_mask, attn_mask)
    mean_len += torch.mean(torch.sum((attn_mask > -1).float(), dim=-1))
    if(w_count == 1) :
        with open('mask_output/sparsity_mask.txt', 'w') as txt:
            txt.writelines('eval_sparsity:\n')
    if(w_count%100 == 0) :
        with open('mask_output/sparsity_mask.txt', 'a') as txt:
            # txt.writelines('mean_sparisity: {:<10.5f} mean_util: {:<10.5f} mean_dontQK: {:<10.5f} mean_len: {:<10.5f} w_count: {:<10}\n'
            #                .format(round(all_sparisity/w_count, 5), round(all_util/w_count, 5), round(all_dontQK/w_count, 5), round(mean_len.item()/w_count, 5), w_count))
            txt.writelines('mean_sparisity: {:<10.5f} mean_util_64_16: {:<10.5f} mean_util_32_8: {:<10.5f} mean_len: {:<10.5f} w_count: {:<10}\n'
                           .format(round(all_sparisity/w_count, 5), round(all_util/w_count, 5), round(all_util2/w_count, 5), round(mean_len.item()/w_count, 5), w_count))

    
    sparsity_mask = sparsity_mask.type_as(attention_scores)


    # Sanger随机评估
    if LOG_LOAD_BALANCE and random.random() < 3e-2:
        attn_mask = (attn_mask > -1).float()
        attn_mask = attn_mask * attn_mask.permute(0, 1, 3, 2)
        logs = [
            _eval_load_balance(sparsity_mask, attn_mask, num_pes=32, no_skip=True), 
            _eval_load_balance(sparsity_mask, attn_mask, num_pes=32, no_skip=False), 
            _eval_load_balance(sparsity_mask, attn_mask, num_pes=16, no_skip=True), 
            _eval_load_balance(sparsity_mask, attn_mask, num_pes=16, no_skip=False),
            _eval_overall_sparsity(sparsity_mask, attn_mask)
        ]
        csv_file.write(','.join([f'{stat:.6f}' for stat in logs]) + '\n')
    sparsity_mask = (1.0 - sparsity_mask) * -10000.0
    
    return sparsity_mask.detach()


def gen_sparsity_mask_ours(threshold, attention_scores, attn_mask, token_mask, q1 = 2.0, q2 = 0.5):
    global w_count
    global mean_len
    global all_sparisity
    global all_util
    global all_dontQK
    # attention_scores: [batch_size, num_attention_heads, seq_len, seq_len]
    # attn_mask: 0 OR -10000
    attention_scores = F.softmax(attention_scores + attn_mask, dim=-1)

    sparsity_mask = torch.full_like(attention_scores, False, dtype=torch.bool)
    n1 = 8
    n2 = 4
    m = 32

    attn_mask_tmp = (attn_mask > -1).float()
    token_len = torch.sum(attn_mask_tmp, dim=-1)
    # token_len = torch.where(token_len < m, m * 1.2 , token_len)
    slices_a = list(torch.split(attention_scores, m, dim=-1))
    slices_s = list(torch.split(sparsity_mask, m, dim=-1))
    for i in range(len(slices_a)):
        sum_a = torch.sum(slices_a[i], dim=-1) # 范围内分数和
        sum_a2 = sum_a * token_len / m #根据占比划分
        sum_a3 = torch.unsqueeze(sum_a2, dim=-1).expand(sum_a2.shape+(m,))

        # print(torch.round(sum_a * 100) / 100)
        # print(torch.round(sum_a2 * 100) / 100)

        # # cloth专用
        # a_padding = (0, m-slices_a[i].shape[-1]) 
        # s_padding = (0, m-slices_s[i].shape[-1]) 
        # slices_a[i] = F.pad(slices_a[i], a_padding, mode='constant', value=0)
        # slices_s[i] = F.pad(slices_s[i], s_padding, mode='constant', value=0)

        index_n1 = torch.topk(slices_a[i], n1, dim=-1, largest=True)[1]
        index_n2 = torch.topk(slices_a[i], n2, dim=-1, largest=True)[1]
        slices_s1 = slices_s[i].clone()
        slices_s2 = slices_s[i].clone()
        slices_s3 = slices_s[i].clone()
        slices_s1.scatter_(-1, index_n1, True)
        slices_s2.scatter_(-1, index_n2, True)
        slices_s[i] = torch.where(sum_a3 < q2, slices_s3, slices_s2)
        slices_s[i] = torch.where(sum_a3 > q1, slices_s1, slices_s[i])
    sparsity_mask = torch.cat(slices_s, dim=-1)

    # # cloth专用
    # s_len = attention_scores.shape[-1]
    # sparsity_mask = sparsity_mask[:, :, :s_len, :s_len]


    w_count += 1
    all_sparisity += _eval_overall_sparsity2(sparsity_mask, attn_mask)
    all_util += _eval_load_balance2(sparsity_mask, attn_mask, num_ports=64, num_pes=16, no_skip=False)
    # all_util += _eval_load_balance2(sparsity_mask, attn_mask, num_ports=32, num_pes=8, no_skip=False)
    # all_util += _eval_load_balance2(sparsity_mask, attn_mask, num_ports=32, num_pes=4, no_skip=False)
    # all_dontQK += _eval_overall_dontQV(sparsity_mask, attn_mask)
    mean_len += torch.mean(torch.sum((attn_mask > -1).float(), dim=-1))
    if(w_count == 1) :
        with open('mask_output/sparsity_mask.txt', 'w') as txt:
            txt.writelines('eval_sparsity:\n')
    if(w_count%100 == 1) :
        with open('mask_output/sparsity_mask.txt', 'a') as txt:
            txt.writelines('mean_sparisity: {:<10.5f} mean_util: {:<10.5f} mean_dontQK: {:<10.5f} mean_len: {:<10.5f} w_count: {:<10}\n'.format(round(all_sparisity/w_count, 5), round(all_util/w_count, 5), round(all_dontQK/w_count, 5), round(mean_len.item()/w_count, 5), w_count))
            # txt.writelines('mean_sparisity: {:<10.5f} mean_util: {:<10.5f} mean_len: {:<10.5f} w_count: {:<10}\n'.format(round(all_sparisity/w_count, 5), round(all_util/w_count, 5), round(mean_len.item()/w_count, 5), w_count))

    
    sparsity_mask = sparsity_mask.type_as(attention_scores)


    # Sanger随机评估
    if LOG_LOAD_BALANCE and random.random() < 3e-2:
        attn_mask = (attn_mask > -1).float()
        attn_mask = attn_mask * attn_mask.permute(0, 1, 3, 2)
        logs = [
            _eval_load_balance(sparsity_mask, attn_mask, num_pes=32, no_skip=True), 
            _eval_load_balance(sparsity_mask, attn_mask, num_pes=32, no_skip=False), 
            _eval_load_balance(sparsity_mask, attn_mask, num_pes=16, no_skip=True), 
            _eval_load_balance(sparsity_mask, attn_mask, num_pes=16, no_skip=False),
            _eval_overall_sparsity(sparsity_mask, attn_mask)
        ]
        csv_file.write(','.join([f'{stat:.6f}' for stat in logs]) + '\n')
    sparsity_mask = (1.0 - sparsity_mask) * -10000.0
    
    return sparsity_mask.detach()


def gen_sparsity_mask_topk(threshold, attention_scores, attn_mask, token_mask):
    global w_count
    global mean_len
    global all_sparisity
    global all_util
    global all_dontQK
    # attention_scores: [batch_size, num_attention_heads, seq_len, seq_len]
    # attn_mask: 0 OR -10000
    attention_scores = F.softmax(attention_scores + attn_mask, dim=-1)

    sparsity_mask = torch.full_like(attention_scores, False, dtype=torch.bool)
    top_k = 10
    index = torch.topk(attention_scores, top_k, dim=-1, largest=True)[1]
    sparsity_mask.scatter_(-1, index, True)


    # # cloth专用
    # s_len = attention_scores.shape[-1]
    # sparsity_mask = sparsity_mask[:, :, :s_len, :s_len]


    w_count += 1
    all_sparisity += _eval_overall_sparsity2(sparsity_mask, attn_mask)
    all_util += _eval_load_balance2(sparsity_mask, attn_mask, num_ports=64, num_pes=16, no_skip=False)
    # all_util += _eval_load_balance2(sparsity_mask, attn_mask, num_ports=32, num_pes=8, no_skip=False)
    # all_util += _eval_load_balance2(sparsity_mask, attn_mask, num_ports=32, num_pes=4, no_skip=False)
    all_dontQK += _eval_overall_dontQV(sparsity_mask, attn_mask)
    mean_len += torch.mean(torch.sum((attn_mask > -1).float(), dim=-1))
    if(w_count == 1) :
        with open('mask_output/sparsity_mask.txt', 'w') as txt:
            txt.writelines('eval_sparsity:\n')
    if(w_count%100 == 0) :
        with open('mask_output/sparsity_mask.txt', 'a') as txt:
            txt.writelines('mean_sparisity: {:<10.5f} mean_util: {:<10.5f} mean_dontQK: {:<10.5f} mean_len: {:<10.5f} w_count: {:<10}\n'.format(round(all_sparisity/w_count, 5), round(all_util/w_count, 5), round(all_dontQK/w_count, 5), round(mean_len.item()/w_count, 5), w_count))
            # txt.writelines('mean_sparisity: {:<10.5f} mean_util: {:<10.5f} mean_len: {:<10.5f} w_count: {:<10}\n'.format(round(all_sparisity/w_count, 5), round(all_util/w_count, 5), round(mean_len.item()/w_count, 5), w_count))

    
    sparsity_mask = sparsity_mask.type_as(attention_scores)


    # Sanger随机评估
    if LOG_LOAD_BALANCE and random.random() < 3e-2:
        attn_mask = (attn_mask > -1).float()
        attn_mask = attn_mask * attn_mask.permute(0, 1, 3, 2)
        logs = [
            _eval_load_balance(sparsity_mask, attn_mask, num_pes=32, no_skip=True), 
            _eval_load_balance(sparsity_mask, attn_mask, num_pes=32, no_skip=False), 
            _eval_load_balance(sparsity_mask, attn_mask, num_pes=16, no_skip=True), 
            _eval_load_balance(sparsity_mask, attn_mask, num_pes=16, no_skip=False),
            _eval_overall_sparsity(sparsity_mask, attn_mask)
        ]
        csv_file.write(','.join([f'{stat:.6f}' for stat in logs]) + '\n')
    sparsity_mask = (1.0 - sparsity_mask) * -10000.0
    
    return sparsity_mask.detach()


def gen_sparsity_mask_nm(threshold, attention_scores, attn_mask, token_mask):
    global w_count
    global mean_len
    global all_sparisity
    global all_util
    global all_dontQK
    # attention_scores: [batch_size, num_attention_heads, seq_len, seq_len]
    # attn_mask: 0 OR -10000
    attention_scores = F.softmax(attention_scores + attn_mask, dim=-1)

    sparsity_mask = torch.full_like(attention_scores, False, dtype=torch.bool)
    n = 8
    m = 32
    slices_a = torch.split(attention_scores, m, dim=-1)
    slices_s = torch.split(sparsity_mask, m, dim=-1)
    for i in range (len(slices_a)):
        index = torch.topk(slices_a[i], n, dim=-1, largest=True)[1]
        slices_s[i].scatter_(-1, index, True)
    sparsity_mask = torch.cat(slices_s, dim=-1)


    # # cloth专用
    # s_len = attention_scores.shape[-1]
    # sparsity_mask = sparsity_mask[:, :, :s_len, :s_len]


    w_count += 1
    all_sparisity += _eval_overall_sparsity2(sparsity_mask, attn_mask)
    all_util += _eval_load_balance2(sparsity_mask, attn_mask, num_ports=64, num_pes=16, no_skip=False)
    # all_util += _eval_load_balance2(sparsity_mask, attn_mask, num_ports=32, num_pes=8, no_skip=False)
    # all_util += _eval_load_balance2(sparsity_mask, attn_mask, num_ports=32, num_pes=4, no_skip=False)
    all_dontQK += _eval_overall_dontQV(sparsity_mask, attn_mask)
    mean_len += torch.mean(torch.sum((attn_mask > -1).float(), dim=-1))
    if(w_count == 1) :
        with open('mask_output/sparsity_mask.txt', 'w') as txt:
            txt.writelines('eval_sparsity:\n')
    if(w_count%100 == 0) :
        with open('mask_output/sparsity_mask.txt', 'a') as txt:
            txt.writelines('mean_sparisity: {:<10.5f} mean_util: {:<10.5f} mean_dontQK: {:<10.5f} mean_len: {:<10.5f} w_count: {:<10}\n'.format(round(all_sparisity/w_count, 5), round(all_util/w_count, 5), round(all_dontQK/w_count, 5), round(mean_len.item()/w_count, 5), w_count))
            # txt.writelines('mean_sparisity: {:<10.5f} mean_util: {:<10.5f} mean_len: {:<10.5f} w_count: {:<10}\n'.format(round(all_sparisity/w_count, 5), round(all_util/w_count, 5), round(mean_len.item()/w_count, 5), w_count))

    
    sparsity_mask = sparsity_mask.type_as(attention_scores)


    # Sanger随机评估
    if LOG_LOAD_BALANCE and random.random() < 3e-2:
        attn_mask = (attn_mask > -1).float()
        attn_mask = attn_mask * attn_mask.permute(0, 1, 3, 2)
        logs = [
            _eval_load_balance(sparsity_mask, attn_mask, num_pes=32, no_skip=True), 
            _eval_load_balance(sparsity_mask, attn_mask, num_pes=32, no_skip=False), 
            _eval_load_balance(sparsity_mask, attn_mask, num_pes=16, no_skip=True), 
            _eval_load_balance(sparsity_mask, attn_mask, num_pes=16, no_skip=False),
            _eval_overall_sparsity(sparsity_mask, attn_mask)
        ]
        csv_file.write(','.join([f'{stat:.6f}' for stat in logs]) + '\n')
    sparsity_mask = (1.0 - sparsity_mask) * -10000.0
    
    return sparsity_mask.detach()


def gen_sparsity_mask_all(threshold, attention_scores, attn_mask, token_mask):
    global w_count
    global mean_len
    global all_sparisity
    global all_util
    global all_dontQK
    # attention_scores: [batch_size, num_attention_heads, seq_len, seq_len]
    # attn_mask: 0 OR -10000
    attention_scores = F.softmax(attention_scores + attn_mask, dim=-1)
    if (token_mask == None) :
        token_mask = attn_mask > -1

    
    ''' -------------------------------------------------------threshold'''
    sparsity_mask = attention_scores > 2e-2
    # sparsity_mask = attention_scores > threshold
    # sparsity_mask = attention_scores >= 0

    ''' 统计token分数与稀疏度的关系'''
    # for i in range(len(attention_scores)) :
    #     real_len = torch.sum((attn_mask[i] > -1).float(), dim=-1) # [ 1, 1]
    #     real_len = torch.round(real_len).long()
    #     col_sums = attention_scores[i][..., :real_len[0], :real_len[0]].sum(dim=(-2, 0)) # [seq_len]
    #     spa_sums = sparsity_mask[i][..., :real_len[0], :real_len[0]].sum(dim = (-1, 0)) # [seq_len]
    #     if(w_count == 0) :
    #         with open('mask_output/token.txt', 'w') as txt:
    #             txt.writelines('token to sparsity:\n')
    #     if(w_count%1000 == 1) :
    #         with open('mask_output/token.txt', 'a') as txt:
    #             txt.writelines('token_len:\n')
    #             txt.writelines(str(real_len[0]))
    #             txt.writelines('\n')
    #             txt.writelines('score_col_sums:\n')
    #             txt.writelines(str(torch.round(col_sums * 100) / 100))
    #             txt.writelines('\n')
    #             txt.writelines('non_zero_sums:\n')
    #             txt.writelines(str(spa_sums))
    #             txt.writelines('\n============================================================================================\n')


    ''' -------------------------------------------------------topk'''
    # sparsity_mask = torch.full_like(attention_scores, False, dtype=torch.bool)
    # top_k = 9
    # index = torch.topk(attention_scores, top_k, dim=-1, largest=True)[1]
    # sparsity_mask.scatter_(-1, index, True)


    ''' -------------------------------------------------------N:M'''
    # sparsity_mask = torch.full_like(attention_scores, False, dtype=torch.bool)
    # n = 8
    # m = 32
    # slices_a = torch.split(attention_scores, m, dim=-1)
    # slices_s = torch.split(sparsity_mask, m, dim=-1)
    # for i in range (len(slices_a)):
    #     index = torch.topk(slices_a[i], n, dim=-1, largest=True)[1]
    #     slices_s[i].scatter_(-1, index, True)
    # sparsity_mask = torch.cat(slices_s, dim=-1)


    ''' -------------------------------------------------------fusion N:M (0/4/8 : 32)'''
    # sparsity_mask = torch.full_like(attention_scores, False, dtype=torch.bool)
    # n1 = 8
    # n2 = 4
    # m = 32

    # q1 = 2.0
    # q2 = 0.5

    # attn_mask_tmp = (attn_mask > -1).float()
    # token_len = torch.sum(attn_mask_tmp, dim=-1)
    # # token_len = torch.where(token_len < m, m * 1.2 , token_len)
    # slices_a = list(torch.split(attention_scores, m, dim=-1))
    # slices_s = list(torch.split(sparsity_mask, m, dim=-1))
    # for i in range(len(slices_a)):
    #     sum_a = torch.sum(slices_a[i], dim=-1) # 范围内分数和
    #     sum_a2 = sum_a * token_len / m #根据占比划分
    #     sum_a3 = torch.unsqueeze(sum_a2, dim=-1).expand(sum_a2.shape+(m,))

    #     # print(torch.round(sum_a * 100) / 100)
    #     # print(torch.round(sum_a2 * 100) / 100)

    #     # # cloth专用
    #     # a_padding = (0, m-slices_a[i].shape[-1]) 
    #     # s_padding = (0, m-slices_s[i].shape[-1]) 
    #     # slices_a[i] = torch.nn.functional.pad(slices_a[i], a_padding, mode='constant', value=0)
    #     # slices_s[i] = torch.nn.functional.pad(slices_s[i], s_padding, mode='constant', value=0)

    #     index_n1 = torch.topk(slices_a[i], n1, dim=-1, largest=True)[1]
    #     index_n2 = torch.topk(slices_a[i], n2, dim=-1, largest=True)[1]
    #     slices_s1 = slices_s[i].clone()
    #     slices_s2 = slices_s[i].clone()
    #     slices_s3 = slices_s[i].clone()
    #     slices_s1.scatter_(-1, index_n1, True)
    #     slices_s2.scatter_(-1, index_n2, True)
    #     slices_s[i] = torch.where(sum_a3 < q2, slices_s3, slices_s2)
    #     slices_s[i] = torch.where(sum_a3 > q1, slices_s1, slices_s[i])
    # sparsity_mask = torch.cat(slices_s, dim=-1)

    # # 双重稀疏
    # sparsity_mask2 = attention_scores > 2e-3
    # sparsity_mask = sparsity_mask & sparsity_mask2


    ''' -------------------------------------------------------动态 fusion N:M (0/4/8 : 32)'''
    # sparsity_mask = torch.full_like(attention_scores, False, dtype=torch.bool)
    # n1 = 8
    # n2 = 4
    # m = 32

    # q1 = 1.8
    # q2 = 0.5

    # q1_ = 2.4
    # q2_ = 0.6

    # token_mask = token_mask.expand(sparsity_mask.shape)
    # attn_mask_tmp = (attn_mask > -1).float()
    # token_len = torch.sum(attn_mask_tmp, dim=-1)
    # # token_len = torch.where(token_len < m, m * 1.2 , token_len)
    # slices_a = list(torch.split(attention_scores, m, dim=-1))
    # slices_s = list(torch.split(sparsity_mask, m, dim=-1))
    # slices_s_ = list(torch.split(sparsity_mask, m, dim=-1))
    # for i in range(len(slices_a)):
    #     sum_a = torch.sum(slices_a[i], dim=-1) # 范围内分数和
    #     sum_a2 = sum_a * token_len / m #根据占比划分
    #     sum_a3 = torch.unsqueeze(sum_a2, dim=-1).expand(sum_a2.shape+(m,))

    #     # # cloth专用
    #     # a_padding = (0, m-slices_a[i].shape[-1]) 
    #     # s_padding = (0, m-slices_s[i].shape[-1]) 
    #     # slices_a[i] = torch.nn.functional.pad(slices_a[i], a_padding, mode='constant', value=0)
    #     # slices_s[i] = torch.nn.functional.pad(slices_s[i], s_padding, mode='constant', value=0)

    #     index_n1 = torch.topk(slices_a[i], n1, dim=-1, largest=True)[1]
    #     index_n2 = torch.topk(slices_a[i], n2, dim=-1, largest=True)[1]
    #     slices_s1 = slices_s[i].clone()
    #     slices_s2 = slices_s[i].clone()
    #     slices_s3 = slices_s[i].clone() # zero
    #     slices_s1.scatter_(-1, index_n1, True)
    #     slices_s2.scatter_(-1, index_n2, True)
    #     slices_s[i] = torch.where(sum_a3 < q2, slices_s3, slices_s2)
    #     slices_s[i] = torch.where(sum_a3 > q1, slices_s1, slices_s[i])
    #     slices_s_[i] = torch.where(sum_a3 < q2_, slices_s3, slices_s2)
    #     slices_s_[i] = torch.where(sum_a3 > q1_, slices_s1, slices_s_[i])
    # sparsity_mask_low = torch.cat(slices_s, dim=-1) # 低稀疏
    # sparsity_mask_high = torch.cat(slices_s_, dim=-1) # 高稀疏
    # # 低稀疏token为True
    # sparsity_mask = torch.where(token_mask == True, sparsity_mask_low, sparsity_mask_high)


    ''' -------------------------------------------------------剪枝 fusion N:M (0/4/8 : 32)'''
    # sparsity_mask = torch.full_like(attention_scores, False, dtype=torch.bool)
    # n1 = 8
    # n2 = 4
    # m = 32

    # q1 = 2
    # q2 = 0.5

    # attn_mask_tmp = (attn_mask > -1).float()
    # token_len = torch.sum(attn_mask_tmp, dim=-1)
    # # token_len = torch.where(token_len < m, m * 1.2 , token_len)
    # for b in range(0, len(attn_mask_tmp)):
    #     slices_a = []
    #     slices_s = []
    #     nonzero_indices = torch.nonzero(attn_mask_tmp[b][0][0] != 0).squeeze()
    #     start_idx = 0
    #     end_idx = 0
    #     indices_len = len(nonzero_indices)
    #     for i in range(0, indices_len, m):
    #         if (i + m < indices_len):
    #             end_idx = nonzero_indices[i+m].item()
    #         else:
    #             end_idx = attn_mask[b].shape[-1]
    #         slices_a.append(attention_scores[b][..., start_idx:end_idx])
    #         slices_s.append(sparsity_mask[b][..., start_idx:end_idx])
    #         start_idx = end_idx
    #     for i in range(len(slices_a)):
    #         sum_a = torch.sum(slices_a[i], dim=-1) # 范围内分数和
    #         sum_a2 = sum_a * token_len[b] / m # 根据占比划分
    #         sum_a3 = torch.unsqueeze(sum_a2, dim=-1).expand(sum_a2.shape+(slices_a[i].shape[-1],))

    #         # # cloth专用
    #         # if (m-slices_a[i].shape[-1] > 0) :
    #         #     a_padding = (0, m-slices_a[i].shape[-1]) 
    #         #     s_padding = (0, m-slices_s[i].shape[-1]) 
    #         #     slices_a[i] = torch.nn.functional.pad(slices_a[i], a_padding, mode='constant', value=0)
    #         #     slices_s[i] = torch.nn.functional.pad(slices_s[i], s_padding, mode='constant', value=0)
            
    #         n1 =  min(slices_a[i].shape[-1], n1)
    #         n2 =  min(slices_a[i].shape[-1], n2)
    #         index_n1 = torch.topk(slices_a[i], n1, dim=-1, largest=True)[1]
    #         index_n2 = torch.topk(slices_a[i], n2, dim=-1, largest=True)[1]
    #         slices_s1 = slices_s[i].clone()
    #         slices_s2 = slices_s[i].clone()
    #         slices_s3 = slices_s[i].clone()
    #         slices_s1.scatter_(-1, index_n1, True)
    #         slices_s2.scatter_(-1, index_n2, True)
    #         slices_s[i] = torch.where(sum_a3 < q2, slices_s3, slices_s2)
    #         slices_s[i] = torch.where(sum_a3 > q1, slices_s1, slices_s[i])
    #     sparsity_mask[b] = torch.cat(slices_s, dim=-1)


    # # cloth专用
    # s_len = attention_scores.shape[-1]
    # sparsity_mask = sparsity_mask[:, :, :s_len, :s_len]


    w_count += 1
    all_sparisity += _eval_overall_sparsity2(sparsity_mask, attn_mask)
    all_util += _eval_load_balance2(sparsity_mask, attn_mask, num_ports=64, num_pes=16, no_skip=False)
    # all_util += _eval_load_balance2(sparsity_mask, attn_mask, num_ports=32, num_pes=8, no_skip=False)
    # all_util += _eval_load_balance2(sparsity_mask, attn_mask, num_ports=32, num_pes=4, no_skip=False)
    all_dontQK += _eval_overall_dontQV(sparsity_mask, attn_mask)
    mean_len += torch.mean(torch.sum((attn_mask > -1).float(), dim=-1))
    if(w_count == 1) :
        with open('mask_output/sparsity_mask.txt', 'w') as txt:
            txt.writelines('eval_sparsity:\n')
    if(w_count%100 == 0) :
        with open('mask_output/sparsity_mask.txt', 'a') as txt:
            txt.writelines('mean_sparisity: {:<10.5f} mean_util: {:<10.5f} mean_dontQK: {:<10.5f} mean_len: {:<10.5f} w_count: {:<10}\n'.format(round(all_sparisity/w_count, 5), round(all_util/w_count, 5), round(all_dontQK/w_count, 5), round(mean_len.item()/w_count, 5), w_count))
            # txt.writelines('mean_sparisity: {:<10.5f} mean_util: {:<10.5f} mean_len: {:<10.5f} w_count: {:<10}\n'.format(round(all_sparisity/w_count, 5), round(all_util/w_count, 5), round(mean_len.item()/w_count, 5), w_count))

    
    sparsity_mask = sparsity_mask.type_as(attention_scores)


    # Sanger随机评估
    if LOG_LOAD_BALANCE and random.random() < 3e-2:
        attn_mask = (attn_mask > -1).float()
        attn_mask = attn_mask * attn_mask.permute(0, 1, 3, 2)
        logs = [
            _eval_load_balance(sparsity_mask, attn_mask, num_pes=32, no_skip=True), 
            _eval_load_balance(sparsity_mask, attn_mask, num_pes=32, no_skip=False), 
            _eval_load_balance(sparsity_mask, attn_mask, num_pes=16, no_skip=True), 
            _eval_load_balance(sparsity_mask, attn_mask, num_pes=16, no_skip=False),
            _eval_overall_sparsity(sparsity_mask, attn_mask)
        ]
        csv_file.write(','.join([f'{stat:.6f}' for stat in logs]) + '\n')
    sparsity_mask = (1.0 - sparsity_mask) * -10000.0
    
    return sparsity_mask.detach()


def gen_sparsity_mask(threshold, attn_scores, attn_mask, token_mask): 
    # return gen_sparsity_mask_ours(threshold, attn_scores, attn_mask, token_mask, 2.0, 0.5)
    return gen_sparsity_mask_sanger(threshold, attn_scores, attn_mask, token_mask)
    # return gen_sparsity_mask_salo(threshold, attn_scores, attn_mask, token_mask)


def quant_qk_matmul(query_layer, key_layer, config, quant_matmul=None):

    # 对 Q 使用 1 : 2
    last_dim = query_layer.shape[-1]
    assert last_dim % 2 == 0, "last_dim must even"
    sparse_query_layer = torch.zeros_like(query_layer)
    for i in range(0, last_dim, 2):
        part = query_layer[..., i:i+2]
        abs_part = torch.abs(part)
        max_index = torch.argmax(abs_part, dim=-1, keepdim=True)
        sparse_part = torch.zeros_like(part).scatter_(-1, max_index, part.gather(-1, max_index))
        sparse_query_layer[..., i:i+2] = sparse_part
    query_layer = sparse_query_layer


    assert getattr(config, 'quant_qk', False)
    do_normalize = getattr(config, 'normalize_qk', False)
    if do_normalize:
        assert config.normalize_qk == 'inner_product'
        query_norm = query_layer.norm(dim=-1, keepdim=True)
        key_norm = key_layer.norm(dim=-2, keepdim=True)
        normed_query_layer = query_layer / query_norm
        normed_key_layer = key_layer / key_norm
        quant_attention_scores = quant_matmul(normed_query_layer, normed_key_layer)
        quant_attention_scores *= query_norm * key_norm
    else:
        quant_attention_scores = quant_matmul(query_layer, key_layer)

    return quant_attention_scores


def prune_attn_scores(attn_scores, attn_mask, token_mask, config):
    assert getattr(config, 'prune_score', False)
    threshold = config.prune_score['threshold']
    sparsity_mask = gen_sparsity_mask(threshold, attn_scores, attn_mask, token_mask)
    return sparsity_mask
