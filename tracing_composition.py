import numpy as np
import json#, jsonlines
import matplotlib.pyplot as plt
from eval_qa import eval_file, eval_items
import os
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
from tqdm import tqdm
import torch
import torch.nn.functional as F
import pandas as pd
from copy import deepcopy
import random
import argparse
# python tracing_composition.py --dataset composition1_2000_200_3.6
def main():
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", default=r'composition1_2000_200_12.6', type=str, required=True, help="dataset name")
    parser.add_argument("--model_dir", default=r'output/12.6_0.1_8use_cot', type=str, help="parent directory of saved model checkpoints")
    parser.add_argument("--save_path", default=r'output/causal_result_hop1.json', type=str, help="path to save result")

    parser.add_argument("--num_layer", default=8, type=int, help="number of layer of the model")
    parser.add_argument("--wd", default=0.1, type=float, help="weight decay being used")
    
    args = parser.parse_args()
    dataset, model_dir = args.dataset, args.model_dir

    # directory = os.path.join(model_dir, "{}_{}_{}".format(dataset, args.wd, args.num_layer))
    directory = model_dir

    device = torch.device('cuda:0')

    all_atomic = set()     # (h,r,t)
    atomic_dict = dict()   # (h,r) -> t
    with open("data/{}/train.json".format(dataset)) as f:
        train_items = json.load(f)
    for item in tqdm(train_items):
        temp = item['target_text'].strip("><").split("><")
        if len(temp) != 4:
            continue
        h,r,t = temp[:3]
        atomic_dict[(h,r)] = t
        all_atomic.add((h,r,t))

    id_atomic = set()
    for item in tqdm(train_items):
        temp = item['target_text'].strip("><").split("><")
        if len(temp) == 4:
            continue
        h, r1, r2, b, t = temp[:5]
        b = atomic_dict[(h, r1)]
        try:
            assert atomic_dict[(b, r2)] == t
            id_atomic.add((h,r1,b))
            id_atomic.add((b,r2,t))
        except:
            continue

    ood_atomic = all_atomic - id_atomic
    print("# id_atomic, # ood_atomic:", len(id_atomic), len(ood_atomic))

    h2rt_train = dict()
    for (h,r,t) in id_atomic:
        if h not in h2rt_train:
            h2rt_train[h] = []
        h2rt_train[h].append((r,t))

    with open("data/{}/test.json".format(dataset)) as f:
        pred_data = json.load(f)
    d = dict()
    for item in pred_data:
        t = item['type']
        if t not in d:
            d[t] = []
        d[t].append(item)

    def return_rank(hd, word_embedding_, token, metric='dot', token_list=None):  #target t' rank in hidden emb
        if metric == 'dot':
            word_embedding = word_embedding_
        elif metric == 'cos':
            word_embedding = F.normalize(word_embedding_, p=2, dim=1)
        else:
            assert False

        logits_ = torch.matmul(hd, word_embedding.T)

        rank = [] 
        for j in range(len(logits_)):
            log = logits_[j].cpu().numpy()
            if token_list is None:
                temp = [[i, log[i]] for i in range(len(log))]
            else:
                temp = [[i, log[i]] for i in token_list]
            temp.sort(key=lambda var: var[1], reverse=True)
            rank.append([var[0] for var in temp].index(token))
        return rank

    all_checkpoints = [checkpoint for checkpoint in os.listdir(directory) if checkpoint.startswith("checkpoint")]
    all_checkpoints.sort(key=lambda var: int(var.split("-")[1]))

    results = []

    np.random.seed(0)
    split = 'train_inferred'
    split = 'test_inferred_iid'
    # split = 'test_inferred_ood'
    rand_inds = np.random.choice(len(d[split]), 300, replace=False).tolist()[:10]

    target_layer = 8

    for checkpoint in tqdm(all_checkpoints[-1:]):
        print("now checkpoint", checkpoint)
        model_path = os.path.join(directory, checkpoint)

        torch.cuda.empty_cache()

        model = GPT2LMHeadModel.from_pretrained(model_path).to(device)
        word_embedding = model.lm_head.weight.data
        tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        tokenizer.padding_side = "left" 
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = model.config.eos_token_id

        full_list = []
        for index in tqdm(rand_inds):

            random.seed(index)
            
            res_dict = dict()

            query = d[split][index]['input_text']
            h,n_r,r = query.strip("><").split("><")
            b = atomic_dict[(h, n_r)]
            t = atomic_dict[(b, r)]
            # TODO:check the first hop
            decoder_temp = tokenizer([query], return_tensors="pt", padding=True)
            decoder_input_ids, decoder_attention_mask = decoder_temp["input_ids"], decoder_temp["attention_mask"]
            decoder_input_ids, decoder_attention_mask = decoder_input_ids.to(device), decoder_attention_mask.to(device)
            
            with torch.no_grad():
                outputs = model(
                    input_ids=decoder_input_ids,
                    attention_mask=decoder_attention_mask,
                    output_hidden_states=True
                )
            all_hidden_states = outputs['hidden_states']
            print(len(all_hidden_states))
            print(all_hidden_states[target_layer][0, :, :].shape)
            rank_before = return_rank(all_hidden_states[target_layer][0, :, :], word_embedding, tokenizer("<"+b+">")['input_ids'][0])[-1]
            res_dict['b_rank_before'] = rank_before
            rank_before = return_rank(all_hidden_states[target_layer][0, :, :], word_embedding, tokenizer("<"+t+">")['input_ids'][0])[-1]
            res_dict['t_rank_before'] = rank_before

            # MRRs
            for layer_ind in range(0, target_layer):
                hidden_states_orig = all_hidden_states[layer_ind]
                with torch.no_grad():
                    temp = model.transformer.ln_f(hidden_states_orig)

                    # f_layer = model.transformer.h[layer_ind]
                    # # attn
                    # residual = hidden_states_orig
                    # hidden_states1 = f_layer.ln_1(hidden_states_orig)
                    # attn_output = f_layer.attn(hidden_states1)[0] 
                    # hidden_states1 = attn_output + residual
                    # attn_temp = model.transformer.ln_f(hidden_states1)
                    # # mlp
                    # residual = hidden_states1
                    # hidden_states1 = f_layer.ln_2(hidden_states1)
                    # feed_forward_hidden_states = f_layer.mlp.c_proj(f_layer.mlp.act(f_layer.mlp.c_fc(hidden_states1)))
                    # hidden_states1 = residual + feed_forward_hidden_states
                    # mlp_temp = model.transformer.ln_f(hidden_states1)

                # (h,r1,b)  (b,r2,t) -- Pre (h,r1,r2,t)  Now (h,r1,r2,b,t)
                res_dict['b_rank_pos1_'+str(layer_ind)] = return_rank(temp[0, :, :], word_embedding, tokenizer("<"+b+">")['input_ids'][0])[1] 
                res_dict['b_rank_pos2_'+str(layer_ind)] = return_rank(temp[0, :, :], word_embedding, tokenizer("<"+b+">")['input_ids'][0])[2]
                res_dict['r2_rank_pos2_'+str(layer_ind)] = return_rank(temp[0, :, :], word_embedding, tokenizer("<"+r+">")['input_ids'][0])[2]


                # res_dict['attn_b_rank_pos1_'+str(layer_ind)] = return_rank(attn_temp[0, :, :], word_embedding, tokenizer("<"+b+">")['input_ids'][0])[1] 
                # res_dict['mlp_b_rank_pos1_'+str(layer_ind)] = return_rank(mlp_temp[0, :, :], word_embedding, tokenizer("<"+b+">")['input_ids'][0])[1] 
                # res_dict['attn_b_rank_pos2_'+str(layer_ind)] = return_rank(attn_temp[0, :, :], word_embedding, tokenizer("<"+b+">")['input_ids'][0])[2]
                # res_dict['mlp_b_rank_pos2_'+str(layer_ind)] = return_rank(mlp_temp[0, :, :], word_embedding, tokenizer("<"+b+">")['input_ids'][0])[2]

            # perturb the head entity
            all_h = set()
            assert (h, n_r) in atomic_dict
            for head in h2rt_train:
                if (head, n_r) not in atomic_dict:
                    all_h.add(head)
                    continue
                tail = atomic_dict[(head, n_r)]
                if (tail, r) not in atomic_dict or atomic_dict[(tail, r)] != t:
                    all_h.add(head)
            query = "<{}>".format(random.choice(list(all_h)))

            decoder_temp = tokenizer([query], return_tensors="pt", padding=True)
            decoder_input_ids_, decoder_attention_mask = decoder_temp["input_ids"], decoder_temp["attention_mask"]
            decoder_input_ids_, decoder_attention_mask = decoder_input_ids_.to(device), decoder_attention_mask.to(device)

            with torch.no_grad():
                outputs_ctft = model(
                    input_ids=decoder_input_ids_,
                    attention_mask=decoder_attention_mask,
                    output_hidden_states=True
                )
            all_hidden_states_ctft = outputs_ctft['hidden_states']

            for layer_to_intervene in range(0, target_layer):
                hidden_states = all_hidden_states[layer_to_intervene].clone()
                hidden_states_ctft = all_hidden_states_ctft[layer_to_intervene]
                # intervene
                hidden_states[0, 0, :] = hidden_states_ctft[0, 0, :]

                with torch.no_grad():
                    for i in range(layer_to_intervene, target_layer):
                        f_layer = model.transformer.h[i]
                        # attn
                        residual = hidden_states
                        hidden_states = f_layer.ln_1(hidden_states)
                        attn_output = f_layer.attn(hidden_states)[0] 
                        hidden_states = attn_output + residual
                        # mlp
                        residual = hidden_states
                        hidden_states = f_layer.ln_2(hidden_states)
                        feed_forward_hidden_states = f_layer.mlp.c_proj(f_layer.mlp.act(f_layer.mlp.c_fc(hidden_states)))
                        hidden_states = residual + feed_forward_hidden_states
                    # final ln
                    hidden_states = model.transformer.ln_f(hidden_states)
                # print("--------")
                rank_after = return_rank(hidden_states[0, :, :], word_embedding, tokenizer("<"+b+">")['input_ids'][0])[-1]
                res_dict['b_h_'+str(layer_to_intervene)] = rank_after
            
            # perturb the 1st relation
            all_r1 = set()
            rt_list = h2rt_train[h]
            for (relation, tail) in rt_list:
                if (tail, r) not in atomic_dict or atomic_dict[(tail, r)] != t:
                    all_r1.add(relation)
            query = "<{}><{}>".format(h, random.choice(list(all_r1)))

            decoder_temp = tokenizer([query], return_tensors="pt", padding=True)
            decoder_input_ids_, decoder_attention_mask = decoder_temp["input_ids"], decoder_temp["attention_mask"]
            decoder_input_ids_, decoder_attention_mask = decoder_input_ids_.to(device), decoder_attention_mask.to(device)
            with torch.no_grad():
                outputs_ctft = model(
                    input_ids=decoder_input_ids_,
                    attention_mask=decoder_attention_mask,
                    output_hidden_states=True
                )
            all_hidden_states_ctft = outputs_ctft['hidden_states']

            for layer_to_intervene in range(0, target_layer):
                hidden_states = all_hidden_states[layer_to_intervene].clone()
                hidden_states_ctft = all_hidden_states_ctft[layer_to_intervene]
                # intervene
                hidden_states[0, 1, :] = hidden_states_ctft[0, 1, :]

                with torch.no_grad():
                    for i in range(layer_to_intervene, target_layer):
                        f_layer = model.transformer.h[i]
                        # attn
                        residual = hidden_states
                        hidden_states = f_layer.ln_1(hidden_states)
                        attn_output = f_layer.attn(hidden_states)[0] 
                        hidden_states = attn_output + residual
                        # mlp
                        residual = hidden_states
                        hidden_states = f_layer.ln_2(hidden_states)
                        feed_forward_hidden_states = f_layer.mlp.c_proj(f_layer.mlp.act(f_layer.mlp.c_fc(hidden_states)))
                        hidden_states = residual + feed_forward_hidden_states
                    # final ln
                    hidden_states = model.transformer.ln_f(hidden_states)
                # print("--------")
                rank_after = return_rank(hidden_states[0, :, :], word_embedding, tokenizer("<"+b+">")['input_ids'][0])[-1]
                res_dict['b_r1_'+str(layer_to_intervene)] = rank_after
         

            # perturb the second relation
            all_r2 = set()
            rt_list = h2rt_train[b]
            for (relation, tail) in rt_list:
                if tail != t:
                    assert relation != r
                    all_r2.add(relation)
            query = "<{}><{}><{}>".format(h, n_r, random.choice(list(all_r2)))
            query = "<{}><{}><{}>".format(h, random.choice(list(all_r1)), random.choice(list(all_r2)))

            decoder_temp = tokenizer([query], return_tensors="pt", padding=True)
            decoder_input_ids_, decoder_attention_mask = decoder_temp["input_ids"], decoder_temp["attention_mask"]
            decoder_input_ids_, decoder_attention_mask = decoder_input_ids_.to(device), decoder_attention_mask.to(device)

            with torch.no_grad():
                outputs_ctft = model(
                    input_ids=decoder_input_ids_,
                    attention_mask=decoder_attention_mask,
                    output_hidden_states=True
                )
            all_hidden_states_ctft = outputs_ctft['hidden_states']

            for layer_to_intervene in range(0, target_layer):
                hidden_states = all_hidden_states[layer_to_intervene].clone()
                hidden_states_ctft = all_hidden_states_ctft[layer_to_intervene]
                # intervene
                hidden_states[0, 2, :] = hidden_states_ctft[0, 2, :]

                with torch.no_grad():
                    for i in range(layer_to_intervene, target_layer):
                        f_layer = model.transformer.h[i]
                        # attn
                        residual = hidden_states
                        hidden_states = f_layer.ln_1(hidden_states)
                        attn_output = f_layer.attn(hidden_states)[0] 
                        hidden_states = attn_output + residual
                        # mlp
                        residual = hidden_states
                        hidden_states = f_layer.ln_2(hidden_states)
                        feed_forward_hidden_states = f_layer.mlp.c_proj(f_layer.mlp.act(f_layer.mlp.c_fc(hidden_states)))
                        hidden_states = residual + feed_forward_hidden_states
                    # final ln
                    hidden_states = model.transformer.ln_f(hidden_states)
                # print("--------")
                rank_after = return_rank(hidden_states[0, :, :], word_embedding, tokenizer("<"+b+">")['input_ids'][0])[-1]
                res_dict['b_r2_'+str(layer_to_intervene)] = rank_after

            full_list.append(res_dict)

        results.append(full_list)

    with open(args.save_path, "w", encoding='utf-8') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()


