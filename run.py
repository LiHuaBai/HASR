# -*- coding: utf-8 -*-
# @Time    : 2020/4/25 22:59
# @Author  : Hui Wang

import numpy as np
import random
import torch
import argparse
import tqdm

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.optim import Adam

from dataload import get_user_sessions, HASRDataset
from HASR import HASR
from modules import recall_at_k, ndcg_k, EarlyStopping


def get_scores(epoch, answers, pred_list, log_file):
    recall, ndcg = [], []
    for k in [5, 10, 20]:
        recall.append(recall_at_k(answers, pred_list, k))
        ndcg.append(ndcg_k(answers, pred_list, k))

    post_fix = {
        "Epoch": epoch,
        "HIT@5": '{:.4f}'.format(recall[0]), "NDCG@5": '{:.4f}'.format(ndcg[0]),
        "HIT@10": '{:.4f}'.format(recall[1]), "NDCG@10": '{:.4f}'.format(ndcg[1]),
        "HIT@20": '{:.4f}'.format(recall[2]), "NDCG@20": '{:.4f}'.format(ndcg[2]),
        # "HIT@40": '{:.4f}'.format(recall[3]), "NDCG@40": '{:.4f}'.format(ndcg[3])
        # "HIT@50": '{:.4f}'.format(recall[4]), "NDCG@50": '{:.4f}'.format(ndcg[4])
    }
    print(post_fix)
    with open(log_file, 'a') as f:
        f.write(str(post_fix) + '\n')
    return [recall[0], ndcg[0], recall[1], ndcg[1], recall[2], ndcg[2]]

def eval(rating_pred,batch_user_index,valid_rating_matrix):
    rating_pred[valid_rating_matrix[batch_user_index].toarray() > 0] = 0
    # 加负号"-"表示取大的值
    ind = np.argpartition(rating_pred, -20)[:, -20:]
    arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
    arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]
    batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]
    return batch_pred_list


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', default='./data/', type=str)
    parser.add_argument('--output_dir', default='output/', type=str)
    parser.add_argument('--save_dir', default='save/', type=str)
    parser.add_argument('--data_name', default='Tmall16', type=str)

    # model args
    parser.add_argument("--hidden_size", type=int, default=64, help="hidden size of transformer model")
    parser.add_argument('--num_attention_heads', default=2, type=int)
    parser.add_argument('--hidden_act', default="gelu", type=str)  # gelu relu
    parser.add_argument("--attention_probs_dropout_prob", type=float, default=0.5, help="attention dropout p")
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.5, help="hidden dropout p")
    parser.add_argument("--initializer_range", type=float, default=0.02)
    parser.add_argument('--max_seq_length', default=32, type=int, help="44 for LastFM, 50 for Avito, 32 for Tmall")
    parser.add_argument('--max_session_length', default=12, type=int, help="8 for LastFM, Avito, 12 for Tmall")

    # train args
    parser.add_argument("--lr", type=float, default=0.0003, help="learning rate of adam")
    parser.add_argument("--batch_size", type=int, default=64, help="number of batch_size")
    parser.add_argument("--epochs", type=int, default=511, help="number of epochs")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--eval_freq", type=int, default=10)
    parser.add_argument("--seed", default=77, type=int)

    parser.add_argument("--weight_decay", type=float, default=0, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam second beta value")
    parser.add_argument("--gpu_id", type=str, default="0", help="gpu_id")

    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda

    args.data_file = args.data_dir + args.data_name + '.json'

    data_UserSessions, max_item, valid_rating_matrix, test_rating_matrix = get_user_sessions(args.data_file, args.max_session_length, args.max_seq_length+3)

    print("dataload finish")
    args.user_size = data_UserSessions.shape[0]
    args.item_size = max_item + 2

    train_dataset = HASRDataset(args, data_UserSessions, data_type='train')
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size)
    # sampler 定义从数据集中抽取样本的策略,如果指定，则shuffle不得指定,shuffle = True, 每个epoch重新随即取

    eval_dataset = HASRDataset(args, data_UserSessions, data_type='valid')
    eval_sampler = SequentialSampler(eval_dataset)
    # 顺序采样
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.batch_size)

    test_dataset = HASRDataset(args, data_UserSessions, data_type='test')
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size)

    model = HASR(args=args)
    model.cuda()

    args.item_neibors = " "

    print("model finish")

    args.output_file = args.output_dir + args.data_name + '.txt'
    args.save_file = args.save_dir + args.data_name + '_best.pt'

    print(str(args))
    with open(args.output_file, 'a') as f:
        f.write(str(args) + '\n')

    betas = (args.adam_beta1, args.adam_beta2)
    optim = Adam(model.parameters(), lr=args.lr, betas=betas, weight_decay=args.weight_decay)

    device = torch.device("cuda")
    # model.load_state_dict(torch.load(args.save_file))
    early_stopping = EarlyStopping(args.save_file, patience=10, verbose=True)
    for epoch in range(args.epochs):
        str_code = "train"
        dataloader = train_dataloader
        rec_data_iter = tqdm.tqdm(enumerate(dataloader),
                                  desc="Recommendation EP_%s:%d" % (str_code, epoch),
                                  total=len(dataloader),
                                  bar_format="{l_bar}{r_bar}")

        model.train()

        rec_avg_loss = 0.0
        rec_cur_loss = 0.0 

        for i, batch in rec_data_iter:
            batch = tuple(t.to(device) for t in batch)
            user_ids, input_ids, target_pos, target_neg, answers = batch

            h_u, session_sum = model.forward(input_ids,user_ids)
            loss = model.cross_entropy(h_u,session_sum, target_pos, target_neg, input_ids)

            optim.zero_grad()
            loss.backward()  # 修改loss
            optim.step()

            rec_avg_loss += loss.item()
            rec_cur_loss = loss.item()

        post_fix = {
            "epoch": epoch,
            "rec_avg_loss": '{:.4f}'.format(rec_avg_loss / len(rec_data_iter)),
            "rec_cur_loss": '{:.4f}'.format(rec_cur_loss),
        }
        print(str(post_fix))

        if epoch%args.eval_freq==0:
            model.eval()
            pred_list = None
            answer_list = None
            str_code = "valid"
            dataloader = eval_dataloader
            rec_data_iter = tqdm.tqdm(enumerate(dataloader),
                                      desc="Recommendation EP_%s:%d" % (str_code, epoch),
                                      total=len(dataloader),
                                      bar_format="{l_bar}{r_bar}")

            for i, batch in rec_data_iter:
                batch = tuple(t.to(device) for t in batch)
                user_ids, input_ids, target_pos, target_neg, answers = batch

                h_u, session_sum = model.forward(input_ids,user_ids)

                rating_pred = model.predict(h_u,session_sum)
                rating_pred = rating_pred.cpu().data.numpy().copy()

                batch_user_index = user_ids.cpu().numpy()

                batch_pred_list = eval(rating_pred,batch_user_index,valid_rating_matrix)

                if i == 0:
                    pred_list = batch_pred_list
                    answer_list = answers.cpu().data.numpy()
                else:
                    pred_list = np.append(pred_list, batch_pred_list, axis=0)
                    answer_list = np.append(answer_list, answers.cpu().data.numpy(), axis=0)

            score = get_scores(epoch, answer_list, pred_list, args.output_file)
            early_stopping(np.array(score), model, epoch)

        if early_stopping.early_stop:
            print("Early stopping")
            print("best_epoch = ", early_stopping.best_epoch)
            break

    print("best_epoch = ", early_stopping.best_epoch)
    print('---------------Change to test_rating_matrix!-------------------')
    # load the best model
    model.load_state_dict(torch.load(args.save_file))

    model.eval()
    pred_list = None
    answer_list = None
    epoch = early_stopping.best_epoch
    # epoch = 0
    str_code = "test"
    dataloader = test_dataloader
    rec_data_iter = tqdm.tqdm(enumerate(dataloader),
                              desc="Recommendation EP_%s:%d" % (str_code, epoch),
                              total=len(dataloader),
                              bar_format="{l_bar}{r_bar}")

    for i, batch in rec_data_iter:
        batch = tuple(t.to(device) for t in batch)
        user_ids, input_ids, target_pos, target_neg, answers = batch
        h_u, session_sum = model.forward(input_ids,user_ids)

        rating_pred = model.predict(h_u,session_sum)
        rating_pred = rating_pred.cpu().data.numpy().copy()

        batch_user_index = user_ids.cpu().numpy()
        batch_pred_list = eval(rating_pred,batch_user_index,test_rating_matrix)

        if i == 0:
            pred_list = batch_pred_list
            answer_list = answers.cpu().data.numpy()
        else:
            pred_list = np.append(pred_list, batch_pred_list, axis=0)
            answer_list = np.append(answer_list, answers.cpu().data.numpy(), axis=0)

    score = get_scores(epoch, answer_list, pred_list, args.output_file)


import os
main()