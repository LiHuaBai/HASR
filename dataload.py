import random
import json
import numpy as np

import torch
from torch.utils.data import Dataset

from scipy.sparse import csr_matrix


def generate_rating_matrix(Uses_dict, num_users, num_items,max_ses_len,max_seq_len, data_type = 'valid'):
    # three lists are used to construct sparse matrix
    if data_type == 'valid':
        en = -2
    elif data_type == 'test':
        en = -1

    row = []
    col = []
    data = []

    for uid in Uses_dict:
        u_ses = Uses_dict[uid][-max_ses_len:]
        u_seq = u_ses[-1]
        seq = u_seq[-max_seq_len:]
        for item in seq[:en]:
            row.append(uid)
            col.append(item)
            data.append(1)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    rating_matrix = csr_matrix((data, (row, col)), shape=(num_users, num_items))

    return rating_matrix

def get_user_sessions(datafile, max_ses_len, max_seq_len):
    Uses_dict = json.loads(open(datafile).readline())
    num_users = len(Uses_dict)
    data_UserSessions = np.zeros((num_users, max_ses_len, max_seq_len), dtype='int32')
    # 注意uid从0开始,item的id从1开始

    cnt = 0
    for uid in Uses_dict:
        cnt += 1
        if cnt%1000==0:
            print("cnt = ",cnt)
        u_ses = Uses_dict[uid][-max_ses_len:]
        ses_L = len(u_ses)
        st_i = 0
        if ses_L < max_ses_len:
            st_i = max_ses_len-ses_L
        for i in range(ses_L):
            u_seq = u_ses[i][-max_seq_len:]
            seq_len = len(u_seq)
            st_j = max_seq_len-seq_len
            for j in range(seq_len):
                data_UserSessions[int(uid),st_i+i,st_j+j] = u_seq[j]

    max_item = 224185  # Tmall16
    # max_item = 10802  # LastFM
    # max_item = 168254  # Avito

    print("max item number: ",max_item)
    print("max user number: ",num_users)
    print('Shape of user data UserSessions:', data_UserSessions.shape)

    valid_rating_matrix = generate_rating_matrix(Uses_dict, num_users, max_item+2,max_ses_len,max_seq_len, data_type = 'valid')
    test_rating_matrix = generate_rating_matrix(Uses_dict, num_users, max_item+2,max_ses_len,max_seq_len, data_type = 'test')
    return data_UserSessions, max_item, valid_rating_matrix, test_rating_matrix


def neg_sample(item_set, item_size):  # 前闭后闭，本身item_size = max_item+2,
    item = random.randint(1, item_size - 1)
    while item in item_set:
        item = random.randint(1, item_size - 1)
    return item


class HASRDataset(Dataset):
    def __init__(self, args, user_ses, data_type='train'):
        self.args = args
        self.user_ses = user_ses
        self.data_type = data_type
        self.max_ses_len = args.max_session_length
        self.max_seq_len = args.max_seq_length

    def __getitem__(self, index):

        user_id = index

        sessions = self.user_ses[index]

        assert self.data_type in {"train", "valid", "test"}

        if self.data_type == "train":
            input_ids = sessions[:,:-3]
            # [L1,L2]
            target_pos = sessions[:,1:-2]
            # [L1,L2]
            answer = [0] # no use

        elif self.data_type == 'valid':
            input_ids = sessions[:,1:-2]
            # [L1,L2]
            target_pos = sessions[:,2:-1] # no use
            # [L1,L2]
            answer = [sessions[-1,-2]]
            # [1]
        else:
            input_ids = sessions[:,2:-1]
            # [L1,L2]
            target_pos = sessions[:,3:] # no use
            # [L1,L2]
            answer = [sessions[-1,-1]]
            # [1]

        rlen, clen = input_ids.shape
        target_neg = np.zeros(target_pos.shape, dtype='int32')
        item_set = set()
        for i in range(rlen):
            item_set = item_set | set(input_ids[i])
            for j in range(clen):
                if target_pos[i][j]!=0:
                    target_neg[i][j] = neg_sample(item_set, self.args.item_size)

        assert input_ids.shape == (self.max_ses_len, self.max_seq_len)
        assert target_pos.shape == (self.max_ses_len, self.max_seq_len)
        assert target_neg.shape == (self.max_ses_len, self.max_seq_len)

        cur_tensors = (
                       torch.tensor(user_id, dtype=torch.long),
                       torch.tensor(input_ids, dtype=torch.long),
                       torch.tensor(target_pos, dtype=torch.long),
                       torch.tensor(target_neg, dtype=torch.long),
                       torch.tensor(answer, dtype=torch.long))
        return cur_tensors

    def __len__(self):
        return len(self.user_ses)