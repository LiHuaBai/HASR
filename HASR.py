
import torch
import torch.nn as nn

# import torchsnooper
from modules import GPAttention, TransformerLayer, LayerNorm

import numpy as np


class HASR(nn.Module):
    def __init__(self, args):
        super(HASR, self).__init__()
        self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
        self.outitem_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)
        self.user_embeddings = nn.Embedding(args.user_size, args.hidden_size)

        self.item_encoder = TransformerLayer(args)
        self.item_decoder = TransformerLayer(args)
        self.GP_encoder = GPAttention(args)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.args = args

        self.PCG_linear = nn.Linear(args.hidden_size*2, args.hidden_size)
        self.PCG_dropout = nn.Dropout(args.attention_probs_dropout_prob)
        self.PCG_linear2 = nn.Linear(args.hidden_size, 1)

        self.GP_index = self.get_GP_index(self.args.max_session_length,self.args.max_seq_length)
        self.GP_mask = self.get_GP_mask(self.args.max_session_length,self.args.max_seq_length)

        self.apply(self.init_weights)

    def get_GP_mask(self, max_ses_len, max_seq_len):
        GP_mask = np.zeros([max_ses_len*max_seq_len,max_ses_len])

        row_mask = np.zeros([max_ses_len])

        for rid in range(max_ses_len):
            row_mask[rid] = 1
            for vid in range(max_seq_len):
                GP_mask[rid*max_seq_len+vid]=row_mask

        return GP_mask


    def get_GP_index(self, max_ses_len, max_seq_len):
        total_index = np.zeros([max_ses_len*max_seq_len,max_ses_len])
        row_index = np.zeros([max_ses_len])
        for i in range(max_ses_len):
            row_index[i] = i*max_seq_len
            # [0,30,60,90,...,210]
        for rid in range(max_ses_len):
            for vid in range(max_seq_len):
                row_index[rid] = rid*max_seq_len+vid
                total_index[rid*max_seq_len+vid] = row_index

        return total_index

    def add_position_embedding(self, sequence_ids):

        seq_length = sequence_ids.size(2)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence_ids.device)
        position_ids = position_ids.unsqueeze(0).unsqueeze(0).expand_as(sequence_ids)

        item_embeddings = self.item_embeddings(sequence_ids)
        position_embeddings = self.position_embeddings(position_ids)

        sequence_emb = item_embeddings + position_embeddings
        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)

        return sequence_emb

    def consistency_gate(self, gp_emb, cp_emb):
        s_emb = torch.cat([cp_emb,gp_emb],dim=-1)

        s_emb = self.PCG_linear(s_emb)
        s_emb = self.PCG_dropout(s_emb)
        s_emb = self.PCG_linear2(s_emb)
        return nn.Sigmoid()(s_emb)

    def get_hu(self, item_ids, user_ids):

        attention_mask = (item_ids > 0).long()
        # [batch,8,50]
        extended_attention_mask = attention_mask.unsqueeze(2).unsqueeze(3)

        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)
        subsequent_mask = (subsequent_mask == 0).unsqueeze(0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long()
        if self.args.cuda_condition:
            subsequent_mask = subsequent_mask.cuda()

        subsequent_extended_attention_mask = extended_attention_mask * subsequent_mask
        subsequent_extended_attention_mask = subsequent_extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        subsequent_extended_attention_mask = (1.0 - subsequent_extended_attention_mask) * -10000.0


        sequence_emb = self.add_position_embedding(item_ids)

        item_encoded_emb = self.item_encoder(sequence_emb, subsequent_extended_attention_mask)
        item_encoded_emb = self.item_decoder(item_encoded_emb, subsequent_extended_attention_mask)

        encoded_emb = item_encoded_emb

        GP_encoder_index = torch.tensor(self.GP_index, dtype=torch.long).cuda()
        GP_subsession_mask = torch.tensor(self.GP_mask, dtype=torch.long).cuda()

        view_attention_mask = attention_mask.view(-1,self.args.max_session_length*self.args.max_seq_length)
        GP_extend_mask = view_attention_mask[:,GP_encoder_index]

        GP_extend_subsession_mask = GP_extend_mask*GP_subsession_mask


        GP_extend_subsession_mask = GP_extend_subsession_mask.to(dtype=next(self.parameters()).dtype)
        GP_extend_subsession_mask = (1.0 - GP_extend_subsession_mask) * -10000.0

        user_emb = self.user_embeddings(user_ids)
        GP_emb = self.GP_encoder(user_emb, encoded_emb, GP_extend_subsession_mask, GP_encoder_index)

        z = self.consistency_gate(GP_emb,encoded_emb)

        h = z*encoded_emb+(1-z)*GP_emb
        return h

    def get_session_score(self,item_ids):
        # item_embeddings = self.item_embeddings(item_ids)
        item_embeddings = self.outitem_embeddings(item_ids)
        sub_item_emb = item_embeddings.unsqueeze(2)

        pad_mask = (item_ids > 0).long()
        extended_pad_mask = pad_mask.unsqueeze(3).unsqueeze(2) 
        mask_shape = (self.args.max_seq_length, self.args.max_seq_length)
        sub_mask = torch.triu(torch.ones(mask_shape), diagonal=1)
        sub_mask = (sub_mask == 0).long()
        sub_mask = sub_mask.unsqueeze(2).unsqueeze(0).unsqueeze(0).cuda()
        sub_extend_mask = sub_mask*extended_pad_mask

        sub_extend_mask = sub_extend_mask.float()
        sub_item_emb = sub_item_emb*sub_extend_mask

        sum_item_emb = torch.sum(sub_item_emb,dim = -2)

        return sum_item_emb

    def forward(self,item_ids,user_ids):
        h_u = self.get_hu(item_ids,user_ids)
        session_sum = self.get_session_score(item_ids)
        return h_u, session_sum

    def cross_entropy(self, h_u, session_sum, pos_ids, neg_ids, input_ids):
        # pos_emb = self.item_embeddings(pos_ids)
        # neg_emb = self.item_embeddings(neg_ids)
        pos_emb = self.outitem_embeddings(pos_ids)
        neg_emb = self.outitem_embeddings(neg_ids)

        pos = pos_emb.view(-1, pos_emb.size(3))
        neg = neg_emb.view(-1, neg_emb.size(3))

        hu_emb = h_u.view(-1, self.args.hidden_size)
        ses_sum = session_sum.view(-1,session_sum.size(3))

        hu_pos = torch.sum(pos*hu_emb, -1)
        hu_neg = torch.sum(neg*hu_emb, -1)

        ses_pos = torch.sum(pos*ses_sum,-1)
        ses_neg = torch.sum(neg*ses_sum,-1)

        pos_logits = (hu_pos+ses_pos)/2
        neg_logits = (hu_neg+ses_neg)/2

        istarget = (input_ids > 0).view(input_ids.size(0) * self.args.max_session_length*self.args.max_seq_length).float()

        loss = torch.sum(
            - torch.log(torch.sigmoid(pos_logits) + 1e-24) * istarget -
            torch.log(1 - torch.sigmoid(neg_logits) + 1e-24) * istarget
        ) / torch.sum(istarget)

        # loss = torch.sum(- torch.log(torch.sigmoid(pos_logits - neg_logits) + 1e-24)* istarget)/ torch.sum(istarget)

        return loss

    def predict(self, h_u, session_sum):
        # test_item_emb = self.item_embeddings.weight
        test_item_emb = self.outitem_embeddings.weight

        last_hu = h_u[:,-1,-1,:]
        h_u_pred = torch.matmul(last_hu, test_item_emb.transpose(0, 1))

        last_ses_sum = session_sum[:,-1,-1,:]
        ses_pred = torch.matmul(last_ses_sum, test_item_emb.transpose(0, 1))

        rating_pred_sum = (h_u_pred+ses_pred)
        return rating_pred_sum

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()




