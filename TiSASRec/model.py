import numpy as np
import torch
import sys

FLOAT_MIN = -sys.float_info.max

class PointWiseFFN(torch.nn.Module):
    def __init__(self,hidden_unit,dropout_rate):
        super(PointWiseFFN,self).__init__()

        self.Linear1_conv1=torch.nn.Conv1d(hidden_unit,hidden_unit,kernel_size=1)
        self.dropout1=torch.nn.Dropout(p=dropout_rate)
        self.relu=torch.nn.ReLU()
        self.Linear2_conv2 = torch.nn.Conv1d(hidden_unit, hidden_unit, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self,z):
        outputs=z.transpose(-1,-2)
        outputs=self.Linear1_conv1(outputs)
        outputs=self.dropout1(outputs)
        outputs=self.relu(outputs)
        outputs=self.Linear2_conv2(outputs)
        outputs=self.dropout2(outputs)

        outputs=outputs.transpose(-1,-2)
        outputs+=z

        return outputs

class TimeAwareSelfAttention(torch.nn.Module):
    def __init__(self,hidden_unit,dropout_rate,head_num,dev):
        super(TimeAwareSelfAttention,self).__init__()
        self.Q_w=torch.nn.Linear(hidden_unit,hidden_unit)
        self.K_w = torch.nn.Linear(hidden_unit, hidden_unit)
        self.V_w = torch.nn.Linear(hidden_unit, hidden_unit)

        self.dropout=torch.nn.Dropout(p=dropout_rate)
        self.softmax=torch.nn.Softmax(dim=-1)

        self.head_num=head_num
        self.after_head_size=hidden_unit // head_num
        self.dropout_rate=dropout_rate
        self.dev=dev
    def forward(self,queries,keys,values,time_matrix_K,time_matrix_V,abs_pos_K,abs_pos_V,time_mask,future_time_mask):
        Q=self.Q_w(queries)
        K =self.K_w(keys)
        V=self.V_w(values)

        cut_Q = torch.split(Q, self.after_head_size, dim=-1)
        cut_K = torch.split(K, self.after_head_size, dim=-1)
        cut_V = torch.split(V, self.after_head_size, dim=-1)
        cut_time_V = torch.split(time_matrix_V, self.after_head_size, dim=-1)
        cut_time_K = torch.split(time_matrix_K, self.after_head_size, dim=-1)
        cut_abs_timeK=torch.split(abs_pos_K,self.after_head_size,dim=-1)
        cut_abs_timeV = torch.split(abs_pos_V, self.after_head_size, dim=-1)

        multi_Q = torch.cat(cut_Q,dim=0)
        multi_K = torch.cat(cut_K, dim=0)
        multi_V = torch.cat(cut_V, dim=0)
        multi_time_K=torch.cat(cut_time_K,dim=0)
        multi_time_V = torch.cat(cut_time_V, dim=0)
        multi_abs_timeK=torch.cat(cut_abs_timeK,dim=0)
        multi_abs_timeV = torch.cat(cut_abs_timeV, dim=0)

        multi_K_T=torch.transpose(multi_K,1,2)
        multi_abs_timeK_T=torch.transpose(multi_abs_timeK,1,2)
        raw_attention_weight = multi_Q.matmul(multi_K_T)
        raw_attention_weight += multi_Q.matmul(multi_abs_timeK_T)
        raw_attention_weight += multi_time_K.matmul(multi_Q.unsqueeze(-1)).squeeze(-1)
        raw_attention_weight = raw_attention_weight/(self.after_head_size**0.5)

        #其实有点不太能理解
        #time_mask.shape=[batch_size,seq_len]
        #与其叫time_mask不如叫padding_mask 实际上是mask掉padding部分
        #if true 变为最小值 在softmax的时候就会变成0
        time_mask = time_mask.unsqueeze(-1).repeat(self.head_num, 1, 1)
        time_mask = time_mask.expand(-1, -1, raw_attention_weight.shape[-1])
        future_time_mask=future_time_mask.unsqueeze(0).repeat(raw_attention_weight.shape[0],1,1)
        padding_tensor=torch.ones(raw_attention_weight.shape)*(-2**32+1)

        #放到device里面是有什么作用吗
        padding_tensor=padding_tensor.to(self.dev)

        raw_attention_weight = torch.where(time_mask,padding_tensor,raw_attention_weight)
        raw_attention_weight = torch.where(future_time_mask, padding_tensor, raw_attention_weight)

        attention_weight=self.softmax(raw_attention_weight)
        attention_weight=self.dropout(attention_weight)

        output=attention_weight.matmul(multi_V)
        output+=attention_weight.matmul(abs_pos_V)
        output+=attention_weight.unsqueeze(2).matmul(multi_time_V).squeeze(2)
        #为什么要reshape？？

        #还原多头注意力原本的维度
        output =torch.split(output,Q.shape[0],dim=0)
        output=torch.cat(output,dim=2)

        return output

class TiSASRec(torch.nn.Module):
    def __init__(self,user_num,item_num,time_span,args):
        super(TiSASRec,self).__init__()

        self.user_num=user_num
        self.item_num=item_num
        self.time_span=time_span
        self.dev=args.device

        #实际上是一个embedding查找工作 +1为了padding
        self.item_emb = torch.nn.Embedding(self.item_num+1,args.hidden_units,padding_idx=0)
        self.item_emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.abs_time_K_emb=torch.nn.Embedding(args.maxlen,args.hidden_units)
        self.abs_time_V_emb = torch.nn.Embedding(args.maxlen, args.hidden_units)
        self.time_matrix_K_emb = torch.nn.Embedding(args.time_span+1,args.hidden_units)
        self.time_matrix_V_emb = torch.nn.Embedding(args.time_span+1, args.hidden_units)
        #time_span 相对最大时间间隔 这是已经经过个性化处理之后的
        #我真不知道什么时候要用到dropout啊……

        self.abs_time_K_emb_dropout=torch.nn.Dropout(p=args.dropout_rate)
        self.abs_time_V_emb_dropout=torch.nn.Dropout(p=args.dropout_rate)
        self.time_matrix_K_dropout = torch.nn.Dropout(p=args.dropout_rate)
        self.time_matrix_V_dropout = torch.nn.Dropout(p=args.dropout_rate)

        #终于开始模型训练的过程了！
        #一层一层叠layernorm+attention+layernorm+ffn->循环
        self.attention_layernorm = torch.nn.ModuleList()
        self.attention_layers = torch.nn.ModuleList()
        self.ffn_layernorm = torch.nn.ModuleList()
        self.ffn_layers = torch.nn.ModuleList()

        #啊最后还有一个layernorm 我不懂了
        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units,eps=1e-8)

        #对于每一个block
        #ps:只是append层进去 并没有运行 所以也不带dropout？
        for _ in range(args.num_blocks):
            attention_layernorms_new = torch.nn.LayerNorm(args.hidden_units,eps=1e-8)
            self.attention_layernorm.append(attention_layernorms_new)

            attention_layer_new=TimeAwareSelfAttention(args.hidden_units,
                                                       args.dropout_rate,
                                                       args.num_heads,
                                                       args.device)
            self.attention_layers.append(attention_layer_new)

            ffn_layernorms_new = torch.nn.LayerNorm(args.hidden_units,eps=1e-8)
            self.ffn_layernorm.append(ffn_layernorms_new)

            ffn_layer_new = PointWiseFFN(args.hidden_units,
                                         args.dropout_rate)
            self.ffn_layers.append(ffn_layer_new)
    #对数据进行处理 全部变成emb
    def seq2feature(self,user_id,log_seqs,time_matrix):
        #一定要转换为Longtensor 原来可能不是long tensor类型
        seqs=self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
        #attention is all your need里面的原做法
        seqs *= self.item_emb.embedding_dim ** 0.5
        seqs=self.item_emb_dropout(seqs)

        #绝对位置嵌入 用np.array弄出绝对位置的索引[0，1，2，……，maxlen]
        #做的第一个改动
        all_batch_abs_position_index = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        all_batch_abs_position_index = torch.LongTensor(all_batch_abs_position_index).to(self.dev)
       
        abs_pos_K=self.abs_time_K_emb(all_batch_abs_position_index)
        abs_pos_V = self.abs_time_V_emb(all_batch_abs_position_index)
        abs_pos_K= self.abs_time_K_emb_dropout(abs_pos_K)
        abs_pos_V = self.abs_time_V_emb_dropout(abs_pos_V)

        time_matrix = torch.LongTensor(time_matrix).to(self.dev)
        time_matrix_K = self.time_matrix_K_emb(time_matrix)
        time_matrix_V = self.time_matrix_V_emb(time_matrix)
        time_matrix_K = self.time_matrix_K_dropout(time_matrix_K)
        time_matrix_V = self.time_matrix_V_dropout(time_matrix_V)

        #mask掉padding 即seqs中为0的部分
        #log_seqs.shape=[batch_size,maxlen] 所以将所有等于0的部分都变成true
        #在item和id映射成数字的时候应该有提到 padding部分已经变成0 指定padding_idx=0 不会被更新 那为什么还要再mask一次呢
        padding_mask = torch.BoolTensor(log_seqs==0).to(self.dev)
        #先取反再加层
        seqs *=~padding_mask.unsqueeze(-1)
        #有自动broadcast的功能
        #tensor的*是每个同样位置的元素分别相乘

        maxlen=seqs.shape[1]
        attention_mask = ~torch.tril(torch.ones((maxlen,maxlen),dtype=torch.bool,device=self.dev))
        #取反是因为在timeawareattention里面true表示需要被隐瞒

        #正经开始进行模型训练部分
        for i in range(len(self.attention_layers)):
            Q = self.attention_layernorm[i](seqs)
            #为什么只对q做layernorm 论文中也没有提到 这是特殊处理吗
            attention_output = self.attention_layers[i](Q,
                                             seqs,seqs,
                                            time_matrix_K,
                                            time_matrix_V,
                                            abs_pos_K,
                                            abs_pos_V,
                                            padding_mask,
                                            attention_mask)
            seqs = Q + attention_output
            seqs=self.ffn_layernorm[i](seqs)
            seqs=self.ffn_layers[i](seqs)
            seqs*=~padding_mask.unsqueeze(-1)
            #padding始终置0
            #所以更新之后有可能padding位置不为0了？
            # 为啥啊 感觉完全就只有0的可能啊

        final_output=self.last_layernorm(seqs)
        return final_output
    def forward(self,user_id,log_seqs,time_matrix,pos_seq,neg_seq):
        final_output = self.seq2feature(user_id,log_seqs,time_matrix)

        pos_seq_emb = self.item_emb(torch.LongTensor(pos_seq).to(self.dev))
        neg_seq_emb = self.item_emb(torch.LongTensor(neg_seq).to(self.dev))
        #pos_seq_emb.shape=[batch_size,seq_len,hidden_unit]
        #我理解 final_output里面每个item都是预测出来的 pos_item就是本身应该得到的序列
        #final_output.shape=[batch_size,seq_len,hidden_unit]
        #用点积来确认每一个物品的得分
        pos_logits=(final_output*pos_seq_emb).sum(dim=-1)
        neg_logits=(final_output*neg_seq_emb).sum(dim=-1)

        return pos_logits,neg_logits

    def predict(self,user_id,log_seqs,time_matrix,all_item_num):
        log_feats = self.seq2feature(user_id, log_seqs, time_matrix)
        final_feat = log_feats[:, -1, :]
        #只用到每个batch的seq的最后一个 即最后一份预测的预测数据
        #all_item_num的大小应该是[batch_size,item_num]
        all_item_emb = self.item_emb(torch.LongTensor(all_item_num).to(self.dev))
        all_item_logits = all_item_emb.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        return all_item_logits
    #得到的就是[batch_size,item_num]




