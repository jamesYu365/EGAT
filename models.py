import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphAttentionLayer


class GAT(nn.Module):
    def __init__(self, nfeat,ef_sz, nhid,nclass, dropout, alpha, nheads):
        """
        Dense version of GAT.
        nfeat输入节点的特征向量长度，标量
        ef_sz输入edge特征矩阵的大小，列表，PxNxN
        nhid隐藏节点的特征向量长度，标量
        nclass输出节点的特征向量长度，标量
        dropout：drpout的概率
        alpha：leakyrelu的第三象限斜率
        nheads：attention_head的个数
        """
        super(GAT, self).__init__()
        self.dropout = dropout
        
        #起始层
        self.attentions = [GraphAttentionLayer(nfeat, nhid[0], dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads[0])]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        
        # #hidden层
        # self.hidden_atts=[GraphAttentionLayer(nhid[0]*nheads[0]*ef_sz[0], nhid[1], dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads[1])]
        # for i, attention in enumerate(self.hidden_atts):
        #     self.add_module('hidden_att_{}'.format(i), attention)
        
        #输出层
        self.out_att = GraphAttentionLayer(nhid[0]*nheads[0]*ef_sz[0], nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, edge_attr):
        
        #起始层
        x = F.dropout(x, self.dropout, training=self.training)#起始层
        temp_x=[]
        for att in self.attentions:
            inn_x,edge_attr=att(x, edge_attr)
            temp_x.append(inn_x)
        x = torch.cat(temp_x, dim=1)#起始层
        
        # #中间层
        # x = F.dropout(x, self.dropout, training=self.training)#中间层
        # temp_x=[]
        # for att in self.hidden_atts:
        #     inn_x,edge_attr=att(x, edge_attr)
        #     temp_x.append(inn_x)
        # x = torch.cat(temp_x, dim=1)#中间层
        
        
        #输出层
        x = F.dropout(x, self.dropout, training=self.training)#输出层   
        x = F.elu(self.out_att(x, edge_attr))#输出层
        return F.log_softmax(x, dim=1)



