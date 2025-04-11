"""VSE modules"""

import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
import math
from transformers import BertModel
from lib.modules.attention_block import Block
from lib.modules.resnet import ResnetFeatureExtractor
from lib.modules.aggr.gpo import GPO, soft_max
# from lib.modules.attention.muti_head_attention import MultiHeadAttention, MyMultiHeadAttention
# from lib.modules.aggr.gpo import GPO

from lib.modules.mlp import MLP

import logging

logger = logging.getLogger(__name__)

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        # '-1' means last dimension.

        out = (x - mean) / (std + self.eps)
        out = self.gamma * out + self.beta
        return out

class ScaleDotProductAttention(nn.Module):
    """
    compute scale dot product attention

    Query : given sentence that we focused on (decoder)
    Key : every sentence to check relationship with Qeury(encoder)
    Value : every sentence same with Key (encoder)
    """

    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None, e=1e-12):
        # input is 4 dimension tensor
        # [batch_size, head, length, d_tensor]
        batch_size, head, length, d_tensor = k.size()

        # 1. dot product Query with Key^T to compute similarity
        k_t = k.transpose(2, 3)  # transpose
        score = (q @ k_t) / math.sqrt(d_tensor)  # scaled dot product

        # 2. apply masking (opt)
        if mask is not None:
            score = score.masked_fill(mask == 0, -e)

        # 3. pass them softmax to make [0, 1] range
        score = self.softmax(score)

        # 4. multiply with Value
        v = score @ v

        return v, score

class MyMultiHeadAttention(nn.Module):

    def __init__(self, d_model, n_head):
        super(MyMultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.attention = ScaleDotProductAttention()
        # self.w_q = nn.Linear(d_model, d_model)
        # self.w_k = nn.Linear(d_model, d_model)
        # self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)
        nn.init.xavier_normal_(self.w_concat.weight, gain=1.414)

    def forward(self, q, k, v, mask=None):
        # 1. dot product with weight matrices
        # q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)
        # TODO : input shape and output shape?
        # 2. split tensor by number of heads
        q, k, v = self.split(q), self.split(k), self.split(v)

        # 3. do scale dot product to compute similarity
        out, attention = self.attention(q, k, v, mask=mask)

        # 4. concat and pass to linear layer
        out = self.concat(out)
        out = self.w_concat(out)

        return out

    def split(self, tensor):
        """
        split tensor by number of head

        :param tensor: [batch_size, length, d_model]
        :return: [batch_size, head, length, d_tensor]
        """
        batch_size, length, d_model = tensor.size()

        d_tensor = d_model // self.n_head
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)
        # it is similar with group convolution (split by number of heads)

        return tensor

    def concat(self, tensor):
        """
        inverse function of self.split(tensor : torch.Tensor)

        :param tensor: [batch_size, head, length, d_tensor]
        :return: [batch_size, length, d_model]
        """
        batch_size, head, length, d_tensor = tensor.size()
        d_model = head * d_tensor

        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor

class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, n_head):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.attention = ScaleDotProductAttention()
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        # 1. dot product with weight matrices
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        # 2. split tensor by number of heads
        q, k, v = self.split(q), self.split(k), self.split(v)

        # 3. do scale dot product to compute similarity
        out, attention = self.attention(q, k, v, mask=mask)

        # 4. concat and pass to linear layer
        out = self.concat(out)
        out = self.w_concat(out)

        # 5. visualize attention map
        # TODO : we should implement visualization

        return out

    def split(self, tensor):
        """
        split tensor by number of head

        :param tensor: [batch_size, length, d_model]
        :return: [batch_size, head, length, d_tensor]
        """
        batch_size, length, d_model = tensor.size()

        d_tensor = d_model // self.n_head
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)
        # it is similar with group convolution (split by number of heads)

        return tensor

    def concat(self, tensor):
        """
        inverse function of self.split(tensor : torch.Tensor)

        :param tensor: [batch_size, head, length, d_tensor]
        :return: [batch_size, length, d_model]
        """
        batch_size, head, length, d_tensor = tensor.size()
        d_model = head * d_tensor

        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor

class Rs_GCN(nn.Module):

    def __init__(self, in_channels, inter_channels, bn_layer=True):
        super(Rs_GCN, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1


        conv_nd = nn.Conv1d
        max_pool = nn.MaxPool1d
        bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant(self.W[1].weight, 0)
            nn.init.constant(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant(self.W.weight, 0)
            nn.init.constant(self.W.bias, 0)

        self.theta = None
        self.phi = None


        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)




    def forward(self, v):
        '''
        :param v: (B, D, N)
        :return:
        '''
        batch_size = v.size(0)

        g_v = self.g(v).view(batch_size, self.inter_channels, -1)
        g_v = g_v.permute(0, 2, 1)

        theta_v = self.theta(v).view(batch_size, self.inter_channels, -1)
        theta_v = theta_v.permute(0, 2, 1)
        phi_v = self.phi(v).view(batch_size, self.inter_channels, -1)
        R = torch.matmul(theta_v, phi_v)
        N = R.size(-1)
        R_div_C = R / N

        y = torch.matmul(R_div_C, g_v)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *v.size()[2:])
        W_y = self.W(y)
        v_star = W_y + v

        return v_star

def l1norm(X, dim, eps=1e-8):
    """L1-normalize columns of X
    """
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X


def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


def maxk_pool1d_var(x, dim, k, lengths):
    results = list()
    lengths = list(lengths.cpu().numpy())
    lengths = [int(x) for x in lengths]
    for idx, length in enumerate(lengths):
        k = min(k, length)
        max_k_i = maxk(x[idx, :length, :], dim - 1, k).mean(dim - 1)
        results.append(max_k_i)
    results = torch.stack(results, dim=0)
    return results


def maxk_pool1d(x, dim, k):
    max_k = maxk(x, dim, k)
    return max_k.mean(dim)


def maxk(x, dim, k):
    index = x.topk(k, dim=dim)[1]
    return x.gather(dim, index)


def get_text_encoder(embed_size, no_txtnorm=False):
    return EncoderText(embed_size, no_txtnorm=no_txtnorm)


def get_image_encoder(data_name, img_dim, embed_size, precomp_enc_type='basic',
                      backbone_source=None, backbone_path=None, no_imgnorm=False):
    """A wrapper to image encoders. Chooses between an different encoders
    that uses precomputed image features.
    """
    if precomp_enc_type == 'basic':
        img_enc = EncoderImageAggr(
            img_dim, embed_size, precomp_enc_type, no_imgnorm)
    elif precomp_enc_type == 'backbone':
        backbone_cnn = ResnetFeatureExtractor(backbone_source, backbone_path, fixed_blocks=2)
        img_enc = EncoderImageFull(backbone_cnn, img_dim, embed_size, precomp_enc_type, no_imgnorm)
    else:
        raise ValueError("Unknown precomp_enc_type: {}".format(precomp_enc_type))

    return img_enc


class EncoderImageAggr(nn.Module):
    def __init__(self, img_dim, embed_size, precomp_enc_type='basic', no_imgnorm=False):
        super(EncoderImageAggr, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.fc = nn.Linear(img_dim, embed_size)
        self.precomp_enc_type = precomp_enc_type
        if precomp_enc_type == 'basic':
            self.mlp = MLP(img_dim, embed_size // 2, embed_size, 2)
        self.gpool = GPO(32, 32) #soft_max
        # self.gpool = soft_max(32, 32)
        # GCN reasoning 
        self.Rs_GCN_1 = Rs_GCN(in_channels=embed_size, inter_channels=embed_size)
        self.Rs_GCN_2 = Rs_GCN(in_channels=embed_size, inter_channels=embed_size)
        self.Rs_GCN_3 = Rs_GCN(in_channels=embed_size, inter_channels=embed_size)
        self.Rs_GCN_4 = Rs_GCN(in_channels=embed_size, inter_channels=embed_size)
        #self.soft = nn.Softmax(dim = 1)
        # self.attention = featureInteraction(embed_size)

        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images, image_lengths):
        """Extract image feature vectors."""
        features = self.fc(images)
        if self.precomp_enc_type == 'basic':
            # When using pre-extracted region features, add an extra MLP for the embedding transformation
            features = self.mlp(images) + features

        GCN_img_emd = features.permute(0, 2, 1)
        GCN_img_emd = self.Rs_GCN_1(GCN_img_emd)
        features4 = GCN_img_emd.permute(0, 2, 1)
        GCN_img_emd = self.Rs_GCN_2(GCN_img_emd)
        features3 = GCN_img_emd.permute(0, 2, 1)
        GCN_img_emd = self.Rs_GCN_3(GCN_img_emd)
        features2 = GCN_img_emd.permute(0, 2, 1)
        GCN_img_emd = self.Rs_GCN_4(GCN_img_emd)
        # -> B,N,D
        features1 = GCN_img_emd.permute(0, 2, 1)
        features1, pool_weights = self.gpool(features1, image_lengths)
        features2, pool_weights = self.gpool(features2, image_lengths)
        features3, pool_weights = self.gpool(features3, image_lengths)
        features4, pool_weights = self.gpool(features4, image_lengths)
        features_gcn = torch.cat([features1.unsqueeze(1), features2.unsqueeze(1),
                            features3.unsqueeze(1), features4.unsqueeze(1)], dim=1)
        # print(features)
        # features_gcn = self.attention(features_gcn)
        feature_mean = torch.mean(features_gcn, dim=1)
        # # print(feature_mean)
        # feature_var = ((features1 - feature_mean).pow(2) + (features2 - feature_mean).pow(2)\
        #               + (features3 - feature_mean).pow(2) + (features4 - feature_mean).pow(2)) / 4
        feature_var = (torch.mean((features_gcn - feature_mean.unsqueeze(1)).pow(2), dim=1) + 1e-8).sqrt()
        # print(feature_var)
        features_s = []
        num_sample = 4
        for i in range(num_sample):
            epsilon = torch.randn_like(feature_var)
            features_i = feature_mean + epsilon * feature_var
            features_s.append(features_i.unsqueeze(1))
        features = torch.cat(features_s, 1)
        # epsilon = torch.randn_like(feature_var)
        # features = feature_mean + epsilon * feature_var

        if not self.no_imgnorm:
            feature_mean = l2norm(feature_mean, dim=-1)
            feature_var = l2norm(feature_var, dim=-1)
            features = l2norm(features, dim=-1)
        # print((features - feature_mean.unsqueeze(1)).sum(dim=-1).softmax(dim=-1))

        return feature_mean, feature_var, features

class EncoderImageFull(nn.Module):
    def __init__(self, backbone_cnn, img_dim, embed_size, precomp_enc_type='basic', no_imgnorm=False):
        super(EncoderImageFull, self).__init__()
        self.backbone = backbone_cnn
        self.image_encoder = EncoderImageAggr(img_dim, embed_size, precomp_enc_type, no_imgnorm)
        self.backbone_freezed = False

    def forward(self, images):
        """Extract image feature vectors."""
        base_features = self.backbone(images)

        if self.training:
            # Size Augmentation during training, randomly drop grids
            base_length = base_features.size(1)
            features = []
            feat_lengths = []
            rand_list_1 = np.random.rand(base_features.size(0), base_features.size(1))
            rand_list_2 = np.random.rand(base_features.size(0))
            for i in range(base_features.size(0)):
                if rand_list_2[i] > 0.2:
                    feat_i = base_features[i][np.where(rand_list_1[i] > 0.20 * rand_list_2[i])]
                    len_i = len(feat_i)
                    pads_i = torch.zeros(base_length - len_i, base_features.size(-1)).to(base_features.device)
                    feat_i = torch.cat([feat_i, pads_i], dim=0)
                else:
                    feat_i = base_features[i]
                    len_i = base_length
                feat_lengths.append(len_i)
                features.append(feat_i)
            base_features = torch.stack(features, dim=0)
            base_features = base_features[:, :max(feat_lengths), :]
            feat_lengths = torch.tensor(feat_lengths).to(base_features.device)
        else:
            feat_lengths = torch.zeros(base_features.size(0)).to(base_features.device)
            feat_lengths[:] = base_features.size(1)

        features = self.image_encoder(base_features, feat_lengths)

        return features

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        logger.info('Backbone freezed.')

    def unfreeze_backbone(self, fixed_blocks):
        for param in self.backbone.parameters():  # open up all params first, then adjust the base parameters
            param.requires_grad = True
        self.backbone.set_fixed_blocks(fixed_blocks)
        self.backbone.unfreeze_base()
        logger.info('Backbone unfreezed, fixed blocks {}'.format(self.backbone.get_fixed_blocks()))


# Language Model with BERT
class EncoderText(nn.Module):
    def __init__(self, embed_size, no_txtnorm=False):
        super(EncoderText, self).__init__()
        self.embed_size = embed_size
        self.no_txtnorm = no_txtnorm

        self.bert = BertModel.from_pretrained('/root/autodl-tmp/bert-base-uncased')
        self.linear = nn.Linear(768, embed_size)

        # GCN reasoning 
        self.Rs_GCN_1 = Rs_GCN(in_channels=embed_size, inter_channels=embed_size)
        self.Rs_GCN_2 = Rs_GCN(in_channels=embed_size, inter_channels=embed_size)
        self.Rs_GCN_3 = Rs_GCN(in_channels=embed_size, inter_channels=embed_size)
        self.Rs_GCN_4 = Rs_GCN(in_channels=embed_size, inter_channels=embed_size)
        self.gpool = GPO(32, 32)
        # self.attention = featureInteraction(embed_size)
        # self.gpool = soft_max(32, 32)

    def forward(self, x, lengths):
        """Handles variable size captions
        """
        # Embed word ids to vectors
        bert_attention_mask = (x != 0).float()
        bert_emb = self.bert(x, bert_attention_mask)[0]  # B x N x D
        cap_len = lengths

        cap_emb = self.linear(bert_emb)

        GCN_img_emd = cap_emb.permute(0, 2, 1)
        #print(GCN_img_emd.size(),"ffffffffffffffffffff############################")
        GCN_img_emd = self.Rs_GCN_1(GCN_img_emd)
        cap_emb4 = GCN_img_emd.permute(0, 2, 1)
        GCN_img_emd = self.Rs_GCN_2(GCN_img_emd)
        cap_emb3 = GCN_img_emd.permute(0, 2, 1)
        GCN_img_emd = self.Rs_GCN_3(GCN_img_emd)
        cap_emb2 = GCN_img_emd.permute(0, 2, 1)
        GCN_img_emd = self.Rs_GCN_4(GCN_img_emd)
        # -> B,N,D
        cap_emb1 = GCN_img_emd.permute(0, 2, 1)

        #4_GCN_POOL
        pooled_features1, pool_weights = self.gpool(cap_emb1, cap_len.to(cap_emb.device))
        pooled_features2, pool_weights = self.gpool(cap_emb2, cap_len.to(cap_emb.device))
        pooled_features3, pool_weights = self.gpool(cap_emb3, cap_len.to(cap_emb.device))
        pooled_features4, pool_weights = self.gpool(cap_emb4, cap_len.to(cap_emb.device))
        #
        pooled_features_gcn = torch.cat([pooled_features1.unsqueeze(1), pooled_features2.unsqueeze(1),
                                     pooled_features3.unsqueeze(1), pooled_features4.unsqueeze(1)], dim=1)
        # pooled_features_gcn = self.attention(pooled_features_gcn)
        #求均值和方差
        feature_mean = torch.mean(pooled_features_gcn, dim=1)
        # print(pooled_features)
        feature_var = (torch.mean((pooled_features_gcn - feature_mean.unsqueeze(1)).pow(2), dim=1) + 1e-8).sqrt()
        # print(feature_var)
        #采样
        features_s = []
        num_sample = 4
        for i in range(num_sample):
            epsilon = torch.randn_like(feature_var)
            features_i = feature_mean + epsilon * feature_var
            features_s.append(features_i.unsqueeze(1))
        pooled_features = torch.cat(features_s, 1)
        # epsilon = torch.randn_like(feature_var)
        # pooled_features = feature_mean + epsilon * feature_var
        # print(pooled_features.shape)
        # normalization in the joint embedding space


        if not self.no_txtnorm:
            feature_mean = l2norm(feature_mean, dim=-1)
            feature_var = l2norm(feature_var, dim=-1)
            pooled_features = l2norm(pooled_features, dim=-1)
        # print((pooled_features - feature_mean.unsqueeze(1)).sum(dim=-1).softmax(dim=-1))
        return feature_mean, feature_var, pooled_features

class featureInteraction(nn.Module):
    def __init__(self, embed_size):
        super(featureInteraction, self).__init__()
        self.self_attention = MyMultiHeadAttention(d_model=embed_size, n_head=16)
        self.norm1 = LayerNorm(d_model=embed_size)
        self.dropout1 = nn.Dropout(p=0.2)


    def forward(self, img_emb):

        # self_attention
        img_emb_s = self.self_attention(q=img_emb, k=img_emb, v=img_emb)

        # add and norm
        img_emb = self.norm1(img_emb_s + img_emb)
        img_emb = self.dropout1(img_emb)

        return img_emb
