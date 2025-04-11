import torch
import torch.nn as nn
from torch.autograd import Variable


class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss (max-margin based)
    """

    def __init__(self, opt, margin=0, max_violation=False):
        super(ContrastiveLoss, self).__init__()
        self.opt = opt
        self.margin = margin
        self.max_violation = max_violation

    def max_violation_on(self):
        self.max_violation = True
        print('Use VSE++ objective.')

    def max_violation_off(self):
        self.max_violation = False
        print('Use VSE0 objective.')

    def forward(self, img_emb_mu, img_emb_var, im, cap_emb_mu, cap_emb_var, s):
        # print(img_emb_mu)
        # print(img_emb_var)
        # print(im)
        # print(cap_emb_mu)
        # print(cap_emb_var)
        # print(s)
        # compute image-sentence score matrix
        scores = get_sim(im, s)
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

        # kl_loss
        kl_loss_img = -(1 + torch.log(img_emb_var.pow(2)) - img_emb_mu.pow(2) - img_emb_var.pow(2)) / 2
        kl_loss_img = kl_loss_img.sum(dim=1).mean()

        kl_loss_text = -(1 + torch.log(cap_emb_var.pow(2)) - cap_emb_mu.pow(2) - cap_emb_var.pow(2)) / 2
        kl_loss_text = kl_loss_text.sum(dim=1).mean()

        # 跨模态高斯分布loss
        # cross_gaussian_los = cross_gaussian_loss(im, s)
        # cross_gaussian_los = product_gaussians(img_emb_mu, img_emb_var, cap_emb_mu, cap_emb_var)
        cross_gaussian_los = distribution_distance(img_emb_mu, img_emb_var, cap_emb_mu, cap_emb_var)
        # cross_gaussian_los2 = cross_modal_loss(img_emb_mu, img_emb_var, cap_emb_mu, cap_emb_var)
        # print(cross_gaussian_los)
        # print(cross_gaussian_los2)

        return cost_s.sum() + cost_im.sum() + 0.0001 * (kl_loss_img + kl_loss_text) + 0.01 * cross_gaussian_los
               # + 0.01 * cross_gaussian_los2


def get_sim(images, captions):
    sims = torch.zeros(len(images), len(captions)).to(images.device)
    for idx, sample_image in enumerate(images.transpose(0, 1)):
        # print(idx)
        for idy, sample_cap in enumerate(captions.transpose(0, 1)):
            sim = (sample_image.mm(sample_cap.t()))
            # print(sim.shape)
            sims = sims.add(sim)
    #
    similarities = sims.div(16)
    # similarities = images.mm(captions.t())
    # sim = images[:, 0, :].mm(captions[:, 0, :].t())
    # sim2 = images[:, 1, :].mm(captions[:, 1, :].t())
    # sim3 = images[:, 2, :].mm(captions[:, 2, :].t())
    # sim4 = images[:, 3, :].mm(captions[:, 3, :].t())
    # similarities = (sim + sim2 + sim3 + sim4)/4
    return similarities

def cross_gaussian_loss(images, captions):
    n_caption = captions.size(0)
    n_image = images.size(0)
    # 每行代表一张图像，每列代表一个文本
    covariance_matrix = []
    for i in range(n_caption):

        captions_i = captions[i, :, :].unsqueeze(0)
        captions_i_expand = captions_i.repeat(n_image, 1, 1)
        concat = torch.cat((images, captions_i_expand), dim=1)
        concat_mean = torch.mean(concat, dim=1)
        concat_var = (torch.mean((concat - concat_mean.unsqueeze(1)).pow(2), dim=1) + 1e-8).sqrt()
        concat_var = torch.mean(concat_var, dim=-1, keepdim=True)
        covariance_matrix.append(concat_var)

    covariance_matrix = torch.cat(covariance_matrix, 1)

    diagonal = covariance_matrix.diag().view(n_image, 1)
    d1 = diagonal.expand_as(covariance_matrix)
    d2 = diagonal.t().expand_as(covariance_matrix)

    cost_s = (0.2 + d1 - covariance_matrix).clamp(min=0)
    cost_im = (0.2 + d2 - covariance_matrix).clamp(min=0)

    mask = torch.eye(covariance_matrix.size(0)) > .5
    I = Variable(mask)
    if torch.cuda.is_available():
         I = I.cuda()
    cost_s = cost_s.masked_fill_(I, 0)
    cost_im = cost_im.masked_fill_(I, 0)

    # if self.max_violation:
    #     cost_s = cost_s.max(1)[0]
    #     cost_im = cost_im.max(0)[0]
    return cost_s.sum() + cost_im.sum()

from torch.distributions import MultivariateNormal

def product_2_gaussians(mean1, variance1, mean2, variance2):
    target_mean = mean2
    target_variance = variance2

    inv_variance1 = 1 / variance1
    inv_target_variance = 1 / target_variance
    # C = torch.diag_embed(1 / (inv_variance1 + inv_target_variance))
    C = 1 / (inv_variance1 + inv_target_variance)
    c = torch.mul(C, torch.mul(inv_variance1, mean1) + torch.mul(inv_target_variance, target_mean))
    # log_z = MultivariateNormal(target_mean, torch.diag_embed(variance1 + target_variance + 1e-6)).log_prob(mean1)
    # C = torch.diagonal(C, dim1=-2, dim2=-1)
    # C = torch.log(C)
    log_z = 0
    return c, C, log_z

def product_gaussians(mean1, variance1, mean2, variance2):
    n_caption = mean1.size(0)
    n_image = mean2.size(0)
    # 每行代表一张图像，每列代表一个文本
    covariance_matrix = []
    variance2 = variance2.pow(2)
    variance1 = variance1.pow(2)

    # print(variance1.shape)
    for i in range(n_caption):
        mean2_i = mean2[i, :].unsqueeze(0)
        mean2_i_expand = mean2_i.repeat(n_image, 1)
        # print(mean2_i_expand.shape)
        # print(variance2.shape)
        variance2_i = variance2[i, :].unsqueeze(0)

        variance2_i_expand = variance2_i.repeat(n_image, 1)
        # print(variance2_i_expand.shape)
        concat_mean_i, concat_variance_i, log_z_i = \
            product_2_gaussians(mean1, variance1, mean2_i_expand, variance2_i_expand)
        concat_variance_i = torch.mean(concat_variance_i, dim=-1, keepdim=True)
        covariance_matrix.append(concat_variance_i)

    covariance_matrix = torch.cat(covariance_matrix, 1)

    diagonal = covariance_matrix.diag().view(n_image, 1)
    d1 = diagonal.expand_as(covariance_matrix)
    d2 = diagonal.t().expand_as(covariance_matrix)

    cost_s = (0.0 + d1 - covariance_matrix).clamp(min=0)
    cost_im = (0.0 + d2 - covariance_matrix).clamp(min=0)

    mask = torch.eye(covariance_matrix.size(0)) > .5
    I = Variable(mask)
    if torch.cuda.is_available():
        I = I.cuda()
    cost_s = cost_s.masked_fill_(I, 0)
    cost_im = cost_im.masked_fill_(I, 0)

    # if self.max_violation:
    cost_s = cost_s.max(1)[0]
    cost_im = cost_im.max(0)[0]
    return cost_s.sum() + cost_im.sum()

def elk_dist(mu1, mu2, log_sigma1, log_sigma2):
    sum_sigma_sq = torch.exp(2 * log_sigma1) + torch.exp(2 * log_sigma2)
    dist = (mu1 - mu2) ** 2
    dist = dist.squeeze(1)
    elk = dist / (sum_sigma_sq) + torch.log(sum_sigma_sq)
    return -0.5 * torch.sum(elk, dim=1)


def kl_dist(mu1, mu2, log_sigma1, log_sigma2):
    dist = (mu1 - mu2) ** 2
    dist = dist/log_sigma2 - (torch.log(log_sigma1/log_sigma2)) + log_sigma1/log_sigma2
    return -torch.sum(dist, dim=1)


def js_dist(mu1, mu2, log_sigma1, log_sigma2):
    dist = (mu1 - mu2) ** 2
    var1 = log_sigma1
    var2 = log_sigma2
    dist = (dist + var1)/var2 + (dist + var2)/var1
    return -torch.sum(dist, dim=1)


def bhattacharyya_dist(mu1, mu2, log_sigma1, log_sigma2):
    dist = (mu1 - mu2) ** 2
    # dist = dist.squeeze(1)
    # sigma1 = torch.exp(log_sigma1)
    # sigma2 = torch.exp(log_sigma2)

    dist = dist / (torch.exp(log_sigma1 * 2) + torch.exp(log_sigma2 * 2))
    dist = dist + 2 * torch.log(sigma1 / sigma2 + sigma2 / sigma1)
    ddd = 2 * torch.log(torch.ones(1) * 2)
    dist = dist.float() - ddd.to(dist.device).float()
    dist = dist / 4
    return -torch.sum(dist, dim=1)


def wasserstein_dist(mu1, mu2, log_sigma1, log_sigma2):
    dist = (mu1 - mu2) ** 2
    # dist = dist.squeeze(1)
    dist = dist + (log_sigma1 - log_sigma2) ** 2
    return -torch.sum(dist, dim=1)

def distribution_distance(mean1, variance1, mean2, variance2):
    n_caption = mean1.size(0)
    n_image = mean2.size(0)
    # 每行代表一张图像，每列代表一个文本
    img_distance_matrix = []
    txt_distance_matrix = []
    variance1 = variance1.pow(2)
    variance2 = variance2.pow(2)

    # print(variance1.shape)
    for i in range(n_caption):
        mean2_i = mean2[i, :].unsqueeze(0)
        mean2_i_expand = mean2_i.repeat(n_image, 1)
        # print(mean2_i_expand.shape)
        # print(variance2.shape)
        variance2_i = variance2[i, :].unsqueeze(0)
        # print(variance2_i)
        variance2_i_expand = variance2_i.repeat(n_image, 1)
        # print(variance2_i_expand.shape)
        concat_mean_i, concat_variance_i, log_z_i = \
            product_2_gaussians(mean1, variance1, mean2_i_expand, variance2_i_expand)
        # concat_variance_i = torch.mean(concat_variance_i, dim=-1, keepdim=True)
        # print(concat_mean_i)
        # print(concat_variance_i)
        distance_i = wasserstein_dist(mean2_i_expand, concat_mean_i, variance2_i_expand, concat_variance_i).unsqueeze(0)
        # print(distance_i)
        txt_distance_matrix.append(distance_i)

        mean1_i = mean1[i, :].unsqueeze(0)
        mean1_i_expand = mean1_i.repeat(n_caption, 1)
        # print(mean2_i_expand.shape)
        # print(variance2.shape)
        variance1_i = variance1[i, :].unsqueeze(0)
        variance1_i_expand = variance1_i.repeat(n_caption, 1)

        concat_mean_i, concat_variance_i, log_z_i = \
            product_2_gaussians(mean1_i_expand, variance1_i_expand, mean2, variance2)
        # concat_variance_i = torch.mean(concat_variance_i, dim=-1, keepdim=True)
        distance_i = wasserstein_dist(mean1_i_expand, concat_mean_i, variance1_i_expand, concat_variance_i).unsqueeze(0)
        img_distance_matrix.append(distance_i)

    img_distance_matrix = torch.cat(img_distance_matrix, 0)
    txt_distance_matrix = torch.cat(txt_distance_matrix, 0)
    # print(img_distance_matrix)
    # print(txt_distance_matrix)

    img_diagonal = img_distance_matrix.diag().view(n_caption, 1)
    # print(img_diagonal)
    txt_diagonal = txt_distance_matrix.diag().view(n_image, 1)
    d1 = img_diagonal.expand_as(img_distance_matrix)
    d2 = txt_diagonal.expand_as(txt_distance_matrix)
    # print(d1)
    cost_s = (0.2 + img_distance_matrix - d1).clamp(min=0)
    cost_im = (0.2 + txt_distance_matrix - d2).clamp(min=0)

    mask = torch.eye(txt_diagonal.size(0)) > .5
    I = Variable(mask)
    if torch.cuda.is_available():
        I = I.cuda()
    cost_s = cost_s.masked_fill_(I, 0)
    cost_im = cost_im.masked_fill_(I, 0)

    # if self.max_violation:
    cost_s = cost_s.max(1)[0]
    cost_im = cost_im.max(1)[0]

    return cost_s.sum() + cost_im.sum()

def distribution_distance2(mean1, variance1, mean2, variance2):
    n_caption = mean1.size(0)
    n_image = mean2.size(0)
    # 每行代表一张图像，每列代表一个文本
    img_distance_matrix = []
    txt_distance_matrix = []
    variance1 = variance1.pow(2)
    variance2 = variance2.pow(2)
    match_mean, match_variance, _ = \
        product_2_gaussians(mean1, variance1, mean2, variance2)
    # print(variance1.shape)
    for i in range(n_caption):
        # Match to Gaussian

        mean2_i = mean2[i, :].unsqueeze(0)
        mean2_i_expand = mean2_i.repeat(n_image, 1)
        # print(mean2_i_expand.shape)
        # print(variance2.shape)
        variance2_i = variance2[i, :].unsqueeze(0)
        # print(variance2_i)
        variance2_i_expand = variance2_i.repeat(n_image, 1)
        # print(variance2_i_expand.shape)
        concat_mean_i, concat_variance_i, log_z_i = \
            product_2_gaussians(mean1, variance1, mean2_i_expand, variance2_i_expand)
        # concat_variance_i = torch.mean(concat_variance_i, dim=-1, keepdim=True)
        # print(concat_mean_i)
        # print(concat_variance_i)
        distance_i = js_dist(mean2_i_expand, concat_mean_i, variance2_i_expand, concat_variance_i).unsqueeze(0)
        # print(distance_i)
        txt_distance_matrix.append(distance_i)

        mean1_i = mean1[i, :].unsqueeze(0)
        mean1_i_expand = mean1_i.repeat(n_caption, 1)
        # print(mean2_i_expand.shape)
        # print(variance2.shape)
        variance1_i = variance1[i, :].unsqueeze(0)
        variance1_i_expand = variance1_i.repeat(n_caption, 1)

        concat_mean_i, concat_variance_i, log_z_i = \
            product_2_gaussians(mean1_i_expand, variance1_i_expand, mean2, variance2)
        # concat_variance_i = torch.mean(concat_variance_i, dim=-1, keepdim=True)
        distance_i = js_dist(mean1_i_expand, concat_mean_i, variance1_i_expand, concat_variance_i).unsqueeze(0)
        img_distance_matrix.append(distance_i)

    img_distance_matrix = torch.cat(img_distance_matrix, 0)
    txt_distance_matrix = torch.cat(txt_distance_matrix, 0)
    # print(img_distance_matrix)
    # print(txt_distance_matrix)

    img_diagonal = img_distance_matrix.diag().view(n_caption, 1)
    # print(img_diagonal)
    txt_diagonal = txt_distance_matrix.diag().view(n_image, 1)
    d1 = img_diagonal.expand_as(img_distance_matrix)
    d2 = txt_diagonal.expand_as(txt_distance_matrix)
    # print(d1)
    cost_s = (0.2 + img_distance_matrix - d1).clamp(min=0)
    cost_im = (0.2 + txt_distance_matrix - d2).clamp(min=0)

    mask = torch.eye(txt_diagonal.size(0)) > .5
    I = Variable(mask)
    if torch.cuda.is_available():
        I = I.cuda()
    cost_s = cost_s.masked_fill_(I, 0)
    cost_im = cost_im.masked_fill_(I, 0)

    # if self.max_violation:
    cost_s = cost_s.max(1)[0]
    cost_im = cost_im.max(1)[0]

    return cost_s.sum() + cost_im.sum()

def cross_modal_loss(mean1, variance1, mean2, variance2):
    n_caption = mean1.size(0)
    n_image = mean2.size(0)
    # 每行代表一张图像，每列代表一个文本
    img_distance_matrix = []
    txt_distance_matrix = []
    variance1 = variance1.pow(2)
    variance2 = variance2.pow(2)
    match_mean, match_variance, _ = \
        product_2_gaussians(mean1, variance1, mean2, variance2)
    # print(variance1.shape)
    for i in range(n_caption):
        # Match to Gaussian

        match_mean_i = match_mean[i, :].unsqueeze(0)
        match_mean_i_expand = match_mean_i.repeat(n_caption, 1)

        match_variance_i = match_variance[i, :].unsqueeze(0)
        match_variance_expand = match_variance_i.repeat(n_caption, 1)

        distance_i_img = wasserstein_dist(match_mean_i_expand, mean1, match_variance_expand, variance1).unsqueeze(0)
        img_distance_matrix.append(distance_i_img)

        distance_i_txt = wasserstein_dist(match_mean_i_expand, mean2, match_variance_expand, variance2).unsqueeze(0)
        txt_distance_matrix.append(distance_i_txt)

    img_distance_matrix = torch.cat(img_distance_matrix, 0)
    txt_distance_matrix = torch.cat(txt_distance_matrix, 0)
    # print(img_distance_matrix)
    # print(txt_distance_matrix)

    img_diagonal = img_distance_matrix.diag().view(n_caption, 1)
    # print(img_diagonal)
    txt_diagonal = txt_distance_matrix.diag().view(n_image, 1)
    d1 = img_diagonal.expand_as(img_distance_matrix)
    d2 = txt_diagonal.expand_as(txt_distance_matrix)
    # print(d1)
    cost_s = (0.2 + img_distance_matrix - d1).clamp(min=0)
    cost_im = (0.2 + txt_distance_matrix - d2).clamp(min=0)
    cost_s_i = (0.2 + img_distance_matrix - d2).clamp(min=0)
    cost_im_s = (0.2 + txt_distance_matrix - d1).clamp(min=0)

    mask = torch.eye(txt_diagonal.size(0)) > .5
    I = Variable(mask)
    if torch.cuda.is_available():
        I = I.cuda()

    cost_s = cost_s.masked_fill_(I, 0)
    cost_im = cost_im.masked_fill_(I, 0)
    cost_s_i = cost_s_i.masked_fill_(I, 0)
    cost_im_s = cost_im_s.masked_fill_(I, 0)
    # if self.max_violation:

    cost_s = cost_s.max(1)[0]
    cost_im = cost_im.max(1)[0]
    cost_s_i = cost_s_i.max(1)[0]
    cost_im_s = cost_im_s.max(1)[0]

    return cost_s.sum() + cost_im.sum() + cost_s_i.sum() + cost_im_s.sum()


def shard_xattn(images, captions, caplens, opt, shard_size):
    """
    Computer pairwise t2i image-caption distance with locality sharding
    """
    n_im_shard = (len(images) - 1) // shard_size + 1
    n_cap_shard = (len(captions) - 1) // shard_size + 1

    d1 = np.zeros((len(images), len(captions)))
    for i in range(n_im_shard):
        im_start, im_end = shard_size * i, min(shard_size * (i + 1), len(images))
        for j in range(n_cap_shard):
            cap_start, cap_end = shard_size * j, min(shard_size * (j + 1), len(captions))
            im = Variable(torch.from_numpy(images[im_start:im_end]), volatile=True).cuda()
            s = Variable(torch.from_numpy(captions[cap_start:cap_end]), volatile=True).cuda()
            similarities = xattn_score_test(im, s, l, opt)
            d1[im_start:im_end, cap_start:cap_end] = similarities.data.cpu().numpy()
    return d1

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
        # self.w_concat = nn.Linear(d_model, d_model)
        # nn.init.xavier_normal_(self.w_concat.weight, gain=1.414)

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
        # out = self.w_concat(out)

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

class self_attention_cross_modal_encoding(nn.Module):
    def __init__(self, embed_size):
        super(self_attention_cross_modal_encoding, self).__init__()
        self.self_attention = MyMultiHeadAttention(d_model=embed_size, n_head=16)
        # self.norm1 = LayerNorm(d_model=embed_size)
        # self.dropout1 = nn.Dropout(p=0.2)

    def forward(self, img_emb, cap_emb):
        # print(img_emb.shape)
        # print(cap_emb.shape)
        n_caption = cap_emb.size(0)
        n_image = img_emb.size(0)
        # 每行代表一张图像，每列代表一个文本
        cap_embs = []
        img_embs = []
        for i in range(n_caption):
            img_emb_i = img_emb[i, :, :].unsqueeze(0)
            img_emb_i_expand = img_emb_i.repeat(n_caption, 1, 1)
            img_emb_i_expand = self.self_attention(q=img_emb_i_expand, k=cap_emb, v=cap_emb)
            # img_emb_i_expand = self.norm1(img_emb_i_expand + img_emb_i_expand_s)
            # img_emb_i_expand = self.dropout1(img_emb_i_expand)
            img_embs.append(img_emb_i_expand.unsqueeze(0))

            cap_emb_i = cap_emb[i, :, :].unsqueeze(0)
            cap_emb_i_expand = cap_emb_i.repeat(n_image, 1, 1)
            cap_emb_i_expand = self.self_attention(q=cap_emb_i_expand, k=img_emb, v=img_emb)
            # cap_emb_i_expand = self.norm1(cap_emb_i_expand + cap_emb_i_expand_s)
            # cap_emb_i_expand = self.dropout1(cap_emb_i_expand)
            cap_embs.append(cap_emb_i_expand.unsqueeze(0))

        cap_embs = torch.cat(cap_embs, dim=0)
        img_embs = torch.cat(img_embs, dim=0)
        cap_embs = cap_embs.transpose(0, 1)
        cap_embs = l2norm(cap_embs, dim=-1)
        img_embs = l2norm(img_embs, dim=-1)
        cap_embs = cap_embs.transpose(2, 3)
        # sim = torch.mul(img_embs, cap_embs).sum(dim=-1).mean(dim=-1)
        sim = torch.matmul(img_embs, cap_embs).mean(dim=-1).mean(dim=-1)
        # print(img_embs.shape)
        # print(cap_embs.shape)
        return sim

