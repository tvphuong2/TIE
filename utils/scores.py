import torch
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def distmult(s, r, o, mode='single'):
    # import pdb; pdb.set_trace()
    if mode == 'tail':
        return torch.sum((s * r).unsqueeze(1) * o, dim=-1)
    elif mode == 'head':
        return torch.sum(s * (r * o).unsqueeze(1), dim=-1)
    else:
        return torch.sum(s * r * o, dim=-1)

def simple(head, head_inv, rel, rel_inv, tail, tail_inv, mode='tail'):
    if mode == 'tail':
        scores1 = torch.sum((head * rel).unsqueeze(1) * tail_inv, dim=-1)
        scores2 = torch.sum((head_inv * rel_inv).unsqueeze(1) * tail, dim=-1)
    elif mode == 'head':
        scores1 = torch.sum(head * (rel * tail_inv).unsqueeze(1), dim=-1)
        scores2 = torch.sum(head_inv * (rel_inv * tail).unsqueeze(1), dim=-1)
    else:
        scores1 = torch.sum(head * rel * tail_inv, dim=-1)
        scores2 = torch.sum(head_inv * rel_inv * tail, dim=-1)
    return (scores1 + scores2) / 2


# "head" means to corrupt head
def complex(head, relation, tail, mode='single'):
    re_head, im_head = torch.chunk(head, 2, dim=-1)
    re_relation, im_relation = torch.chunk(relation, 2, dim=-1)
    re_tail, im_tail = torch.chunk(tail, 2, dim=-1)
    if mode == 'tail':
        re_score = re_head * re_relation - im_head * im_relation
        im_score = re_head * im_relation + im_head * re_relation
        score = re_score.unsqueeze(1) * re_tail + im_score.unsqueeze(1) * im_tail
    elif mode == 'head':
        re_score = re_relation * re_tail + im_relation * im_tail
        im_score = re_relation * im_tail - im_relation * re_tail
        score = re_head * re_score.unsqueeze(1) + im_head * im_score.unsqueeze(1)
    elif mode == 'relation':
        # import pdb; pdb.set_trace()
        re_score = re_head * re_tail + im_head * im_tail
        im_score = re_head * im_tail - im_head * re_tail
        score = re_relation * re_score.unsqueeze(1) + im_relation * im_score.unsqueeze(1)
    else:
        re_score = re_head * re_relation - im_head * im_relation
        im_score = re_head * im_relation + im_head * re_relation
        score = re_score * re_tail + im_score * im_tail

    return score.sum(dim=-1)


def transE(head, relation, tail, mode='single'):
    print("head, r, t.shape")
    print(head.shape)
    print(relation.shape)
    print(tail.shape)

    if mode == 'tail':
        score = (head + relation).unsqueeze(1) - tail
    elif mode == 'head':
        score = head + (relation - tail).unsqueeze(1)
    else:
        score = head + relation - tail
    score = - torch.norm(score, p=1, dim=-1)
    return score


def chunk(x):
  if len(x.shape) == 3:
    b = torch.reshape(x, (x.shape[0], x.shape[1], -1, 2))
    b = torch.transpose(b, 2, 3)
    a,b = torch.chunk(b, 2, dim=-2)
    a = torch.reshape(a, (a.shape[0], x.shape[1], -1))
    b = torch.reshape(b, (b.shape[0], x.shape[1], -1))
  else:
    b = torch.reshape(x, (x.shape[0], -1, 2))
    b = torch.transpose(b, 1, 2)
    a,b = torch.chunk(b, 2, dim=-2)
    a = torch.reshape(a, (a.shape[0], -1))
    b = torch.reshape(b, (b.shape[0], -1))
  return a,b


def rotate(head, relation, tail, mode='single'):
    pi = 3.14159265358979323846
    if mode == "head": tail = tail.unsqueeze(1)
    if mode == "tail": head = head.unsqueeze(1)
    if mode != "single": relation = relation.unsqueeze(1)
    
    re_head, im_head = chunk(head)
    re_tail, im_tail = chunk(tail)
    re_relation, im_relation = chunk(relation)

    relation = (re_relation + im_relation)/2

    print(relation.shape)
    print(re_head.shape)
    print(re_tail.shape)
    #Make phases of relations uniformly distributed in [-pi, pi]
    print("chucked ------------------------------------------")
    print((re_head.shape, re_head.shape, re_tail.shape))

    embedding_range = torch.nn.Parameter(
        torch.Tensor([(24 + 2) / head.shape[-1]]), 
        requires_grad=False
    )

    gamma = torch.nn.Parameter(
        torch.Tensor([24]), 
        requires_grad=False
    )

    phase_relation = relation/(embedding_range.item()/pi)

    re_relation = torch.cos(phase_relation)
    im_relation = torch.sin(phase_relation)

    if mode == 'head':
        re_score = re_relation * re_tail + im_relation * im_tail
        im_score = re_relation * im_tail - im_relation * re_tail
        re_score = re_score - re_head
        im_score = im_score - im_head
    else:
        re_score = re_head * re_relation - im_head * im_relation
        im_score = re_head * im_relation + im_head * re_relation
        re_score = re_score - re_tail
        im_score = im_score - im_tail

    print(re_score.shape)
    print(im_score.shape)

    score = torch.stack([re_score, im_score], dim = 0)
    score = score.norm(dim = 0)

    print(score.shape)
    score = F.dropout(score, p=0.2)
    return gamma.item() - score.sum(dim = -1)


def ATiSE_score(head_mean, head_cov, tail_mean, tail_cov, rel_mean, rel_cov, mode='single'):
    # # Calculate KL(r, e)
    if mode == 'tail':
        error_mean =  head_mean.unsqueeze(1) - tail_mean
        error_cov = head_cov.unsqueeze(1) + tail_cov
        rel_mean = rel_mean.unsqueeze(1)
        rel_cov = rel_cov.unsqueeze(1)
    elif mode == 'head':
        error_mean =  head_mean - tail_mean.unsqueeze(1)
        error_cov = head_cov + tail_cov.unsqueeze(1)
        rel_mean = rel_mean.unsqueeze(1)
        rel_cov = rel_cov.unsqueeze(1)
    else:
        error_mean =  head_mean - tail_mean
        error_cov = head_cov + tail_cov
    
    lossp1 = torch.sum(error_cov/rel_cov, dim=-1)
    lossp2 = torch.sum((error_mean - rel_mean) ** 2 / rel_cov, dim=-1)
    lossp3 = - torch.sum(torch.log(error_cov), dim=-1) + torch.sum(torch.log(rel_cov), dim=-1)
    KLre = - (lossp1 + lossp2 + lossp3 - 128) / 2
 
    return KLre


def ConvKB(h, r, t, hidden_size, entTotal, relTotal, out_channels=64, kernel_size=1, convkb_drop_prob=0.5, mode='single'):
    
    # self.ent_embeddings = nn.Embedding(self.config.entTotal, self.config.hidden_size) 
    # self.rel_embeddings = nn.Embedding(self.config.relTotal, self.config.hidden_size)
    conv1_bn = torch.nn.BatchNorm2d(1)
    conv_layer = torch.nn.Conv2d(1, out_channels, (kernel_size, 3))  # kernel size x 3
    conv2_bn = torch.nn.BatchNorm2d(out_channels)
    dropout = torch.nn.Dropout(convkb_drop_prob)
    non_linearity = torch.nn.ReLU() # you should also tune with torch.tanh() or torch.nn.Tanh()
    fc_layer = torch.nn.Linear((hidden_size - kernel_size + 1) * out_channels, 1, bias=False)

    print("h shape")
    print(h.shape)
    print("r shape")
    print(r.shape)
    print("t shape")
    print(t.shape)
    print("---------------------------")

    h = h.unsqueeze(1) # bs x 1 x dim
    r = r.unsqueeze(1)
    # t = t.unsqueeze(1)
    print("h shape")
    print(h.shape)
    print("r shape")
    print(r.shape)
    print("t shape")
    print(t.shape)

    conv_input = torch.cat([h, r, t], 1)  # bs x 3 x dim
    conv_input = conv_input.transpose(1, 2)
    # To make tensor of size 4, where second dim is for input channels
    conv_input = conv_input.unsqueeze(1)

    conv_input = conv_input.to(device)

    conv1_bn = conv1_bn.to(device)
    conv_input = conv1_bn(conv_input)

    conv_layer = conv_layer.to(device)
    out_conv = conv_layer(conv_input)

    conv2_bn = conv2_bn.to(device)
    out_conv = conv2_bn(out_conv)

    non_linearity = non_linearity.to(device)
    out_conv = non_linearity(out_conv)
    out_conv = out_conv.view(-1, (hidden_size - kernel_size + 1) * out_channels)
    
    dropout = dropout.to(device)
    input_fc = dropout(out_conv)

    fc_layer = fc_layer.to(device)
    score = fc_layer(input_fc).view(-1)

    return -score

