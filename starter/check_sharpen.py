import torch, torch.nn.functional as F
torch.manual_seed(0)
E, H = 512, 8; hd = E // H
m = torch.nn.MultiheadAttention(E, H).eval()
with torch.no_grad():
    m.in_proj_bias.normal_(0, 0.1)
q, k, v = torch.randn(5, 1, E), torch.randn(9, 1, E), torch.randn(9, 1, E)

def attn_weights(mod):
    Wq, Wk, _ = mod.in_proj_weight.chunk(3); bq, bk, _ = mod.in_proj_bias.chunk(3)
    Q = F.linear(q, Wq, bq).view(5, H, hd).transpose(0, 1); K = F.linear(k, Wk, bk).view(9, H, hd).transpose(0, 1)
    return Q @ K.transpose(1, 2) / hd ** 0.5  # logits (H, 5, 9)

with torch.no_grad():
    base = attn_weights(m)
    h, s = 2, 3.0
    m.in_proj_weight[h * hd:(h + 1) * hd] *= s       # query rows of head h
    m.in_proj_bias[h * hd:(h + 1) * hd] *= s
    new = attn_weights(m)
    print("head 2 logits scaled by", float((new[h] / base[h]).mean()), "| max dev", float((new[h] - s * base[h]).abs().max()))
    others = [i for i in range(H) if i != h]
    print("other heads unchanged:", float((new[others] - base[others]).abs().max()))
    out_ref = m(q, k, v, need_weights=True, average_attn_weights=False)[1][0]  # (H, 5, 9) attention after softmax
    print("torch attention for head 2 equals softmax of scaled logits:", float((out_ref[h] - torch.softmax(new[h], -1)).abs().max()))
