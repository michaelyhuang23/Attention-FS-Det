import torch
from torch import nn
import torch.nn.functional as F


class CrossAttention(nn.Module):
    def __init__(self, cfg, input_shape):
        super().__init__()

        self.input_shape = input_shape
        self.attn_size = cfg.MODEL.ATTENTION.INNER_SIZE
        self.self_attn_weight = cfg.MODEL.ATTENTION.SELF_ATTENTION_WEIGHT

        self.conv_support = nn.Conv2d(input_shape, self.attn_size, kernel_size=1, bias=False)
        self.conv_query = nn.Conv2d(input_shape, self.attn_size, kernel_size=1, bias=False)
        self.conv_self_support_attn = nn.Conv2d(input_shape, 1, kernel_size=1, bias=False)
        # maybe one is enough?

    def forward(self, features, supports):
        """
        features: (B, C, H, W) or (C, H, W)
        supports: (N, C, H, W)
        """
        if len(features.shape)==3:
            features = features[None,...]
        features_tr = self.conv_query(features)
        if len(supports.shape)!=4:
            print("bad thing 1")
            print(supports.shape)
            print(query.shape)
        supports_s = self.conv_self_support_attn(supports)[:, 0, :, :]
        # self_supports shape: (N, H, W)
        if len(supports_s.shape)!=3:
            print("bad thing 2")
            print(supports_s.shape)
            print(supports.shape)
        supports_s = supports_s.permute(1,2,0)[None,None,None,...]

        # supports shape: (N, C, H, W)
        supports_k = self.conv_support(supports)
        supports_k = supports_k.permute(1,2,3,0)
        # supports_k shape: (C', H, W, N)

        # features_tr shape: (B, C', H, W)
        features_tr = features_tr.permute(0,2,3,1)
        # features_tr shape: (B, H, W, C')
        CAtt = torch.tensordot(features_tr-torch.mean(features_tr), supports_k-torch.mean(supports_k), dims=1)
        # CAtt shape: (B, Hq, Wq, Hs, Ws, Ns)
        CAtt += supports_s * self.self_attn_weight

        CAtt = torch.reshape(CAtt, (CAtt.shape[0], CAtt.shape[1], CAtt.shape[2], -1))
        # CAtt shape: (B, Hq, Wq, L)
        CAtt = F.softmax(CAtt, dim=-1)

        supports_n = supports.permute(2,3,0,1)
        # supports_n shape: (H, W, N, C)
        supports_n = torch.reshape(supports_n, (-1, supports_n.shape[-1]))
        # supports_n shape: (L, C)

        support_ft = torch.tensordot(CAtt, supports_n, dims=1)
        # support_ft shape: (B, Hq, Wq, C)
        support_ft = support_ft.permute(0,3,1,2)
        return support_ft