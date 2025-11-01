# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from math import sqrt


# class MHA(nn.Module):
#     def __init__(self, n_model, n_head, n_kvhead, dropout=0, causal=False):
#         super().__init__()
#         assert n_model % n_head == 0 and n_head % n_kvhead == 0

#         self.n_model, self.n_head, self.n_kvhead = n_model, n_head, n_kvhead
#         self.d_head, self.n_rep = n_model // n_head, n_head // n_kvhead
#         self.causal = causal

#         self.Wq = nn.Linear(n_model, n_model)
#         self.Wk = nn.Linear(n_model, self.d_head * self.n_kvhead)
#         self.Wv = nn.Linear(n_model, self.d_head * self.n_kvhead)
#         self.Wo = nn.Linear(n_model, n_model)
#         self.attnDrop = nn.Dropout(dropout)

#     def _split_q(self, x):
#         # (B, L, N) -> (B, NH, L, DH)
#         B, L, N = x.shape
#         return x.view(B, L, self.n_head, -1).transpose(1,2)

#     def _split_kv(self, x):
#         # (B, L, NKV*DH) -> (B, NKV, L, DH)
#         B, L, N = x.shape
#         return x.view(B, L, self.n_kvhead, -1).transpose(1,2)
    
#     def _tile_kv(self, x):
#         # (B, NKV, L, DH) -> (B, NH, L, DH)
#         B, _, L, _ = x.shape
#         return x.unsqueeze(2).repeat(1,1,self.n_rep,1,1).view(B, self.n_head, L, -1)
    
#     def _merge(self, x):
#         # (B, NH, L, DH) -> (B, L, N)
#         B, _, L, _ = x.shape
#         return x.transpose(1,2).contiguous().view(B, L, -1)

#     def forward(self, q_in, k_in, v_in, mask=None, cache=None):
#         _, Tq, _ = q_in.shape
#         Q, K, V = self.Wq(q_in), self.Wk(k_in), self.Wv(v_in)
#         Q = self._split_q(Q)
#         K, V = self._split_kv(K), self._split_kv(V)

#         if cache:
#             cK = torch.concat((cache["K"], K), dim=2)
#             cV = torch.concat((cache["V"], V), dim=2)
        
#         K, V = self._tile_kv(cK), self._tile_kv(cV)

#         scores = torch.matmul(Q, K.transpose(-2,-1)) / sqrt(self.d_head)

#         if self.causal:
#             Tk = K.size(2)
#             msk = torch.triu(torch.ones(Tq, Tk), diagonal=Tk-Tq+1).bool()
#             scores = scores.masked_fill(msk, -1e9)
        
#         if mask:
#             scores = scores.masked_fill(~mask, -1e9)
        
#         probs = self.attnDrop(F.softmax(scores, dim=-1))
#         attnOutput = torch.matmul(probs, V)

#         finalOutput = self.Wo(self._merge(attnOutput))

#         return finalOutput, {"K": K, "V": V}

from qd_search import QDArchive, qd_back_and_forth

if __name__ == "__main__":
    myQD = QDArchive()
    print("hello world")