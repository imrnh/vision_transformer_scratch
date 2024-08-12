import torch
import torch.nn as nn


class MultiHeadedSelfAttention(nn.Module):
    def __init__(self, d, n_heads):
        super(MultiHeadedSelfAttention, self).__init__()

        self.d = d
        self.n_heads = n_heads

        assert d % n_heads == 0, "Dimension of heads must be divisible by number of heads"

        self.d_head = d // n_heads

        self.q_map = nn.ModuleList(
            [nn.Linear(in_features=self.d_head, out_features=self.d_head) for _ in range(self.n_heads)])
        self.k_map = nn.ModuleList(
            [nn.Linear(in_features=self.d_head, out_features=self.d_head) for _ in range(self.n_heads)])
        self.v_map = nn.ModuleList(
            [nn.Linear(in_features=self.d_head, out_features=self.d_head) for _ in range(self.n_heads)])
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, sequences):
        """
            Each head would be trained on specific subsegment of x.
            For example:
                [0,1,2,3,4,5,6,7,8,9] total 10 elements. And say we have 2 head
                0 - 4 i.e. first 5 elements would be trained on head 1
                5-9 i.e. last 5 elements would be trained on head 2. 

                Source needed to verify this. But this is indeed true. 
                Cause, in the code of Ross Wightman, he used the following code seq:
                
                        out = rearrange(out, pattern="b h n d -> b n (h d)")  
                
                Where n is seq_len i.e. number of tokens, d is token_dim. And the eqn's out matches fully with what we want to achieve.
                
                

        :param sequences: Input sequence i.e. x
        :return:
        """

        # Sequences has shape (N, seq_length, token_dim)
        # We go into shape    (N, seq_length, n_heads, token_dim / n_heads)
        # And come back to    (N, seq_length, item_dim)  (through concatenation)

        concatenated_sequence = []
        token_dim = sequences.shape[-1]  # Accessing the shape of each patch i.e. the last element of the shape tuple.

        for seq in sequences:  # seq have shape = seq_length, token_dim    i.e.  50, 8
            seq_concat_res = []
            for head_idx in range(self.n_heads):
                sidx = head_idx * self.d_head
                eidx = (head_idx + 1) * self.d_head
                sub_seq = seq[: , sidx : eidx]  # 0,1,2,3 for head 1. 4,5,6,7 for head 2

                q = self.q_map[head_idx](sub_seq)
                k = self.k_map[head_idx](sub_seq)
                v = self.v_map[head_idx](sub_seq)

                pre_attn = self.softmax(torch.matmul(q, k.transpose(-1, -2)))  # q, k both are 2d with shape (50,1)
                attention = torch.matmul(pre_attn, v)  # v have a shape of (50 x 4) and pre_attn have a shape of (50 x 50)

                seq_concat_res.append(attention)
            concatenated_sequence.append(torch.hstack(seq_concat_res))

        return torch.cat([torch.unsqueeze(r, dim=0) for r in concatenated_sequence])
