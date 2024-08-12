import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import dataset, dataloader
from torchvision.transforms import ToTensor
from torchvision.datasets.mnist import MNIST
from util_fn import positional_embedding, make_patches
from encoder import Encoder


class ViT(nn.Module):
    def __init__(self, chw=(1, 28, 28), n_patches=7, n_blocks=2, hidden_d=8, n_heads=2, out_d=10):
        super(ViT, self).__init__()

        self.chw = chw
        self.n_patches = n_patches
        self.hidden_d = hidden_d  # This would be input by user.
        self.n_heads = n_heads
        self.n_blocks = n_blocks

        assert chw[1] % n_patches == 0, "Height of image should be divided by n_patches"
        assert chw[2] % n_patches == 0, "Width of image should be divided by n_patches"

        self.patch_size = (chw[1] // n_patches, chw[2] // n_patches)

        # 1) Linear mapper
        self.input_d = int(chw[0] * self.patch_size[0] * self.patch_size[1])  # Get size of each flattened patch
        self.linear_mapper = nn.Linear(in_features=self.input_d, out_features=self.hidden_d)  # Takes a flatten patch and convert into an embedding or a vector.

        # 2) Learnable "cls" token. As each token is convereted into a vector of hidden_d using linear_mapper, cls_token should a vec of hidden_d to match other tokens.
        self.cls_token = nn.Parameter(torch.rand(1, self.hidden_d))  # We could have created a patch_size * patch_size 1D array. But it would make adding this to tokens list harder.
        # Therefore, we created a 2D array and used torch.vStack() method to add it to the tokens list for each image. That's the only reason for 1 at the beginning of the rand function.

        # 3) Positional Embeddings
        self.position_embedding = nn.Parameter(torch.tensor(positional_embedding(self.n_patches ** 2 + 1, self.hidden_d)))  # 50, 8
        self.position_embedding.requires_grad = False  # It is not trainable.

        # 4) MHSA blocks
        self.encoder_blocks = nn.ModuleList([Encoder(hidden_d=self.hidden_d, n_heads=n_heads) for _ in range(self.n_blocks)])

        # 5) Classification MLPk
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_d, out_d),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        # 0) Dividing images into patches
        n, _, _, _ = x.shape
        patches = make_patches(x, self.n_patches)  # batch_size, 49,

        # 1) Running linear layer tokenization. Map the vector corresponding to each patch to the hidden size dimension
        tokens = self.linear_mapper(patches)  # This would take as input each of the patch of size 16 automatically and return the embeddings. Out shape = batch_size, 49, 8

        # 2) Adding classification token to the tokens
        tokens = torch.stack([torch.vstack([self.cls_token, tokens[i]]) for i in range(len(tokens))])  # Adding cls_token at the beginning of each image's embeddings.  out shape = batch_size, 50, 8

        # 3) Add positional embedding to the patches
        pos_embeds = self.position_embedding.repeat(n, 1, 1)  # N, 50, 8 --- The 3'rd axis would be repeated n times, 2nd axis value for 1 time i.e. not repeated, 3rd axis 1 time i.e. not repeated.
        embd_token = tokens + pos_embeds

        # 4) Transformer Blocks
        for enc_block in self.encoder_blocks:
            embd_token = enc_block(embd_token)

        # 5) Getting the classification token only and get classification result
        cls_token = embd_token[:, 0]
        cls_res = self.mlp(cls_token)

        return cls_res
