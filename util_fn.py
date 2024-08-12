import torch
import numpy as np


def make_patches(images, n_patches):
    """

    :param images: take a batch of input images
    :param n_patches: number of patches per dimension (width, height). Total patch = n_patches * n_patches
    :return: Array of 1D patch embeddings.
    """

    n, c, h, w = images.shape  # n = batch_size, c = num_channels
    patch_size = h // n_patches  # As h == w, no need for w.
    patch_dimension = (h * w) // (n_patches * n_patches)

    assert h == w, "Patch embeddings require both height and weight to be equal"

    patches = torch.zeros((n, n_patches * n_patches,
                           patch_dimension))  # 2nd param = patch count per image = n_patches * n_patches. 3rd param to store the patches.

    for image_idx, image in enumerate(images):
        for i in range(n_patches):
            for j in range(n_patches):
                patch = image[:, i * patch_size: (i + 1) * patch_size,
                        j * patch_size: (j + 1) * patch_size]  # First indexing done for channel.
                patches[image_idx, i * n_patches + j] = patch.flatten()

    return patches


def positional_embedding(sequence_length, d):  # input shape =  49, 8. Output shape = 49, 8
    embed_res = torch.ones(sequence_length, d)

    for i in range(sequence_length):
        for j in range(d):
            embed_res[i][j] = np.sin(i / (10000 ** (j / d))) if j % 2 == 0 else np.cos(i / (10000 ** ((j - 1) / d)))

    return embed_res
