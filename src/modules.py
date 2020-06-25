#Code is partly adapted from https://github.com/kevinzakka/recurrent-visual-attention

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from src.utils import *

import time
import pdb


class retina(object):
    """
    A retina that extracts a foveated glimpse 'phi'
    around location 'l' from an image 'x'. It encodes
    the region around 'l' at a high-resolution but uses
    a progressively lower resolution for pixels further
    from 'l', resulting in a compressed representation
    of the original image 'x'.
    Params
    ----------------
    :param x    : (array_like) a 4D Tensor of shape (B, H, W, C). The minibatch
      of images.
    :param l    : (array_like) a 2D Tensor of shape (B, 2). Contains normalized
      coordinates in the range [-1, 1].
    :param g    : (int) size of the first square patch.
    :param k    : (int) number of patches to extract in the glimpse.
    :param s    : (int) scaling factor that controls the size of successive patches.
    Returns
    ----------------
    :returns phi: (array_like) a 5D tensor of shape (B, k, g, g, C). The
      foveated glimpse of the image.
    """

    def __init__(self, g, k, s):
        self.g = g
        self.k = k
        self.s = int(s)

    def foveate(self, x, l):
        """
        Extract 'k' square patches of size 'g', centered
        at location 'l'. The initial patch is a square of
        size 'g', and each subsequent patch is a square
        whose side is 's' times the size of the previous
        patch.
        The 'k' patches are finally resized to (g, g) and
        concatenated into a tensor of shape (B, k, g, g, C).
        Params
        ----------------
        :param x    : (array_like) a 4D Tensor of shape (B, H, W, C). The minibatch
        :param l    : (array_like) a 2D Tensor of shape (B, 2).
        """

        self.glimpse = torch.zeros_like(x)
        self.cumulate = torch.zeros_like(x)

        batch_size = x.size(0)
        phi = torch.zeros(batch_size, self.k, self.g, self.g).type(x.type())
        size = self.g * self.s ** (self.k - 1)

        # extract k patches of decreasing size
        for i in range(self.k):
            k = size // self.g
            phi[:, i : i + 1] = self.extract_patch(x, l, size, resize=k)
            size = size // self.s

        return phi

    def extract_patch(self, x, l, size, resize):
        """
        Extract a single patch for each image in the
        minibatch 'x'.
        Params
        ----------------
        :param x        : (array_like) a 4D Tensor of shape (B, H, W, C). The minibatch
        :param l        : (array_like) a 2D Tensor of shape (B, 2).
        :param size     : (int) a scalar defining the size of the extracted patch.
        :param resize   : (int) a scalar defining by which factor to rescale the patch
        Returns
        ----------------
        :returns patch: (array_like) a 4D Tensor of shape (B, size, size, C)
        """
        B, C, H, W = x.shape

        # denormalize coords of patch center
        coords = denormalize(H, l)

        # compute top left corner of patch
        patch_x = coords[:, 0] - (size // 2)
        patch_y = coords[:, 1] - (size // 2)

        # loop through mini-batch and extract
        patch = []
        for i in range(B):
            im = x[i].unsqueeze(dim=0)
            tar  = self.glimpse[i].unsqueeze(dim=0)
            cum  = self.cumulate[i].unsqueeze(dim=0)
            T = im.shape[-1]

            # compute slice indices
            from_x, to_x = patch_x[i], patch_x[i] + size
            from_y, to_y = patch_y[i], patch_y[i] + size
            # cast to ints
            from_x, to_x = from_x.item(), to_x.item()
            from_y, to_y = from_y.item(), to_y.item()
            # pad tensor in case exceeds
            pad_dims = None
            if self.exceeds(from_x, to_x, from_y, to_y, T):
                pad_dims = (
                    size // 2 + 1,
                    size // 2 + 1,
                    size // 2 + 1,
                    size // 2 + 1,
                    0,
                    0,
                    0,
                    0,
                )
                im = F.pad(im, pad_dims, "constant", 0)
                tar = F.pad(tar, pad_dims, "constant", 0)
                cum = F.pad(cum, pad_dims, "constant", 0)
                #msk = F.pad(msk, pad_dims, "constant", 0)

                # add correction factor
                from_x += size // 2 + 1
                to_x += size // 2 + 1
                from_y += size // 2 + 1
                to_y += size // 2 + 1
            # and finally extract
            p = im[:, :, from_y:to_y, from_x:to_x]
            if resize > 1:
                p = F.avg_pool2d(p, resize)
                patch.append(p)
                p = F.interpolate(p, scale_factor=resize, mode='bilinear', align_corners=False)
            else:
                patch.append(p)
            tar[:, :, from_y:to_y, from_x:to_x] = p
            if pad_dims:
                self.glimpse[i] = tar[:,:,pad_dims[0]:-pad_dims[1],pad_dims[2]:-pad_dims[3]]
        # concatenate into a single tensor
        patch = torch.cat(patch)

        return patch

    def exceeds(self, from_x, to_x, from_y, to_y, T):
        """
        Check whether the extracted patch will exceed
        the boundaries of the image of size 'T'.
        Params
        ----------------
        :param from_x   : (int) lower x coordinate
        :param to_x     : (int) higher x coordinate
        :param from_y   : (int) lower y coordinate
        :param to_y     : (int) higher y coordinate
        :param T        : (int) size of the image
        Returns
        ----------------
        :returns : (bool) whether patch exceed image
        """
        if (from_x < 0) or (from_y < 0) or (to_x > T) or (to_y > T):
            return True
        return False


class glimpse_network(nn.Module):
    """
    A network that combines the "what" and the "where"
    into a glimpse feature vector 'g_t'.
    - "what": glimpse extracted from the retina.
    - "where": location tuple where glimpse was extracted.
    Concretely, feeds the output of the retina 'phi' to
    a fc layer and the glimpse location vector 'l_t_prev'
    to a fc layer. Finally, these outputs are fed each
    through a fc layer and their sum is rectified.
    In other words:
        'g_t = leaky_relu( fc( fc(l) ) + fc( fc(phi) ) )'
    Params
    ----------------
    :param h_g      : (int) hidden layer size of the fc layer for 'phi'.
    :param h_l      : (int) hidden layer size of the fc layer for 'l'.
    :param g        : (int) size of the square patches in the glimpses extracted by the retina.
    :param k        : (int) number of patches to extract per glimpse.
    :param s        : (int) scaling factor that controls the size of successive patches.
    :param c        : (int) number of channels in each image.
    :param x        : (array_like) a 4D Tensor of shape (B, C, H, W). The minibatch.
    :param l_t_prev : (array_like) a 2D tensor of shape (B, 2). Contains the glimpse coordinates [x, y] for the previous timestep 't-1'.
    :param bias     : (bool) whether to use bias in layers
    Returns
    ----------------
    :returns g_t: (array_like) a 2D tensor of shape (B, hidden_size). The glimpse
      representation returned by the glimpse network for the
      current timestep 't'.
    """

    def __init__(self, h_g, h_l, g, k, s, c, bias=False):
        super(glimpse_network, self).__init__()
        self.retina = retina(g, k, s)

        # glimpse layer
        D_in = k * g * g * c
        self.fc1 = nn.Linear(D_in, h_g[0], bias=bias)
        self.fc3 = nn.Linear(h_g[0], h_g[0] + h_l, bias=bias)
        self.fcnorm = nn.LayerNorm(h_g[0])

        # location layer
        D_in = 2
        self.fc2 = nn.Linear(D_in, h_l, bias=bias)
        self.fc4 = nn.Linear(h_l, h_g[0] + h_l, bias=bias)

        self.fcnorm1 = nn.LayerNorm(h_g[0] + h_l)
        self.fcnorm2 = nn.LayerNorm(h_g[0] + h_l)

        self.shape = 0

    def get_glimpse(self, x, l_t):
        return self.retina.foveate(x, l_t)

    def forward(self, x, l_t_prev):
        # generate glimpse phi from image x
        phi = self.get_glimpse(x, l_t_prev)
        self.glimpse = self.retina.glimpse
        self.shape = phi.shape[1:]

        # flatten location vector
        l_t_prev = l_t_prev.view(l_t_prev.size(0), -1)

        # feed phi and l to respective layers
        phi = F.leaky_relu(self.fc1(phi.view(phi.shape[0], -1)))

        l_out = F.leaky_relu(self.fc2(l_t_prev))

        what = self.fc3(phi)
        where = self.fc4(phi)

        # feed to fc layer
        g_t = F.leaky_relu(what + where)

        return g_t


class core_network(nn.Module):
    """
    An RNN that maintains an internal state that integrates
    information extracted from the history of past observations.
    It encodes the agent's knowledge of the environment through
    a state vector 'h_t' that gets updated at every time step 't'.
    Concretely, it takes the glimpse representation 'g_t' as input,
    and combines it with its internal state 'h_t_prev' at the previous
    time step, to produce the new internal state 'h_t' at the current
    time step.
    In other words:
        'h_t = leaky_relu( fc(h_t_prev) + fc(g_t) )'
    Params
    ----------------
    :param input_size   : (int) input size of the rnn.
    :param hidden_size   : (int) hidden size of the rnn.
    :param g_t           :  (array_like) a 2D tensor of shape (B, hidden_size). The glimpse
      representation returned by the glimpse network for the
      current timestep 't'.
    :param h_t_prev     : (array_like) a 2D tensor of shape (B, hidden_size). The
      hidden state vector for the previous timestep 't-1'.
    Returns
    ----------------
    :returns h_t: (array_like) a 2D tensor of shape (B, hidden_size). The hidden
      state vector for the current timestep 't'.
    """

    def __init__(self, input_size, hidden_size):
        super(core_network, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.rnn = torch.nn.RNNCell(input_size, hidden_size)

    def forward(self, g_t, h_t_prev):
        h_t = self.rnn(g_t, h_t_prev)
        return h_t


class latent_network(nn.Module):
    """
    Uses the internal state 'h_t' of the core network to
    produce latent mean and variance representation as well as reparameterization
    Params
    ----------------
    :param input_size   : (int) input size of the fc layer.
    :param hidden_size  : (int) hidden size of the fc layer.
    :param h_t          : (int) the hidden state vector of the core network for
      the current time step 't'.
    Returns
    ----------------
    :returns mu, logvar, z: (array_like) mean and variance vector and reparameterization
    """

    def __init__(self, input_size, hidden_size):
        super(latent_network, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size, bias=False)
        self.fc2 = nn.Linear(input_size, hidden_size, bias=False)

    def reparametrize(self, mu, logvar):
        std = torch.exp(logvar * 0.5)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, h_t):
        mu = self.fc1(h_t)
        logvar = self.fc2(h_t)
        z = self.reparametrize(mu, logvar)

        return mu, logvar, z


class decision_network(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(decision_network, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.fc1 = nn.Linear(input_size, hidden_size, bias=False)
        self.fc2 = nn.Linear(hidden_size, output_size, bias=False)

        self.norm1 = nn.LayerNorm(hidden_size)
        self.drop1 = nn.Dropout(0.4)

    def forward(self, latent):
        hid = self.drop1(self.norm1(F.leaky_relu(self.fc1(latent))))
        hid = self.fc2(hid)
        if self.output_size == 1:
            hid = torch.sigmoid(hid)
        return hid


class location_network(nn.Module):
    """
    Uses the internal state 'h_t' of the core network to
    produce the location coordinates 'l_t' for the next
    time step.
    Concretely, feeds the hidden state 'h_t' through a fc
    layer followed by a tanh to clamp the output beween
    [-1, 1].
    Params
    ----------------
    :param input_size   : (int) input size of the fc layer.
    :param output_size  : (int)output size of the fc layer.
    :param h_t          : (int) the hidden state vector of the core network for
      the current time step 't'.
    Returns
    ----------------
    :returns l_t: (array_like) a 2D vector of shape (B, 2).
    """

    def __init__(self, input_size, output_size, std=0):
        super(location_network, self).__init__()
        self.std = std
        self.fc = nn.Linear(input_size, output_size, bias=False)

    def forward(self, h_t):
        l_t = torch.tanh(self.fc(h_t.detach()))
        return l_t


class decoder_network(nn.Module):
    """
    Uses the internal state 'h_t' of the core network to
    predict the full input stimulus 'rec'
    Params
    ----------------
    :param input_size   : (int) input size of the fc layer.
    :param hidden_sizes : (array_like) list of hidden sizes
    :param output_size  : (array_like) size of the output.
    :param bias         : (bool) whether to use bias in layers
    :param h_t          : (int) the hidden state vector of the core network for the current time step 't'.
    Returns
    ----------------
    :returns rec: (array_like) a 4D vector of shape (B, C, H, W).
    """

    def __init__(self, input_size, hidden_sizes, output_size, bias=False):
        super(decoder_network, self).__init__()

        self.output_size = output_size
        self.fc1 = nn.Linear(input_size, hidden_sizes[0], bias=bias)
        self.fcnorm1 = nn.LayerNorm(hidden_sizes[0])

        self.lin = nn.Sequential(
            nn.Linear(hidden_sizes[0], hidden_sizes[1], bias=bias),
            nn.LeakyReLU(),
            nn.LayerNorm(hidden_sizes[1]),
            nn.Linear(
                hidden_sizes[1], self.output_size, bias=bias
            )
        )

    def forward(self, h_t):
        hid = self.fcnorm1(F.leaky_relu(self.fc1(h_t)))

        w = h = int(np.sqrt(self.output_size))
        hid = self.lin(hid)
        hid = hid.view(-1, 1, w, h)
        return torch.sigmoid(hid)
