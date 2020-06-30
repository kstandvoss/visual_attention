#Code is partly adapted from https://github.com/kevinzakka/recurrent-visual-attention

import math
import numpy as np

import torch
import torch.nn as nn

from torch.distributions import Normal
from torch.distributions.one_hot_categorical import OneHotCategorical
from src.modules import *
from src.utils import normalize
import pdb
import time
import logging
import itertools


class VariationalPredictiveAttention(nn.Module):
    """
    A Recurrent Model of Visual Attention (RAM) [1] combined with a
    Variational Autoencoder (VAE) [2].
    RAM is a recurrent neural network that processes
    inputs sequentially, attending to different locations
    within the image one at a time, and incrementally
    combining information from these fixations to build
    up a dynamic internal representation of the image.
    This representation is used to paramterize a probabilisitc latent
    distribution which is used to generate sample predictions
    of the full image. The uncertainty in these predictions is used
    to select the next attended location.
    References
    ----------
    - Minh et al., https://arxiv.org/abs/1406.6247
    - Kingma et al., https://arxiv.org/abs/1312.6114
    """

    def __init__(
        self,
        g,
        k,
        s,
        i,
        h_g,
        h_l,
        hidden_size,
        latent_size,
        dec_size,
        num_classes,
        bias=True,
        locator=False,
        add_location=False,
    ):
        """
        Initialize the recurrent attention model and its
        different components.
        Params
        ----------------
        :param g            : (int) size of the square patches in the glimpses extracted
        :param k            : (int) number of patches to extract per glimpse.
        :param s            : (int) scaling factor that controls the size of successive patches.
        :param i            : (array_like) vector of input image dimensions (C,H,W)
        :param h_g          : (int) hidden layer size of the fc layer for 'phi'.
        :param h_l          : (int) hidden layer size of the fc layer for 'l'.
        :param hidden_size  : (int) hidden size of the rnn.
        :param dec_size     : (array_like) sizes of the decoder layers
        :param num_classes  : (int) number of classes in the dataset.
        :param num_glimpses : (int) number of glimpses to take per image
        :param bias         : (bool) whether to use bias in convolutions
        :param locator      : (bool) whether to use locator network to determine next location
        :param add_location : (bool) whether to add location to decoder and latent network
        """
        super(VariationalPredictiveAttention, self).__init__()
        self.logger = logging.getLogger(__name__)
        self.i = i
        self.locator = locator
        self.latent_size = latent_size
        self.num_classes = num_classes
        add_latent = num_classes if num_classes > 1 else 0
        self.add_loc = 2 if add_location else 0
        self.out_size = i[-1] ** 2

        self.sensor = glimpse_network(h_g, h_l, g, k, s, i[0], bias=bias)
        self.rnn = core_network(h_g[0] + h_l, hidden_size)
        if self.locator:
            self.locator = location_network(hidden_size, 2)

        self.decision = decision_network(hidden_size, latent_size, self.num_classes)

        self.latent = latent_network(hidden_size + add_latent + self.add_loc, latent_size)
        self.decoder = decoder_network(
            latent_size + self.num_classes + self.add_loc, dec_size, self.out_size, bias=bias
        )

    def forward(
        self,
        x,
        l_t_prev,
        s_t_prev=None,
        samples=0,
        classify=False,
    ):
        """
        Run the recurrent attention model for 1 timestep
        on the minibatch of images 'x'.
        Params
        ----------------
        :param x        : (array_like) a 4D Tensor of shape (B, H, W, C). The minibatch.
        :param l_t_prev : (array_like) a 2D tensor of shape (B, 2). The location vector of previous time step.
        :param s_t_prev : (array_like) The hidde state vector for the previous timestep.
        :param samples  : (int) if >0, generate sample many forward predictions
        :param classify : (bool) whether to classify the input image
        Returns
        ----------------
        :returns h_t: (array_like) a 2D tensor of shape (B, hidden_size). The hidden state vector for the current timestep.
        :returns l_t: (array_like) a 2D tensor of shape (B, 2). The location vector.
        :returns r_t: (array_like) a 4D tensor of shape (B,C,H,W)
        :returns out_t: (array_like) a 2D tensor of shape (B,n_classes). The network classification.
        :returns mu: (array_like) mean for KL divergence
        :returns logvar: (array_like) logvariance for KL divergence
        """

        g_t = self.sensor(x, l_t_prev)

        h_t = self.rnn(g_t, s_t_prev)

        if self.locator:
            l_t = self.locator(h_t)
        else:
            l_t = l_t_prev.clone().detach()

        batch_size = h_t.size(0)

        r_t = torch.zeros(samples, batch_size, *self.i).type(h_t.type())
        out_t = torch.zeros(batch_size, self.num_classes).type(h_t.type())

        dist = None
        if classify:
            out_t = self.decision(h_t)
            if self.add_loc:
                attach = torch.cat([out_t, l_t],1)
            else:
                attach = out_t
        else:
            attach = torch.Tensor([]).type(x.type())

        # Generate forward samples for uncertainty estimate
        if self.num_classes > 1:
            mu, logvar, z = self.latent(torch.cat([h_t, attach], 1))
        else:
            mu, logvar, z = self.latent(h_t)

        for i in range(samples):
            r_t[i] = self.decoder(torch.cat([z, attach], 1))
            z = self.latent.reparametrize(mu,logvar)

        return h_t, l_t, r_t, out_t, mu, logvar

    def sample_image(self):
        """
        Generate image by sampling latent from prior distribution
        Returns
        ----------------
        :returns r_t: (array_like) sample image generate by decoder
        """
        z = torch.randn(1, self.latent_size)
        r_t = self.decoder(z)
        return r_t
