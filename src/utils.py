import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import rcParams
rcParams["font.family"] = "serif"
rcParams["font.sans-serif"] = ["Palatino"]
import seaborn as sn
import pandas as pd
from matplotlib.ticker import NullLocator
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from skimage.draw import rectangle_perimeter


def rose_plot(ax, angles, bins=16, density=None, offset=0, lab_unit="degrees",
              start_zero=False, **param_dict):
    """
    Plot polar histogram of angles on ax. ax must have been created using
    subplot_kw=dict(projection='polar'). Angles are expected in radians.
    """
    # Wrap angles to [-pi, pi)
    angles = (angles + np.pi) % (2*np.pi) - np.pi

    # Set bins symetrically around zero
    if start_zero:
        # To have a bin edge at zero use an even number of bins
        if bins % 2:
            bins += 1
        bins = np.linspace(-np.pi, np.pi, num=bins+1)

    # Bin data and record counts
    count, bin = np.histogram(angles, bins=bins)

    # Compute width of each bin
    widths = np.diff(bin)

    # By default plot density (frequency potentially misleading)
    if density is None or density is True:
        # Area to assign each bin
        area = count / angles.size
        # Calculate corresponding bin radius
        radius = (area / np.pi)**.5
    else:
        radius = count

    # Plot data on ax
    ax.bar(bin[:-1], radius, zorder=1, align='edge', width=widths,
           edgecolor='C0', fill=False, linewidth=1)

    # Set the direction of the zero angle
    ax.set_theta_offset(offset)

    # Remove ylabels, they are mostly obstructive and not informative
    ax.set_yticks([])

    if lab_unit == "radians":
        label = ['$0$', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$',
                  r'$\pi$', r'$5\pi/4$', r'$3\pi/2$', r'$7\pi/4$']
        ax.set_xticklabels(label)

def similarity(x,y):
    sim = 0
    for i,a in enumerate(x.view(-1)):
        sim += torch.min(a,y.view(-1)[i])
    return sim

def corr_coef(x,y):

    x -= x.mean()
    y -= y.mean()

    return torch.sum(x*y)/(torch.sqrt(torch.sum(x ** 2)) * torch.sqrt(torch.sum(y ** 2)))

def entropy(y_pred):
    """
    Calculates the entropy
    :param y_pred: 2D array of shape #forwardpasses,#samples)
    """
    return -torch.sum(y_pred * torch.log(y_pred), 0)

def get_loc_target(var, channels=1, max=True):
    """
    :param channel: (int) number of channels in original image
    """


    B, C, H, W = var.shape
    var = var.reshape(len(var), C, -1).squeeze()

    if max:
        amax = var.argmax(-1)
    else:
        sum = var.sum(-1)
        if len(var.shape) > 1:
            sum = sum.unsqueeze(1)
        amax = torch.multinomial(var / sum, 1)

    y = amax / (channels * H)
    x = (amax / channels) % W
    z = amax % channels

    if len(x.shape) < 2:
        x = x.unsqueeze(1)
        y = y.unsqueeze(1)

    coords = torch.cat((x, y), 1)
    coords = normalize(H, coords.float())

    return coords.clamp(min=-1.0, max=1.0)



def convert_heatmap(var, map="YlOrRd_r"):
    cmap = plt.get_cmap(map)
    var = (var-var.min())/(var.max()-var.min())
    var = np.delete(cmap(var), 3, 2)
    return torch.from_numpy(np.transpose(var, (2, 0, 1)))


def normalize(T, coords):
    return (coords / T) * 2 - 1


def denormalize(T, coords):
    return (0.5 * ((coords + 1.0) * T)).long()



def vae_loss(x, mu, logvar, coeff=0.1):
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    KLD /= torch.prod(torch.tensor([*x.size()])).type(KLD.type())
    return coeff * KLD


def resize_array(x, size):
    # 3D and 4D tensors allowed only
    assert x.ndim in [3, 4], "Only 3D and 4D Tensors allowed!"

    # 4D Tensor
    if x.ndim == 4:
        res = []
        for i in range(x.shape[0]):
            img = array2img(x[i])
            img = img.resize((size, size))
            img = np.asarray(img, dtype="float32")
            img = np.expand_dims(img, axis=0)
            img /= 255.0
            res.append(img)
        res = np.concatenate(res)
        res = np.expand_dims(res, axis=1)
        return res

    # 3D Tensor
    img = array2img(x)
    img = img.resize((size, size))
    res = np.asarray(img, dtype="float32")
    res = np.expand_dims(res, axis=0)
    res /= 255.0
    return res


def img2array(data_path, desired_size=None, expand=False, view=False):
    """
    Util function for loading RGB image into a numpy array.
    Returns array of shape (1, H, W, C).
    """
    img = Image.open(data_path)
    img = img.convert("RGB")
    if desired_size:
        img = img.resize((desired_size[1], desired_size[0]))
    if view:
        img.show()
    x = np.asarray(img, dtype="float32")
    if expand:
        x = np.expand_dims(x, axis=0)
    x /= 255.0
    return x


def array2img(x):
    """
    Util function for converting anumpy array to a PIL img.
    Returns PIL RGB img.
    """
    if len(x) > 3:
        x = x.squeeze(0)
    x = x.permute(1, 2, 0).cpu().numpy()
    x = x + max(-np.min(x), 0)
    x_max = np.max(x)
    if x_max != 0:
        x /= x_max
    x *= 255
    if x.shape[-1] > 1:
        i = Image.fromarray(x.astype("uint8"), mode="RGB")
    else:
        i = Image.fromarray(np.squeeze(x).astype("uint8"), mode="L").convert("RGB")
    return i


def draw_locs(x, locs):
    locs = denormalize(x.shape[-1], locs)
    l = locs.flatten().tolist()
    img = array2img(x)
    draw = ImageDraw.Draw(img)

    for i,l in enumerate(locs):
        draw.ellipse([l[0]-1,l[1]-1,l[0]+1,l[1]+1],outline=tuple([int(c*255+.5) for c in plt.cm.hsv(i*20)[:-1]]))
    return torch.from_numpy(np.asarray(img, dtype="float32")).permute(2, 0, 1) / 255


def draw_glimpse(x, locs, extent):
    im = x.clone()
    locs = denormalize(im.shape[-1], locs).cpu()
    start = locs - extent // 2
    xx, yy = rectangle_perimeter(start, extent=extent, shape=im.shape[-2:])
    im[0, yy, xx] = 1
    return im

def create_heatmap(hmap, points):
    #hmap = torch.zeros(shape)
    for i,g in enumerate(points):
        for (x,y) in g:
            hmap[y,x] += 1
    return hmap / hmap.max()

def plot_heatmap(x, filename, cmap="bwr", vmin=None, vmax=None):
    f,ax = plt.subplots(1)
    ax.set_axis_off()
    handle = ax.imshow(x,cmap=cmap, vmin=vmin,vmax=vmax)
    f.colorbar(handle)
    f.savefig(filename, bbox_inches="tight", pad_inches=0)

def plot_images(reconstructed, target, locs, size, num, filename):

    # Create figure with sub-plots.
    l = len(reconstructed)
    fig, ax = plt.subplots(3, l, figsize=(l * 3 + 0.1, 9))

    t = array2img(target)
    for i in range(l):
        # plot the image
        r_mu = reconstructed[i].mean(0)
        r_var = reconstructed[i].var(0)
        loc = denormalize(r_mu.shape[-1], locs[i][0])

        ax[0, i].imshow(array2img(r_mu))
        ax[0, i].set_xticks([])
        ax[0, i].set_yticks([])
        ax[0, i].set_axis_off()
        ax[0, i].xaxis.set_major_locator(NullLocator())

        ax[1, i].imshow(t)
        ax[1, i].set_xticks([])
        ax[1, i].set_yticks([])
        ax[1, i].set_axis_off()
        ax[1, i].xaxis.set_major_locator(NullLocator())
        ax[1, i].yaxis.set_major_locator(NullLocator())
        s = size
        for j in range(num):
            rect = bounding_box(loc[0], loc[1], s)
            s *= 2
            ax[1, i].add_patch(rect)

        ax[2, i].imshow(r_var.squeeze(), interpolation="bilinear", cmap="plasma")
        ax[2, i].set_xticks([])
        ax[2, i].set_yticks([])
        ax[2, i].set_axis_off()
        ax[2, i].xaxis.set_major_locator(NullLocator())
        ax[2, i].yaxis.set_major_locator(NullLocator())

    fig.subplots_adjust(hspace=0, wspace=0)
    plt.margins(0.001, 0.001)
    plt.axis("off")
    fig.savefig(filename, bbox_inches="tight", pad_inches=0)


def plot_confusion_matrix(matrix, filename):
    fig = plt.figure(figsize=(10, 7))
    df_cm = pd.DataFrame(matrix.numpy(), range(matrix.size(0)), range(matrix.size(1)))
    sn.set(font_scale=1.1)  # for label size
    ax = sn.heatmap(
        df_cm, annot=True, annot_kws={"size": 16}, cmap="OrRd", fmt="g",vmin=0, vmax=50
    )  # font size
    ax.set(xlabel='Predicted', ylabel='True')
    fig.savefig(filename, bbox_inches="tight", pad_inches=0)
