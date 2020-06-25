#Code is partly adapted from https://github.com/kevinzakka/recurrent-visual-attention

import torch
import torch.nn.functional as F
from torch import nn

import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import torchvision
from torch.nn.modules.distance import PairwiseDistance

import os
import time
import shutil
import pickle as pkl

from tqdm import tqdm
from src.model import VariationalPredictiveAttention
from tensorboardX import SummaryWriter

import numpy as np
from pathlib import Path
import logging

from shutil import rmtree

from matplotlib import pyplot as plt
from matplotlib import rcParams
rcParams["font.family"] = "serif"
rcParams["font.sans-serif"] = ["Palatino"]
from heatmappy import Heatmapper

plt.switch_backend("agg")
from src.utils import *


class Trainer(object):
    """
    """

    def __init__(self, data_loader, args):
        """
        Construct a new Trainer instance.
        Params
        ----------------
        :param args         : (Object) object containing arguments.
        :param data_loader  : (Object) data iterator
        """

        self.logger = logging.getLogger(__name__)
        logging.getLogger("PIL.PngImagePlugin").setLevel(logging.WARNING)

        self.args = args

        # glimpse network params
        self.patch_size = args.MODEL.GLIMPSE.PATCH_SIZE
        self.glimpse_scale = args.MODEL.GLIMPSE.SCALE
        self.num_patches = args.MODEL.GLIMPSE.NUM
        self.loc_hidden = args.MODEL.GLIMPSE.LOC
        self.glimpse_hidden = args.MODEL.GLIMPSE.GLIMPSE

        # core network params
        self.num_glimpses = args.MODEL.CORE.NUM
        self.hidden_size = args.MODEL.CORE.HIDDEN

        # latent network params
        self.latent_size = args.MODEL.LATENT.HIDDEN

        # decoder network params
        self.dec_size = args.MODEL.DECODER.HIDDEN

        # data params
        if args.TRAIN.IS_TRAIN:
            self.train_loader = data_loader[0]
            self.valid_loader = data_loader[1]
            self.num_train = args.TRAIN.NUM
            self.num_valid = len(self.valid_loader.dataset)
        else:
            self.test_loader = data_loader
            self.num_test = len(self.test_loader.dataset)

        self.gray = args.PRE_PROCESSING.GRAY
        num_channel = 2 * (not self.gray) + 1  # if gray ==1 else ==3
        self.im_size = (
            num_channel,
            args.PRE_PROCESSING.RESIZE,
            args.PRE_PROCESSING.RESIZE,
        )

        if args.CLASSIFY:
            self.classify = True
            self.num_classes = 10 if args.TARGET < 0 else 1
        else:
            self.classify = False
            self.num_classes = 0

        # training params
        self.epochs = args.TRAIN.EPOCHS
        self.start_epoch = 0
        self.momentum = args.TRAIN.MOMENTUM
        self.lr = args.TRAIN.INIT_LR
        self.is_test = not args.TRAIN.IS_TRAIN
        self.decay = args.TRAIN.WEIGHT_DECAY

        # misc params
        self.use_gpu = args.GPU
        self.ckpt_dir = Path(args.CKPT_DIR)
        self.logs_dir = Path(args.TENSORBOARD_DIR)
        self.best_valid_loss = np.inf
        self.counter = 0
        self.lr_patience = args.TRAIN.LR_PATIENCE
        self.train_patience = args.TRAIN.TRAIN_PATIENCE
        self.use_tensorboard = args.USE_TENSORBOARD
        self.resume = args.RESUME
        self.name = args.NAME
        self.model_name = f"model_{self.num_glimpses}_{self.patch_size}x{self.patch_size}_{self.glimpse_scale}_{self.name}"
        self.use_amax = args.AMAX
        self.samples = args.FORWARD
        self.klf = args.TRAIN.KL_FACTOR
        self.clf = args.TRAIN.CL_FACTOR
        self.frac = args.TRAIN.FRAC_LABELS
        self.batch_size = None

        self.plot_dir = Path("./plots/") / self.model_name
        if not self.plot_dir.exists():
            self.plot_dir.mkdir(parents=True)

        if not args.TRAIN.IS_TRAIN:
            self.num_glimpses = args.TEST.NUM

        # tensorboard logging
        if self.use_tensorboard:
            self.tensorboard_dir = self.logs_dir / self.model_name
            self.logger.info(f"[*] Saving tensorboard logs to {self.tensorboard_dir}")
            if not self.tensorboard_dir.exists():
                self.tensorboard_dir.mkdir(parents=True)
            else:
                if self.resume or self.is_test:
                    pass
                else:
                    for x in self.tensorboard_dir.iterdir():
                        if not x.is_dir():
                            x.unlink()
            self.writer = SummaryWriter(self.tensorboard_dir)

        # How many images to save in tensorboard
        self.num_save = 4

        self.logger.debug("Create model")
        # build model
        self.model = VariationalPredictiveAttention(
            self.patch_size,
            self.num_patches,
            self.glimpse_scale,
            self.im_size,
            self.glimpse_hidden,
            self.loc_hidden,
            self.hidden_size,
            self.latent_size,
            self.dec_size,
            self.num_classes,
            args.BIAS,
            args.MODEL.GLIMPSE.SAMPLE,
        )
        self.logger.debug("Model created")

        if self.use_gpu:
            self.model.cuda()

        self.logger.info(
            f"[*] Number of model parameters: {sum([p.data.nelement() for p in self.model.parameters()]):,}"
        )

        self.logger.info("Model:")
        self.logger.info(self.model)

        # initialize optimizer and scheduler
        if args.TRAIN.OPTIM == "sgd":
            self.optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.lr,
                momentum=self.momentum,
                nesterov=True,
                weight_decay=self.decay,
            )
            self.schedule = True
        elif args.TRAIN.OPTIM == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.lr,
                weight_decay=self.decay,
                amsgrad=True,
            )
            self.schedule = False
        elif args.TRAIN.OPTIM == "rmsprop":
            self.optimizer = optim.RMSprop(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.lr,
                weight_decay=self.decay,
            )
            self.schedule = True

        steps = args.TRAIN.SCHEDULE
        self.scheduler = MultiStepLR(self.optimizer, steps, gamma=0.7)

        if args.TRAIN.LOSS == "mse":
            self.rec_crit = nn.MSELoss()
        elif args.TRAIN.LOSS == "bce":
            self.rec_crit = nn.BCELoss()
        self.criterion = vae_loss

    def reset(self):
        """
        Initialize the hidden state of the core network
        and the location vector.
        """
        h_t = torch.randn(self.batch_size, self.hidden_size, requires_grad=False)
        c_t = torch.randn(self.batch_size, self.hidden_size, requires_grad=False)
        l_t = torch.Tensor(self.batch_size, 2).uniform_(-1, 1)

        if self.use_gpu:
            h_t, c_t = h_t.cuda(), c_t.cuda()
            l_t = l_t.cuda()

        return h_t, c_t, l_t

    def train(self):
        """
        Train the model on the training set.
        """
        # load the most recent checkpoint
        if self.resume:
            self.load_checkpoint(best=False)

        self.logger.info(
            f"\n[*] Train on {self.num_train} samples, validate on {self.num_valid} samples"
        )
        for epoch in range(self.start_epoch, self.epochs):

            # train for 1 epoch
            train_loss, imgs = self.train_one_epoch(epoch)

            # evaluate on validation set
            valid_loss, valid_acc = self.validate(epoch)

            if self.schedule:
                # # reduce lr if validation loss plateaus
                self.scheduler.step()
                self.logger.info(
                    f"\nEpoch: {epoch+1}/{self.epochs} - LR: {self.scheduler.get_lr()[0]:.6f}"
                )
            else:
                self.logger.info(
                    f"\nEpoch: {epoch+1}/{self.epochs} - LR: {self.lr:.6f}"
                )

            is_best = valid_loss < self.best_valid_loss
            msg1 = "train loss: {:.3f} "
            msg2 = "- val loss: {:.3f} "
            msg3 = "- val acc: {:.3f} "

            if is_best:
                self.counter = 0
                msg2 += " [*]"
            msg = msg1 + msg2 + msg3
            self.logger.info(msg.format(train_loss, valid_loss, valid_acc))

            # check for improvement
            if not is_best:
                self.counter += 1
            if self.counter > self.train_patience:
                self.logger.info("[!] No improvement in a while, stopping training.")
                return

            self.best_valid_loss = min(valid_loss, self.best_valid_loss)

            self.save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "model_state": self.model.state_dict(),
                    "optim_state": self.optimizer.state_dict(),
                    "best_valid_loss": self.best_valid_loss,
                },
                is_best,
            )
            if self.use_tensorboard:
                self.writer.add_image(
                    "reconstruction",
                    torchvision.utils.make_grid(
                        torch.cat(imgs, 0),
                        nrow=self.num_save,
                        normalize=True,
                        pad_value=1,
                        padding=3,
                        scale_each=True,
                    ),
                    epoch,
                )

    def perform_glimpses(self, x, locs, num=None):
        """
        Performs a series of glimpses
        Params
        ----------------
        :param x : (array_like) input to the model
        :param locs : (array_like) empty array to store the locations
        :param num: (int) number of glimpses
        """
        n = num if num else self.num_glimpses
        # initialize location vector and hidden state
        s_t, _, l_t = self.reset()
        for t in range(n):
            # forward pass through model
            locs[:, t] = l_t.clone().detach()

            s_t, l_pred, pred, classification, mu, logvar = self.model(
                x, l_t, s_t, samples=self.samples, classify=self.classify,
            )
            l_t = get_loc_target(pred.var(0), max=self.use_amax).type(l_t.type())

        return pred, classification, mu, logvar

    def compute_recloss(self, recon, target):
        """
        Compute reconstruction loss
        ----------------
        :param recon : (array_like) reconstruction of the input
        :param target: (array_like) input image
        """
        rec_loss = 0
        for s in range(self.samples):
            rec_loss += self.rec_crit(recon[s], target) / self.samples

        return rec_loss

    def compute_classloss(self, classification, labels):
        """
        Compute classification loss
        ----------------
        :param classification : (array_like) reconstruction of the input
        :param labels          : (array_like) target class
        """
        if self.classify and labels is not None:
            if self.num_classes == 1:
                target = (labels.argmax(1) == self.target_class).float()
                class_loss = F.binary_cross_entropy(classification.squeeze(), target)
                accuracy = (
                    torch.sum(classification.detach().squeeze().round() == target).float() / len(classification)
                ).item()
            else:
                class_loss = F.cross_entropy(classification, labels.argmax(1))
                accuracy = (
                    torch.sum(
                        classification.detach().argmax(1) == labels.detach().argmax(1)
                    ).float()
                    / len(labels)
                ).item()
        else:
            class_loss = 0
            accuracy = None

        return class_loss, accuracy

    def train_one_epoch(self, epoch):
        """
        Train the model for 1 epoch of the training set.
        Params
        ----------------
        :param epoch : (int) epoch number
       """
        self.model.train()
        imgs = []
        store = {"comb": [], "rec": [], "kl": [], "acc": []}
        tic = time.time()
        use_labeled = 0
        with tqdm(total=self.num_train) as pbar:
            for i, (u, x, y) in enumerate(self.train_loader):
                if self.use_gpu:
                    u = None if not len(u) else u.float().cuda()
                    x = None if not len(x) else x.float().cuda()
                    y = None if not len(y) else y.float().cuda()

                # initialize location vector and hidden state
                if u is not None:
                    self.batch_size = u.shape[0]
                else:
                    self.batch_size = x.shape[0]
                s_t, _, l_t = self.reset()
                # extract the glimpses
                num = self.num_glimpses  #np.random.randint(3,9) #
                locs = torch.zeros(self.batch_size, num, 2)


                loss = 0
                rec_loss_u = 0
                kl_factor = np.tanh(self.klf * epoch)
                if self.frac < 1:
                    pred, classification, mu, logvar = self.perform_glimpses(u, locs, num=num)

                    rec_loss_u = self.compute_recloss(pred, u)
                    kl_loss = self.criterion(u, mu, logvar, coeff=1)
                    loss = kl_factor * kl_loss + rec_loss_u
                    target = u.detach()

                rec_loss_l = 0
                accuracy = None
                if self.classify:
                    pred, classification, mu, logvar = self.perform_glimpses(x, locs, num=num)

                    # compute gradients
                    rec_loss_l = self.compute_recloss(pred, x)
                    class_loss, accuracy = self.compute_classloss(classification, y)
                    kl_loss = self.criterion(x, mu, logvar, coeff=1)

                    class_factor = np.tanh(self.clf * epoch)
                    loss += (kl_factor * kl_loss + class_factor*class_loss + rec_loss_l)
                    target = x.detach()

                # update model
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                self.optimizer.step()
                self.optimizer.zero_grad()

                # measure elapsed time
                toc = time.time()

                # store
                rec_loss = torch.mean(rec_loss_u + rec_loss_l)
                store["comb"].append(kl_loss.item() + rec_loss.item())
                store["rec"].append(rec_loss.item())
                store["kl"].append(kl_loss.item())
                if accuracy is not None:
                    store["acc"].append(accuracy)

                pbar.set_description(
                    f"{toc-tic:.1f}s - loss: {np.mean(store['comb']):.3f}, acc: {np.mean(store['acc']):.3f}"
                )
                pbar.update(self.batch_size)

            # log to tensorboard
            if self.use_tensorboard:
                iteration = epoch * len(self.train_loader) + i
                self.writer.add_scalar("train_loss", np.mean(store["comb"]), iteration)
                self.writer.add_scalar("rec_loss", np.mean(store["rec"]), iteration)
                self.writer.add_scalar("kl_loss", np.mean(store["kl"]), iteration)
                self.writer.add_scalar("class_loss", np.mean(store["acc"]), iteration)

            # store images + reconstructions of largest scale
            num = self.num_save
            pred = pred.detach()

            imgs.append(target[:num].expand(-1, 3, -1, -1))
            imgs.append(pred[0, :num].expand(-1, 3, -1, -1))

            heatmap = torch.zeros(num, 3, *target.shape[-2:]).type(target.type())
            glimpses = torch.zeros_like(heatmap)
            for i in range(num):
                heatmap[i] = convert_heatmap(pred.var(0)[i].squeeze().cpu())
                glimpses[i] = draw_locs(target[i], locs[i])
            imgs.append(heatmap)
            imgs.append(glimpses)

            return np.mean(store["comb"]), imgs

    def validate(self, epoch):
        """
        Evaluate the model on the validation set.
        Params
        ----------------
        :param epoch : (int) epoch number
        """

        self.model.eval()
        with torch.no_grad():
            losses = []
            accuracies = []
            for i, (_, x, y) in enumerate(self.valid_loader):
                if self.use_gpu:
                    x = x.cuda()
                    y = y.cuda()

                # initialize location vector and hidden state
                self.batch_size = x.shape[0]
                locs = torch.zeros(self.batch_size, self.num_glimpses, 2)
                pred, classification, mu, logvar = self.perform_glimpses(x, locs, None)

                # compute gradients
                rec_loss = self.compute_recloss(pred, x)
                _, accuracy = self.compute_classloss(classification, y)

                loss = rec_loss

                # store
                losses.append(loss.item())
                if accuracy is not None:
                    accuracies.append(accuracy)

            # log to tensorboard
            if self.use_tensorboard:
                iteration = epoch * len(self.valid_loader) + i
                self.writer.add_scalar("val_loss", np.mean(losses), iteration)
                self.writer.add_scalar("val_acc", np.mean(accuracies), iteration)

            return np.mean(losses), np.mean(accuracies)

    def test(self, add_path="", heatmap=True, confusion_matrix=True):
        """
        Test the model on the held-out test data.
        """
        # load the best checkpoint
        self.load_checkpoint(best=True)
        self.model.eval()
        with torch.no_grad():

            losses = []
            accuracies = []

            confusion_matrix = torch.zeros(self.num_classes, self.num_classes)
            heat_points = [[[] for i in range(self.num_glimpses)] for j in range(self.num_classes+1)]
            heat_mean = torch.zeros(self.num_classes, *self.im_size)

            var_means = torch.zeros(self.num_glimpses,len(self.test_loader))
            loss_means = torch.zeros(self.num_glimpses,len(self.test_loader))
            count_hits = torch.zeros(self.num_glimpses,2)
            var_dist = torch.zeros(len(self.test_loader.dataset),self.num_glimpses,self.im_size[-1],self.im_size[-1])
            err_dist = torch.zeros(len(self.test_loader.dataset),self.num_glimpses,self.im_size[-1],self.im_size[-1])
            labels = torch.zeros(len(self.test_loader.dataset))

            dist = PairwiseDistance(2)
            dists = torch.zeros(len(self.test_loader.dataset),self.num_glimpses)

            run_idx = 0
            for i, (_, x, y) in enumerate(self.test_loader):
                if self.use_gpu:
                    x = x.cuda()
                    y = y.cuda()

                true = y.detach().argmax(1)
                # initialize location vector and hidden state
                self.batch_size = x.shape[0]
                s_t, _, l_t = self.reset()

                targets = torch.zeros(self.num_glimpses, 3, *x.shape[2:])
                glimpses = torch.zeros(self.num_glimpses, 3, *x.shape[2:])
                means = torch.zeros(self.num_glimpses, 3, *x.shape[2:])
                vars = torch.zeros(self.num_glimpses, 3, *x.shape[2:])

                locs = torch.zeros(self.batch_size, self.num_glimpses, 2)
                acc_loss = 0
                sub_dir = self.plot_dir / f"{i:06d}_dir"
                if not sub_dir.exists():
                    sub_dir.mkdir()

                l_t = torch.zeros(self.batch_size, 2).type(l_t.type())
                for t in range(self.num_glimpses):
                    # forward pass through model
                    locs[:, t] = l_t.clone().detach()
                    s_t, l_pred, r_t, out_t, mu, logvar = self.model(
                        x,
                        l_t,
                        s_t,
                        samples=self.samples,
                        classify=self.classify,
                    )
                    draw = x[0].expand(3, -1, -1)
                    extent = self.patch_size
                    for patch in range(self.num_patches):
                        draw = draw_glimpse(
                            draw, l_t[0], extent=extent,
                        )
                        extent *= self.glimpse_scale

                    targets[t] = draw
                    glimpses[t] = self.model.sensor.glimpse[0][0].expand(3, -1, -1)
                    means[t] = r_t[0][0].expand(3, -1, -1)
                    vars[t] = convert_heatmap(r_t.var(0)[0].squeeze().cpu().numpy())

                    if t==0:
                        var_first = r_t.var(0).squeeze().cpu().numpy()
                    var_last = r_t.var(0).squeeze().cpu().numpy()

                    loc_prev = loc = denormalize(x.size(-1),l_t)
                    l_t = get_loc_target(r_t.var(0), max=self.use_amax).type(l_t.type())
                    loc = denormalize(x.size(-1),l_t)
                    dists[run_idx:run_idx+len(x),t] = dist.forward(loc_prev.float(),loc.float()).detach().cpu()
                    var_dist[run_idx:run_idx+len(x),t] = r_t.var(0).detach().cpu().squeeze()

                    temp_loss = torch.zeros(self.batch_size,*err_dist.shape[2:])
                    for s in range(self.samples):
                        temp_loss += F.binary_cross_entropy(r_t[s],x,reduction='none').detach().cpu().squeeze() / self.samples
                    err_dist[run_idx:run_idx+len(x),t] = temp_loss


                    for k,l in enumerate(loc):
                        if x.squeeze()[k][l[0],l[1]] > 0:
                            count_hits[t,0] +=1
                        else:
                            count_hits[t,1] +=1

                    imgs = [targets[t], glimpses[t], means[t], vars[t]]
                    filename = sub_dir / (add_path + f"{i:06d}_glimpse{t:01d}.png")
                    if i == 1:
                        torchvision.utils.save_image(
                            imgs,
                            filename,
                            nrow=1,
                            normalize=True,
                            pad_value=1,
                            padding=1,
                            scale_each=True,
                        )

                    var = r_t.var(0).reshape(self.batch_size,-1)
                    var_means[t,i] = var.sum(-1).mean(0)
                    loss_means[t,i] = self.compute_recloss(r_t,x)

                    for j in range(self.num_classes):
                        heat_points[j][t].extend(
                            denormalize(x.size(-1), l_t[true == j]).tolist()
                        )

                    heat_points[-1][t].extend(loc.tolist())

                labels[run_idx:run_idx+len(true)] = true
                run_idx += len(x)
                for s in range(self.samples):
                    loss = F.mse_loss(r_t[s], x) / self.samples
                    acc_loss += loss.item()  # store

                if self.classify:
                    if self.num_classes == 1:
                        pred = out_t.detach().squeeze().round()
                        target = (y.argmax(1) == self.target_class).float()
                    else:
                        pred = out_t.detach().argmax(1)
                        target = true
                    accuracies.append(torch.sum(pred == target).float().item() / len(y))
                    for p, tr in zip(pred.view(-1), true.view(-1)):
                        confusion_matrix[tr.long(), p.long()] += 1
                for j in range(self.num_classes):
                    if len(x[true == j]):
                        heat_mean[j] += x[true == j].mean(0).cpu()

                losses.append(acc_loss)
                imgs = [targets, glimpses, means, vars]

                for im in imgs:
                    im = (im - im.min()) / (im.max()-im.min())
                # store images + reconstructions of largest scale
                filename = self.plot_dir / (add_path + f"{i:06d}.png")
                if i == 1:
                    torchvision.utils.save_image(
                        torch.cat(imgs, 0),
                        filename,
                        nrow=6,
                        normalize=False,
                        pad_value=1,
                        padding=3,
                        scale_each=False,
                    )


            if self.classify:
                sub_dir = Path('class')
            else:
                sub_dir = Path('no_class')
            pkl.dump(dists,open(sub_dir/'dists.pkl','wb'))
            pkl.dump(var_dist,open(sub_dir/'var_dist.pkl','wb'))
            pkl.dump(err_dist,open(sub_dir/'err_dist.pkl','wb'))
            pkl.dump(heat_points,open(sub_dir/'locs.pkl','wb'))
            pkl.dump(labels,open(sub_dir/'labels.pkl','wb'))
            pkl.dump(heat_mean,open(sub_dir/'mean.pkl','wb'))
            print(count_hits[:,0]/count_hits.sum(1))

            f = plt.figure()
            plt.title('Uncertainty over saccades')
            plt.errorbar(range(self.num_glimpses),var_means.mean(-1).cpu().numpy(), yerr=var_means.std(-1).cpu().numpy(),marker='o',c=(0.04,.34,.57))
            plt.xlabel('Saccade number')
            plt.ylabel('Summed model variance')
            f.tight_layout()
            f.savefig(self.plot_dir / f"uncertainty.pdf", bbox_inches="tight")

            f = plt.figure()
            plt.title('Prediction error over saccades')
            plt.errorbar(range(self.num_glimpses),loss_means.mean(-1).cpu().numpy(), yerr=loss_means.std(-1).cpu().numpy(),marker='o',c=(0.04,.34,.57))
            plt.xlabel('Saccade number')
            plt.ylabel('Binary Cross Entropy')
            plt.ylim(0.,0.6)
            f.tight_layout()
            f.savefig(self.plot_dir / f"pred_error.pdf", bbox_inches="tight")

            f, ax = plt.subplots(1,2, figsize=(8,3))
            ax[0].errorbar(range(self.num_glimpses),loss_means.mean(-1).cpu().numpy(), yerr=loss_means.std(-1).cpu().numpy(),marker='o',c=(0.04,.34,.57))
            ax[0].set_ylim(0.,0.6)
            ax[0].set_ylabel('Binary Cross Entropy')
            ax[0].set_xlabel('Saccade number')
            ax[1].errorbar(range(self.num_glimpses),var_means.mean(-1).cpu().numpy(), yerr=var_means.std(-1).cpu().numpy(),marker='o',c=(0.04,.34,.57))
            ax[1].set_xlabel('Saccade number')
            ax[1].set_ylabel('Summed model variance')
            ax[1].set_yticklabels(['  0','  1','  2','  3','  4','  5'])
            f.tight_layout()
            f.savefig(self.plot_dir / f"comb_var_err.pdf", bbox_inches="tight")

            print("#######################")
            if self.classify:
                print(confusion_matrix)
                print(confusion_matrix.diag() / confusion_matrix.sum(1))
                plot_confusion_matrix(
                    confusion_matrix, self.plot_dir / f"confusion_matrix.png"
                )
                pkl.dump(confusion_matrix,open(sub_dir/'conf_matrix.pkl','wb'))

            #create heatmaps
            flatten = lambda l,i: [item for sublist in l for item in sublist[i]]
            for i in range(self.num_classes):
                img = array2img(heat_mean[i])
                points = np.array(heat_points[i])

                first = points[:3].reshape(-1,2)
                heatmapper = Heatmapper(
                    point_diameter=1,  # the size of each point to be drawn
                    point_strength=0.3,  # the strength, between 0 and 1, of each point to be drawn
                    opacity=0.85,
                )
                heatmap = heatmapper.heatmap_on_img(first, img)
                heatmap.save(self.plot_dir / f"heatmap_bef{i}.png")

                last = points[3:-1].reshape(-1,2)
                heatmap = heatmapper.heatmap_on_img(last, img)
                heatmap.save(self.plot_dir / f"heatmap_aft{i}.png")

                for j in range(self.num_glimpses):
                    heatmap = heatmapper.heatmap_on_img(heat_points[i][j], img)
                    heatmap.save(self.plot_dir / f"heatmap_class{i}_glimpse{j}.png")


            self.logger.info(
                f"[*] Test loss: {np.mean(losses)}, Test accuracy: {np.mean(accuracies)}"
            )
            return np.mean(losses)

    def save_checkpoint(self, state, is_best):
        """
        Save a copy of the model so that it can be loaded at a future
        date. This function is used when the model is being evaluated
        on the test data.
        If this model has reached the best validation accuracy thus
        far, a seperate file with the suffix `best` is created.
        Params
        ----------------
        :param state    : (Object) network state
        :param is_best  : (bool) best model so far
        """
        print("[*] Saving model {} to {}".format(self.model_name, self.ckpt_dir))

        ckpt_path = self.ckpt_dir / (self.model_name + "_ckpt.pth.tar")
        torch.save(state, str(ckpt_path))

        if is_best:
            shutil.copyfile(
                ckpt_path, self.ckpt_dir / (self.model_name + "_model_best.pth.tar")
            )

    def load_checkpoint(self, best=False):
        """
        Load the best copy of a model. This is useful for 2 cases:
        - Resuming training with the most recent model checkpoint.
        - Loading the best validation model to evaluate on the test data.
        Params
        ------
        :param best: (bool) if set to True, loads the best model. Use this if you want
          to evaluate your model on the test data. Else, set to False in
          which case the most recent version of the checkpoint is used.
        """
        self.logger.info(f"[*] Loading model from {self.ckpt_dir}")

        filename = self.model_name + "_ckpt.pth.tar"
        if best:
            filename = self.model_name + "_model_best.pth.tar"
        ckpt_path = self.ckpt_dir / filename
        map_location = 'gpu' if self.use_gpu else 'cpu'
        ckpt = torch.load(ckpt_path, map_location=torch.device(map_location))

        # load variables from checkpoint
        self.start_epoch = ckpt["epoch"]
        self.best_valid_loss = ckpt["best_valid_loss"]
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt['optim_state'])

        if best:
            self.logger.info(
                f"[*] Loaded {filename} checkpoint @ epoch {ckpt['epoch']} "
                f"with best valid loss of {ckpt['best_valid_loss']:.3f}"
            )
        else:
            self.logger.info(
                f"[*] Loaded {filename} checkpoint @ epoch {ckpt['epoch']}"
            )
