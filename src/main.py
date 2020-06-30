#Code is partly adapted from https://github.com/kevinzakka/recurrent-visual-attention

import torch

from src.trainer import Trainer
from src.data_loader import get_data_loader
from src.utils import *
from src.config import cfg_from_file, cfg_from_list, cfg_set_log_file

import argparse
import pprint
import numpy as np
import logging
import logging.config
import sys



def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="Train a network")
    parser.add_argument(
        "-u", "--use_gpu", dest="gpu", help="Use GPU", default=0, type=int
    )
    parser.add_argument(
        "-c",
        "--cfg",
        dest="cfg_file",
        help="optional config file",
        default=None,
        type=str,
    )
    parser.add_argument(
        "-s",
        "--set",
        dest="set_cfgs",
        help="set config keys",
        default=None,
        nargs=argparse.REMAINDER,
    )

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


def main(config):
    """
    Main
    """
    data_loader = get_data_loader(
        config.DATA_DIR,
        config.TRAIN.BATCH_SIZE,
        config.SEED,
        config.TRAIN.NUM,
        config.TRAIN.VAL_NUM,
        config.TRAIN.IS_TRAIN,
        config.TRAIN.NUM_WORKERS,
        config.GPU,
        config.TRAIN.FRAC_LABELS,
    )


    logger.debug("Calling trainer")
    # instantiate trainer
    trainer = Trainer(data_loader, config)

    logger.debug("Start training")
    # either train
    if config.TRAIN.IS_TRAIN:
        trainer.train()
    # or load a pretrained model and test
    else:
        trainer.test()


if __name__ == "__main__":
    args = parse_args()

    print("Called with args:")
    print(args)

    if args.cfg_file is not None:
        cfg = cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg = cfg_from_list(args.set_cfgs,cfg)

    if args.gpu:
        cfg.GPU = True

    # logging
    cfg_set_log_file(cfg)
    logging.config.dictConfig(cfg.LOGGING)
    # pdb.set_trace()
    logger = logging.getLogger(__name__)

    logger.info("Using config:")
    logger.info(pprint.pformat(cfg))

    # fix the random seeds
    np.random.seed(cfg.SEED)
    torch.manual_seed(cfg.SEED)

    if args.gpu:
        torch.cuda.manual_seed(cfg.SEED)

    main(cfg)
