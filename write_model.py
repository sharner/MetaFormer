import os
import time
import argparse
import datetime
import numpy as np

import torch
from config import get_inference_config
from models import build_model

def parse_option():
    parser = argparse.ArgumentParser('Write eval model from checkpoint', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument('--checkpoint', help='MetaFormer checkpoint file',
                        required = True)
    parser.add_argument('--output', help='MetaFormer checkpoint file',
                        required = True)
    parser.add_argument('--image-size', type=int, help='Image size',
                        default=384)
    parser.add_argument('--nclasses', type=int, help='Number classes',
                        default = 1504)
    args, unparsed = parser.parse_known_args()

    config = get_inference_config(args)

    return args, config

if __name__ == '__main__':
    args, config = parse_option()
    config.defrost()
    config.MODEL.NUM_CLASSES = args.nclasses
    config.DATA.IMG_SIZE = args.image_size
    config.freeze()
    model = build_model(config)

    # read checkpoint file
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print("loaded {}; result {}".format(args.checkpoint, msg))

    # put model into evaluation mode
    model.eval()

    # write the model out again for inference
    torch.save(model.state_dict(), args.output)
    print("model rewritten to {}".format(args.output))
