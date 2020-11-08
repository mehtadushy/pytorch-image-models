from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import time
import logging
import torch
import torch.nn as nn
import torch.nn.parallel
from collections import OrderedDict
import numpy as np

from timm.models import create_model, apply_test_time_pool, load_checkpoint, is_model, list_models
from timm.data import Dataset, DatasetTar, create_loader, resolve_data_config
from timm.utils import accuracy, AverageMeter, natural_key, setup_default_logging

from inplace_abn import InPlaceABN

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='PyTorch Network Timing')
parser.add_argument('--model', '-m', metavar='MODEL', default='dpn92',
                    help='model architecture (default: dpn92)')
parser.add_argument('--num-classes', type=int, default=1000,
                    help='Number classes in dataset')
parser.add_argument('--img-size', default=224, type=int,
                    metavar='N', help='Input image dimension, uses model default if empty')
parser.add_argument('--batch_size', default=1024, type=int,
                    metavar='N', help='Batch Size')
parser.add_argument(
        '--gpu-id', type=int, default=0,
        help='Which GPU to use.')
parser.add_argument(
        '--num-iter', type=int, default=50,
        help='Number of iterations to average over.')
parser.add_argument('--fp16', action='store_true', default=False,
                    help='Use half precision (fp16)')
parser.add_argument('--jit', action='store_true', default=False,
                    help='Use JIT')


def measure_cpu(model, x):
    # synchronize gpu time and measure fp
    model.eval()
    with torch.no_grad():
        t0 = time.time()
        y_pred = model(x)
        elapsed_fp_nograd = time.time()-t0
    return elapsed_fp_nograd

def measure_gpu(model, x):
    # synchronize gpu time and measure fp
    model.eval()
    with torch.no_grad():
        torch.cuda.synchronize()
        t0 = time.time()
        y_pred = model(x)
        torch.cuda.synchronize()
        elapsed_fp_nograd = time.time()-t0
    return elapsed_fp_nograd

def IABN2Float(module: nn.Module) -> nn.Module:
    "If `module` is IABN don't use half precision."
    if isinstance(module, InPlaceABN):
        module.float()
    for child in module.children(): IABN2Float(child)
    return module

def benchmark(model, gpu_id, num_classes, num_iter, img_size, fp16, jit, batch_size):
    # Import the model module
    net = create_model(
        model,
        num_classes=num_classes,
        in_chans=3,
        pretrained=False)

    if fp16:
        net = IABN2Float(net.half())
    param_count = sum([m.numel() for m in net.parameters()])
    #logging.info('Model %s created, param count: %d' % (model, param_count))
    print('Model %s created, param count: %d' % (model, param_count))

    device = torch.device("cuda:"+str(gpu_id) if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    print('\nEvaluating on GPU {}'.format(device))

    print('\nGPU, Batch Size: 1')
    x = torch.randn(1, 3, img_size, img_size)
    x = x.half() if fp16 else x
    #Warm up
    for i in range(10):
      _ = measure_gpu(net, x.to(device))
    fp = []
    for i in range(num_iter):
        t  = measure_gpu(net, x.to(device))
        fp.append(t)
    print('Model FP: '+str(np.mean(np.asarray(fp)*1000))+'ms')

    if jit:
        jit_net = torch.jit.trace(net, x.to(device))
        for i in range(10):
            _ = measure_gpu(jit_net, x.to(device))
        fp = []
        for i in range(num_iter):
            t  = measure_gpu(jit_net, x.to(device))
            fp.append(t)
        print('JIT FP: '+str(np.mean(np.asarray(fp)*1000))+'ms')


    print('\nGPU, Batch Size: 16')
    x = torch.randn(16, 3, img_size, img_size)
    x = x.half() if fp16 else x
    #Warm up
    for i in range(10):
        _ = measure_gpu(net, x.to(device))
    fp = []
    for i in range(num_iter):
        t  = measure_gpu(net, x.to(device))
        fp.append(t)
    print('Model FP: '+str(np.mean(np.asarray(fp)*1000))+'ms')

    if jit:
        jit_net = torch.jit.trace(net, x.to(device))
        for i in range(10):
            _ = measure_gpu(jit_net, x.to(device))
        fp = []
        for i in range(num_iter):
            t  = measure_gpu(jit_net, x.to(device))
            fp.append(t)
        print('JIT FP: '+str(np.mean(np.asarray(fp)*1000))+'ms')

    print('\nGPU, Batch Size: {}'.format(batch_size))
    x = torch.randn(batch_size, 3, img_size, img_size)
    x = x.half() if fp16 else x
    #Warm up
    for i in range(10):
        _ = measure_gpu(net, x.to(device))
    fp = []
    for i in range(num_iter):
        t  = measure_gpu(net, x.to(device))
        fp.append(t)
    print('Model FP: '+str(np.mean(np.asarray(fp)*1000))+'ms')

    if jit:
        jit_net = torch.jit.trace(net, x.to(device))
        for i in range(10):
            _ = measure_gpu(jit_net, x.to(device))
        fp = []
        for i in range(num_iter):
            t  = measure_gpu(jit_net, x.to(device))
            fp.append(t)
        print('JIT FP: '+str(np.mean(np.asarray(fp)*1000))+'ms')

    #device = torch.device("cpu")
    #print('\nEvaluating on {}'.format(device))
    #net = net.to(device)

    #print('\nCPU, Batch Size: 1')
    #x = torch.randn(1, 3, img_size, img_size)
    #x = x.half() if fp16 else x
    ###Warm up
    ##for i in range(10):
    ##    _ = measure_cpu(net, x.to(device))
    ##fp = []
    ##for i in range(num_iter):
    ##    t  = measure_cpu(net, x.to(device))
    ##    fp.append(t)
    ##print('Model FP: '+str(np.mean(np.asarray(fp)*1000))+'ms')

    #jit_net = torch.jit.trace(net, x.to(device))
    #for i in range(10):
    #    _ = measure_cpu(jit_net, x.to(device))
    #fp = []
    #for i in range(num_iter):
    #    t  = measure_cpu(jit_net, x.to(device))
    #    fp.append(t)
    #print('JIT FP: '+str(np.mean(np.asarray(fp)*1000))+'ms')



def main():
    # parse command line
    torch.manual_seed(1234)
    args = parser.parse_args()

    # run
    benchmark(**vars(args))

if __name__ == '__main__':
    main()
