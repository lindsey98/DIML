"""==================================================================================================="""
################### LIBRARIES ###################
### Basic Libraries
# import comet_ml
import warnings
warnings.filterwarnings("ignore")

import os, sys, numpy as np, argparse, imp, datetime, pandas as pd, copy
import time, pickle as pkl, random, json, collections
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from evaluation.metrics import get_metrics, get_metrics_rank
from tqdm import tqdm, trange
import shutil

import parameters    as par
import torch.nn.functional as F
from utilities.diml import Sinkhorn, calc_similarity
import dataset
import utils

"""==================================================================================================="""
################### INPUT ARGUMENTS ###################
parser = argparse.ArgumentParser()

parser = par.basic_training_parameters(parser)
parser = par.batch_creation_parameters(parser)
parser = par.batchmining_specific_parameters(parser)
parser = par.loss_specific_parameters(parser)

##### Read in parameters
opt = parser.parse_args()

"""==================================================================================================="""
opt.savename = opt.group + '_s{}'.format(opt.seed)

"""==================================================================================================="""
### Load Remaining Libraries that neeed to be loaded after comet_ml
import torch, torch.nn as nn
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import architectures as archs
import datasampler   as dsamplers
import datasets      as datasets
import criteria      as criteria
import batchminer    as bmine
import evaluation    as eval
from utilities import misc
from utilities import logger

"""==================================================================================================="""
full_training_start_time = time.time()

"""==================================================================================================="""
# opt.source_path += '/'+opt.dataset
opt.save_path   += '/{}_{}_{}'.format(opt.dataset, opt.embed_dim, opt.loss)
print(opt.save_path)

#Assert that the construction of the batch makes sense, i.e. the division into class-subclusters.
assert not opt.bs%opt.samples_per_class, \
    'Batchsize needs to fit number of samples per class for distance sampling and margin/triplet loss!'
opt.pretrained = not opt.not_pretrained

"""==================================================================================================="""
################### GPU SETTINGS ###########################
os.environ["CUDA_DEVICE_ORDER"]   ="PCI_BUS_ID"

"""==================================================================================================="""
#################### SEEDS FOR REPROD. #####################
torch.backends.cudnn.deterministic=True; np.random.seed(opt.seed); random.seed(opt.seed)
torch.manual_seed(opt.seed); torch.cuda.manual_seed(opt.seed); torch.cuda.manual_seed_all(opt.seed)
"""==================================================================================================="""
##################### NETWORK SETUP ##################
opt.device = torch.device('cuda')
model = archs.select(opt.arch, opt)

if opt.fc_lr < 0:
    to_optim   = [{'params':model.parameters(),'lr':opt.lr,'weight_decay':opt.decay}]
else:
    all_but_fc_params = [x[-1] for x in list(filter(lambda x: 'last_linear' not in x[0], model.named_parameters()))]
    fc_params = model.model.last_linear.parameters()
    to_optim  = [{'params': all_but_fc_params, 'lr':opt.lr, 'weight_decay':opt.decay},
                 {'params': fc_params, 'lr':opt.fc_lr, 'weight_decay':opt.decay}]

_  = model.to(opt.device)

"""============================================================================"""
dataset_config = utils.load_config('dataset/config.json')
transform_key = 'transform_parameters'
train_transform = dataset.utils.make_transform(
    **dataset_config[transform_key]
)

tr_dataset = dataset.load(
    name=opt.dataset,
    root=dataset_config['dataset'][opt.dataset]['root'],
    source=dataset_config['dataset'][opt.dataset]['source'],
    classes=dataset_config['dataset'][opt.dataset]['classes']['trainval'],
    transform=train_transform
)

batch_sampler = dataset.utils.BalancedBatchSampler(torch.Tensor(tr_dataset.ys), 8,
                                                   int(opt.bs / 8))

dl_tr = torch.utils.data.DataLoader(
    tr_dataset,
    batch_sampler=batch_sampler,
    num_workers=16,
)

dl_ev = torch.utils.data.DataLoader(
    dataset.load(
        name=opt.dataset,
        root=dataset_config['dataset'][opt.dataset]['root'],
        source=dataset_config['dataset'][opt.dataset]['source'],
        classes=dataset_config['dataset'][opt.dataset]['classes']['eval'],
        transform=dataset.utils.make_transform(
            **dataset_config[transform_key],
            is_train=False
        )
    ),
    batch_size=opt.bs,
    shuffle=False,
    num_workers=16,
    # pin_memory = True
)
opt.n_classes  = dl_tr.dataset.nb_classes()
print(opt.n_classes)
print(len(dl_tr.dataset))
print(len(dl_ev.dataset))

"""============================================================================"""
#################### CREATE LOGGING FILES ###############
sub_loggers = ['Train', 'Test', 'Model Grad']
if opt.use_tv_split: sub_loggers.append('Val')
LOG = logger.LOGGER(opt, sub_loggers=sub_loggers, start_new=True, log_online=False)

"""============================================================================"""
#################### LOSS SETUP ####################
batchminer   = bmine.select(opt.batch_mining, opt)
criterion, to_optim = criteria.select(opt.loss, opt, to_optim, batchminer)
_ = criterion.to(opt.device)

"""============================================================================"""
#################### OPTIM SETUP ####################
if opt.optim == 'adam':
    optimizer    = torch.optim.Adam(to_optim)
elif opt.optim == 'sgd':
    optimizer    = torch.optim.SGD(to_optim, momentum=0.9)
else:
    raise Exception('Optimizer <{}> not available!'.format(opt.optim))
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.tau, gamma=opt.gamma)

"""============================================================================"""
#################### METRIC COMPUTER ####################
opt.rho_spectrum_embed_dim = opt.embed_dim
print(opt)

"""============================================================================"""
################### SCRIPT MAIN ##########################
print('\n-----\n')

iter_count = 0
loss_args  = {'batch':None, 'labels':None, 'batch_features':None, 'f_embed':None}
best_recalls = [0, 0, 0, 0]

if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

if opt.resume:
    best_metrics, start_epoch = misc.load_checkpoint(model, optimizer, opt.resume)
    print(best_metrics)
    best_recalls = best_metrics
else:
    start_epoch = 0

for epoch in range(start_epoch, opt.n_epochs):
    epoch_start_time = time.time()

    opt.epoch = epoch
    ### Scheduling Changes specifically for cosine scheduling
    if opt.scheduler!='none':
        print('Running with learning rates {}...'.format(' | '.join('{}'.format(x) for x in scheduler.get_lr())))

    ### Train one epoch
    start = time.time()
    _ = model.train()

    loss_collect = []
    data_iterator = tqdm(dl_tr, desc='Epoch {} Training...'.format(epoch))

    print(opt.save_path)
    nmi, recall = eval.evaluate(model, dl_ev)
    for i, out in enumerate(data_iterator):
        input, class_labels, input_indices = out

        ### Compute Embedding
        input      = input.to(opt.device)
        model_args = {'x':input.to(opt.device)}
        # Needed for MixManifold settings.
        if 'mix' in opt.arch: model_args['labels'] = class_labels
        embeds  = model(**model_args)
        if isinstance(embeds, tuple): embeds, (avg_features, features) = embeds

        ### Compute Loss
        loss_args['batch']          = embeds
        loss_args['labels']         = class_labels
        # loss_args['f_embed']        = model.module.model.last_linear
        loss_args['batch_features'] = features
        loss = criterion(**loss_args)

        optimizer.zero_grad()

        if loss is not None:
            loss.backward()

            # torch.nn.utils.clip_grad_norm_(model.parameters(), 20)

            ### Compute Model Gradients and log them!
            grads = np.concatenate([p.grad.detach().cpu().numpy().flatten() for p in model.parameters() \
                                                 if p.grad is not None])
            grad_l2, grad_max  = np.mean(np.sqrt(np.mean(np.square(grads)))), \
                                 np.mean(np.max(np.abs(grads)))
            LOG.progress_saver['Model Grad'].log('Grad L2',  grad_l2,  group='L2')
            LOG.progress_saver['Model Grad'].log('Grad Max', grad_max, group='Max')

            ### Update network weights!
            optimizer.step()
            loss_collect.append(loss.item())

        ###
        iter_count += 1

        if i==len(dl_tr)-1:
            data_iterator.set_description('Epoch (Train) {0}: Mean Loss [{1:.4f}]'.format(epoch, np.mean(loss_collect)))

    result_metrics = {'loss': np.mean(loss_collect)}

    ####
    LOG.progress_saver['Train'].log('epochs', epoch)
    for metricname, metricval in result_metrics.items():
        LOG.progress_saver['Train'].log(metricname, metricval)
    LOG.progress_saver['Train'].log('time', np.round(time.time()-start, 4))

    """======================================="""
    ### Evaluate Metric for Training & Test (& Validation)
    _ = model.eval()
    print('\nComputing Testing Metrics...')
    nmi, recall = eval.evaluate(model, dl_ev)

    is_best = False
    if recall[1] > best_recalls[1]:
        best_recalls = recall
        is_best = True

    all_metrics = recall
    best_metrics = best_recalls

    # for k, v in all_metrics:
    #     LOG.progress_saver['Test'].log(k, v)

    print('saving checkpoint...')
    misc.save_checkpoint(model, optimizer, os.path.join(opt.save_path, 'latest.pth'),
                         all_metrics, best_metrics, epoch)
    if is_best:
        print('saving best checkpoint...')
        shutil.copy2(os.path.join(opt.save_path, 'latest.pth'), os.path.join(opt.save_path, 'best.pth'))

    print('###########')
    # print('Now rank-1 acc=%f, RP=%f, MAP@R=%f' % (overall_r1, overall_rp, overall_mapr))
    # print('Best rank-1 acc=%f, RP=%f, MAP@R=%f' % (best_r1,  best_rp, best_mapr))

    print('Now Recall@1,2,4,8 : {:.3f}, {:.3f}, {:.3f}, {:.3f}'.format(100*recall[1],100*recall[2],100*recall[4],100*recall[8] ))
    print('Best Recall@1,2,4,8 : {:.3f}, {:.3f}, {:.3f}, {:.3f}'.format(100*best_recalls[1],100*best_recalls[2],100*best_recalls[4],100*best_recalls[8] ))

    ### Learning Rate Scheduling Step
    if opt.scheduler != 'none':
        scheduler.step()

    print('Total Epoch Runtime: {0:4.2f}s'.format(time.time()-epoch_start_time))
    print('\n-----\n')
