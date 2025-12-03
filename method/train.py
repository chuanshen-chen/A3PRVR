import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
# import json
# import pprint
import random
import numpy as np
# import pickle
# from easydict import EasyDict as EDict
from tqdm import tqdm, trange
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
# import h5py
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
from method.config import BaseOptions
from method.model import MS_SL_Net
from method.data_provider import Dataset4MS_SL,  VisDataSet4MS_SL,\
    TxtDataSet4MS_SL, collate_train, read_video_ids

from method.eval import eval_epoch,start_inference
from method.optimization import BertAdam
from utils.basic_utils import AverageMeter, log_config
from utils.model_utils import count_parameters


import logging
logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

        
def setup_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    
def train_epoch(model, train_loader, optimizer, opt, epoch_i, training=True):
    logger.info("use train_epoch func for training: {}".format(training))
    model.train(mode=training)
    if opt.hard_negative_start_epoch != -1 and epoch_i >= opt.hard_negative_start_epoch:
        model.set_hard_negative(True, opt.hard_pool_size)

    # init meters
    dataloading_time = AverageMeter()
    prepare_inputs_time = AverageMeter()
    model_forward_time = AverageMeter()
    model_backward_time = AverageMeter()
    loss_meters = OrderedDict(neg_query_loss=AverageMeter(), neg_query_loss_action=AverageMeter(),
                              clip_nce_loss=AverageMeter(), clip_trip_loss=AverageMeter(), 
                              clip_nce_loss_action=AverageMeter(), clip_trip_loss_action=AverageMeter(),  
                              Inter_Diversity_loss=AverageMeter(), loss_overall=AverageMeter()
                              )


    num_training_examples = len(train_loader)

    timer_dataloading = time.time()
    for batch_idx, batch in tqdm(enumerate(train_loader), desc="Training Iteration", total=num_training_examples):
        global_step = epoch_i * num_training_examples + batch_idx
        dataloading_time.update(time.time() - timer_dataloading)
      
        timer_start = time.time()
        for k in batch.keys():
            if k != 'text_labels' and k != 'neg_text_labels' and k != 'text_labels_neg' and k != 'final_neg_inputs':
                batch[k] = batch[k].to(opt.device)

        # model_inputs = prepare_batch_inputs(batch[1], opt.device, non_blocking=opt.pin_memory)
        prepare_inputs_time.update(time.time() - timer_start)
       
        timer_start = time.time()
        feat_dict = model(batch=batch, 
                epoch_now=epoch_i)
        
        loss, loss_dict = model.get_all_loss(feat_dict)
        
        model_forward_time.update(time.time() - timer_start)
       
        timer_start = time.time()
        if training:
            optimizer.zero_grad()
            loss.backward()
            if opt.grad_clip != -1:
                nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
            optimizer.step()
            model_backward_time.update(time.time() - timer_start)


        for k, v in loss_dict.items():
            loss_meters[k].update(float(v))

        timer_dataloading = time.time()
        if opt.debug and batch_idx == 3:
            break

    if training:
        to_write = opt.train_log_txt_formatter.format(time_str=time.strftime("%Y_%m_%d_%H_%M_%S"), epoch=epoch_i,
                                                      loss_str=" ".join(["{} {:.4f}".format(k, v.avg)
                                                                         for k, v in loss_meters.items()]))
        with open(opt.train_log_filepath, "a") as f:
            f.write(to_write)
        print("Epoch time stats:")
        print("dataloading_time: max {dataloading_time.max} min {dataloading_time.min} avg {dataloading_time.avg}\n"
              "prepare_inputs_time: max {prepare_inputs_time.max} "
              "min {prepare_inputs_time.min} avg {prepare_inputs_time.avg}\n"
              "model_forward_time: max {model_forward_time.max} "
              "min {model_forward_time.min} avg {model_forward_time.avg}\n"
              "model_backward_time: max {model_backward_time.max} "
              "min {model_backward_time.min} avg {model_backward_time.avg}\n".format(
            dataloading_time=dataloading_time, prepare_inputs_time=prepare_inputs_time,
            model_forward_time=model_forward_time, model_backward_time=model_backward_time))
    else:
        a = 1
        # for k, v in loss_meters.items():
        #     opt.writer.add_scalar("Eval_Loss/{}".format(k), v.avg, epoch_i)


def rm_key_from_odict(odict_obj, rm_suffix):
    """remove key entry from the OrderedDict"""
    return OrderedDict([(k, v) for k, v in odict_obj.items() if rm_suffix not in k])


def train(model, train_dataset, val_video_dataset, val_text_dataset, opt):
    # Prepare optimizer
    if opt.device.type == "cuda":
        logger.info("CUDA enabled.")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
            logger.info("Use multi GPU", opt.device_ids)
        model.to(device)

    train_loader = DataLoader(dataset=train_dataset,batch_size=opt.bsz,shuffle=True,pin_memory=opt.pin_memory,
                                num_workers=opt.num_workers,collate_fn=collate_train)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
        {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0}]

    num_train_optimization_steps = len(train_loader) * opt.n_epoch
    optimizer = BertAdam(optimizer_grouped_parameters, lr=opt.lr, weight_decay=opt.wd, warmup=opt.lr_warmup_proportion,
                         t_total=num_train_optimization_steps, schedule="warmup_linear")
    prev_best_score = 0.
    es_cnt = 0
    start_epoch = -1 if opt.eval_untrained else 0
    torch.cuda.empty_cache()
    for epoch_i in trange(start_epoch, opt.n_epoch, desc="Epoch"):
        # debug_name = 'iccv_250513_changebig_test'
        if epoch_i > -1:
            # with torch.autograd.detect_anomaly(): 
            if not opt.only_eval: 
                train_epoch(model, train_loader, optimizer, opt, epoch_i, training=True)

        
        with torch.no_grad():
            if opt.only_eval:
                checkpoint = opt.eval_ckpt
                state = torch.load(checkpoint)
                model.load_state_dict(state['model'], strict=False)
           
            rsum = eval_epoch(model, val_video_dataset, val_text_dataset, opt,epoch = epoch_i)

        stop_score = rsum
        if stop_score >= prev_best_score:
            es_cnt = 0
            prev_best_score = stop_score
            checkpoint = {"model": model.state_dict() if torch.cuda.device_count() <= 1 else model.module.state_dict(), "epoch": epoch_i}
            torch.save(checkpoint, opt.ckpt_filepath)

            logger.info("The checkpoint file has been updated.")
        else:
            es_cnt += 1
            if opt.max_es_cnt != -1 and es_cnt > opt.max_es_cnt and 'debug' not in opt.description:  # early stop
                with open(opt.train_log_filepath, "a") as f:
                    f.write("Early Stop at epoch {}".format(epoch_i))
                break
        if opt.debug:
            break

    # opt.writer.close()


def start_training(opt):
   

    # opt.writer = SummaryWriter(opt.tensorboard_log_dir)
    opt.train_log_txt_formatter = "{time_str} [Epoch] {epoch:03d} [Loss] {loss_str}\n"
    opt.eval_log_txt_formatter = "{time_str} [Epoch] {epoch:03d} [Metrics] {eval_metrics_str}\n"

    rootpath = opt.root_path

    dataset_files = opt.dataset_name+"_i3d"

    text_feat_path = opt.text_feat_path
    
    train_dataset = Dataset4MS_SL(opt.caption_train_txt, text_feat_path, opt)
 
    val_text_dataset = TxtDataSet4MS_SL(opt.caption_test_txt, text_feat_path, opt)

    val_video_ids_list = read_video_ids(opt.caption_test_txt)
    
    val_video_dataset = VisDataSet4MS_SL(opt, video_ids=val_video_ids_list)

    model_config = opt

    logger.info("model_config {}".format(model_config))

    NAME_TO_MODELS = {'MS_SL_Net':MS_SL_Net}
    model = NAME_TO_MODELS[opt.model_name](model_config)
    count_parameters(model)
    
    logger.info("Start Training...")
    train(model, train_dataset, val_video_dataset, val_text_dataset, opt)
    return opt.results_dir, opt.eval_split_name, opt.eval_path, opt.debug, opt.model_name
    
if __name__ == '__main__':
    
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    logger.info("Setup config, data and model...")
    opt = BaseOptions().parse()
    setup_seed(opt.seed)
    log_config(opt.results_dir, 'performance')
    model_dir, eval_split_name, eval_path, debug, model_name = start_training(opt)
  