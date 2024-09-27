import os
import random
import time
import torch
import torch.backends.cudnn as cudnn
import models
from utils_ssl.logger import Logger
import myexman
from utils_ssl import utils
import sys
import torch.multiprocessing as mp
import torch.distributed as dist
import socket
import sys
import os
import torch.distributed as dist
sys.path.append(os.getcwd())
import requests
import gc
import sys
from datetime import datetime

def add_learner_params(parser):
    parser.add_argument('--problem', default='sim-clr',
        help='The problem to train',
        choices=models.REGISTERED_MODELS,
    )
    parser.add_argument('--name', default='',
        help='Name for the experiment',
    )
    parser.add_argument('--ckpt', default='',
        help='Optional checkpoint to init the model.'
    )
    parser.add_argument('--mobius-prob', default=0.8, type=float, help='MPD Mobius tranformation probability value')
    parser.add_argument('--mobius', default=True, type=bool, help='Make it true to enable Mobius transformation')
    parser.add_argument('--mobius_background', default=False, type=bool, help='Mobius transformation with background interpolation')
    parser.add_argument('--verbose', default=False, type=bool)
    parser.add_argument('--val_portion', default='val', type=str, help='val portion')
    # optimizer params
    parser.add_argument('--lr_schedule', default='warmup-anneal')
    parser.add_argument('--opt', default='lars', help='Optimizer to use', choices=['sgd', 'adam', 'lars'])
    parser.add_argument('--iters', default=-1, type=int, help='The number of optimizer updates')
    parser.add_argument('--warmup', default=0, type=float, help='The number of warmup iterations in proportion to \'iters\'')
    parser.add_argument('--lr', default=0.1, type=float, help='Base learning rate')
    parser.add_argument('--wd', '--weight_decay', default=1e-4, type=float, dest='weight_decay')
    # trainer params
    parser.add_argument('--save_freq', default=10000000000000000, type=int, help='Frequency to save the model')
    parser.add_argument('--log_freq', default=100, type=int, help='Logging frequency')
    parser.add_argument('--eval_freq', default=10000000000000000, type=int, help='Evaluation frequency')
    parser.add_argument('-j', '--workers', default=4, type=int, help='The number of data loader workers')
    parser.add_argument('--eval_only', default=False, type=bool, help='Skips the training step if True')
    parser.add_argument('--seed', default=-1, type=int, help='Random seed')
    # parallelizm params:
    parser.add_argument('--dist', default='dp', type=str,
        help='dp: DataParallel, ddp: DistributedDataParallel',
        choices=['dp', 'ddp'],
    )
    parser.add_argument('--dist_address', default='127.0.0.1:1234', type=str,
        help='the address and a port of the main node in the <address>:<port> format'
    )
    parser.add_argument('--node_rank', default=0, type=int,
        help='Rank of the node (script launched): 0 for the main node and 1,... for the others',
    )
    parser.add_argument('--world_size', default=1, type=int,
        help='the number of nodes (scripts launched)',
    )


def main():
    parser = myexman.ExParser(file=os.path.basename(__file__))
    add_learner_params(parser)

    is_help = False
    if '--help' in sys.argv or '-h' in sys.argv:
        sys.argv.pop(sys.argv.index('--help' if '--help' in sys.argv else '-h'))
        is_help = True

    args, _ = parser.parse_known_args(log_params=False)
    
    #try:

    models.REGISTERED_MODELS[args.problem].add_model_hparams(parser)

    if is_help:
        sys.argv.append('--help')

    args = parser.parse_args(namespace=args)

    if args.data == 'imagenet' and args.aug == False:
        raise Exception('ImageNet models should be eval with aug=True!')

    if args.seed != -1:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    args.gpu = 0
    ngpus = torch.cuda.device_count()
    args.number_of_processes = 1
    if args.dist == 'ddp':
        args.number_of_processes = args.world_size * ngpus
        parser.update_params_file(args)
        args.world_size *= ngpus
        print('args.world_size', args.world_size)
        mp.spawn(
            main_worker,
            nprocs=ngpus,
            args=(ngpus, args),
        )
    else:
        print('args.world_size', args.world_size)
        parser.update_params_file(args)
        main_worker(args.gpu, -1, args)
            
    
def main_worker(gpu, ngpus, args):
    
   
    fmt = {
        'train_time': '.3f',
        'val_time': '.3f',
        'lr': '.1e',
    }
    logger = Logger('logs', base=args.root, fmt=fmt)

    args.gpu = gpu
    torch.cuda.set_device(gpu)
    args.rank = args.node_rank * ngpus + gpu

    device = torch.device('cuda:%d' % args.gpu)
    if args.dist == 'ddp':
        dist.init_process_group(
            backend='nccl',
            init_method='tcp://%s' % args.dist_address,
            world_size=args.world_size,
            rank=args.rank,
        )

        n_gpus_total = dist.get_world_size()
        assert args.batch_size % n_gpus_total == 0
        args.batch_size //= n_gpus_total
        if args.rank == 0:
            print(f'===> {n_gpus_total} GPUs total; batch_size={args.batch_size} per GPU')
            print(f'Model {args.name}, Learning Rate {args.lr}, Mobius {args.mobius} for interpolated background {args.mobius_background} with probability {args.mobius_prob}')

        print(f'===> Proc {dist.get_rank()}/{dist.get_world_size()}@{socket.gethostname()}', flush=True)
    if args.dist =='dp':
        dist.init_process_group(
            backend='nccl',
            init_method='tcp://localhost:2345',
            world_size=args.world_size,
            rank=args.rank,
        )

        n_gpus_total = dist.get_world_size()
        assert args.batch_size % n_gpus_total == 0
        args.batch_size //= n_gpus_total
        if args.rank == 0:
            print(f'===> {n_gpus_total} GPUs total; batch_size={args.batch_size} per GPU')
            print(f'Model {args.name}, Learning Rate {args.lr}, Mobius {args.mobius} for interpolated background {args.mobius_background} with probability {args.mobius_prob}')

        print(f'===> Proc {dist.get_rank()}/{dist.get_world_size()}@{socket.gethostname()}', flush=True)

    # create model
    model = models.REGISTERED_MODELS[args.problem](args, device=device)
    cur_iter = 0
    if args.ckpt != '':
        ckpt = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(ckpt['state_dict'])
        print(f"model from checkpoint loaded for iteration {ckpt['iter']}")
        cur_iter = ckpt['iter']
    else:
        cur_iter = 0
        print('No model state to load')
        
    
    # Data loading code
    model.prepare_data()
    train_loader, val_loader = model.dataloaders(iters=args.iters)

    # define optimizer
        
    if args.ckpt != '' and not args.eval_only:
        print('args.ckpt != ', args.ckpt != '', not args.eval_only)
        
        # optionally resume from a checkpoint
        optimizer, scheduler = models.ssl.configure_optimizers(args, model, cur_iter - 1, LARS_optimizer_weights = ckpt['opt_state_dict'])    
        print(f"optimizer from checkpoint loaded for iteration {ckpt['iter']}")
    else:
        cur_iter = 1
        optimizer, scheduler = models.ssl.configure_optimizers(args, model, cur_iter - 1, LARS_optimizer_weights = None)
        print('No optimzer state to load')
        
    cudnn.benchmark = True

    continue_training = args.iters != 0
    data_time, it_time = 0, 0
    while continue_training:
        train_logs = []
        model.train()
        start_time = time.time()
        for _, batch in enumerate(train_loader):
            torch.cuda.empty_cache()
            cur_iter += 1

            batch = [x.to(device) for x in batch]
            data_time += time.time() - start_time

            logs = {}
            if not args.eval_only:
                # forward pass and compute loss
                logs = model.train_step(batch, cur_iter)
                loss = logs['loss']
                # gradient step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # save logs for the batch
            train_logs.append({k: utils.tonp(v) for k, v in logs.items()})

            
            if cur_iter % args.log_freq == 0 and args.rank == 0:
                save_checkpoint(args.root, model, optimizer, cur_iter)
            
            it_time += time.time() - start_time

            if (cur_iter % args.log_freq == 0 or cur_iter >= args.iters) and args.rank == 0:
                save_checkpoint(args.root, model, optimizer)
                train_logs = utils.agg_all_metrics(train_logs)

                logger.add_logs(cur_iter, train_logs, pref='train_')
                logger.add_scalar(cur_iter, 'lr', optimizer.param_groups[0]['lr'])
                logger.add_scalar(cur_iter, 'data_time', data_time)
                logger.add_scalar(cur_iter, 'it_time', it_time)
                logger.iter_info()
                logger.save()

                data_time, it_time = 0, 0
                train_logs = []

            if scheduler is not None:
                scheduler.step()

            if cur_iter >= args.iters:
                continue_training = False
                break

            start_time = time.time()

        torch.cuda.empty_cache()
        gc.collect()
    save_checkpoint(args.root, model, optimizer)

    if args.dist == 'ddp':
        dist.destroy_process_group()


def save_checkpoint(path, model, optimizer, cur_iter=None):
    if cur_iter is None:
        fname = os.path.join(path, 'checkpoint.pth.tar')
    else:
        fname = os.path.join(path, 'checkpoint.pth.tar')

    ckpt = model.get_ckpt()
    ckpt.update(
        {
            'opt_state_dict': optimizer.state_dict(),
            'iter': cur_iter,
        }
    )

    torch.save(ckpt, fname)
    

if __name__ == '__main__':
    
    main()
    
        
