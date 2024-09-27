import os
#os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb=10'

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
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
    parser.add_argument('--val_portion', default='', type = str, help='Name for the experiment')
    
    parser.add_argument('--ckpt', default='',
        help='Optional checkpoint to init the model.'
    )
    parser.add_argument('--mobius-prob', default=0.0, type=float, help='Mobius tranformation probability value')
    parser.add_argument('--mobius', default=True, type=bool, help='Make it true to enable Mobius transformation')
    parser.add_argument('--mobius_background', default=False, type=bool, help='Mobius transformation with background interpolation')
    parser.add_argument('--verbose', default=False, type=bool)
    # optimizer params
    parser.add_argument('--lr_schedule', default='warmup-anneal')
    parser.add_argument('--opt', default='lars', help='Optimizer to use', choices=['sgd', 'adam', 'lars'])
    parser.add_argument('--iters', default=-1, type=int, help='The number of optimizer updates')
    parser.add_argument('--warmup', default=0, type=float, help='The number of warmup iterations in proportion to \'iters\'')
    parser.add_argument('--lr', default=0.1, type=float, help='Base learning rate')
    parser.add_argument('--wd', '--weight_decay', default=1e-4, type=float, dest='weight_decay')
    # trainer params
    parser.add_argument('--save_freq', default=10000000000000000, type=int, help='Frequency to save the model')
    parser.add_argument('--log_freq', default=1, type=int, help='Logging frequency')
    parser.add_argument('--eval_freq', default=1, type=int, help='Evaluation frequency')
    parser.add_argument('-j', '--workers', default=4, type=int, help='The number of data loader workers')
    parser.add_argument('--eval_only', default=False, type=bool, help='Skips the training step if True')
    parser.add_argument('--seed', default=-1, type=int, help='Random seed')
    # parallelizm params:
    parser.add_argument('--dist', default='dp', type=str,
        help='dp: DataParallel, ddp: DistributedDataParallel',
        choices=['dp', 'ddp'],
    )
    parser.add_argument('--dist_address', default='127.0.0.1:12384', type=str,
        help='the address and a port of the main node in the <address>:<port> format'
    )
    parser.add_argument('--node_rank', default=0, type=int,
        help='Rank of the node (script launched): 0 for the main node and 1,... for the others',
    )
    parser.add_argument('--world_size', default=1, type=int,
        help='the number of nodes (scripts launched)',
    )


#import os
#os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
#torch.backends.cudnn.enabled = False

#os.environ['CUDA_LAUNCH_BLOCKING'] = "0"

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
    #print('ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ')
    if args.dist == 'ddp':
        # add additional argument to be able to retrieve # of processes from logs
        # and don't change initial arguments to reproduce the experiment
        args.number_of_processes = args.world_size * ngpus
        parser.update_params_file(args)
        #print('ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ')
        args.world_size *= ngpus
        print('args.world_size', args.world_size)
        mp.spawn(
            main_worker,
            nprocs=ngpus,
            args=(ngpus, args),
        )
    else:
        #args.number_of_processes = args.world_size * ngpus
        #args.world_size *= ngpus
        print('args.world_size', args.world_size)
        parser.update_params_file(args)
        main_worker(args.gpu, -1, args)
            
    '''except Exception as e:
        print(e)
        if 0 == dist.get_rank():
            send_telegram(experiment= args.name, message=' Hello Prakash! SSL SimcLR with Mobius Model Training Crashed.')'''

'''from gpu_profile import trace_calls
os.environ['GPU_DEBUG']='2'
os.environ['TRACE_INTO'] = 'forward'
sys.settrace(trace_calls)'''

import os

def append_record_to_file(file_path, record):
    # Convert the dictionary to a string representation
    record_str = str(record)

    # Check if the file exists
    if not os.path.exists(file_path):
        # Create the file and write the first record
        with open(file_path, 'w') as file:
            file.write(record_str + '\n')
    else:
        # Append the new record to the existing file
        with open(file_path, 'a') as file:
            file.write(record_str + '\n')


def main_worker(gpu, ngpus, args):
    
    #sys.settrace(gpu_profile)
    
    fmt = {
        'train_time': '.3f',
        'val_time': '.3f',
        'lr': '.1e',
    }
    logger = Logger('logs_mb512_nb_srlr', base=args.root, fmt=fmt)

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
    print('cur_iter ', cur_iter)
    if args.ckpt != '':
        ckpt = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(ckpt['state_dict'])
        print(f"model from checkpoint loaded for iteration {ckpt['iter']}")
        cur_iter = ckpt['iter']
        print('cur_iter args.ckpt', cur_iter)
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
        # otherwise simply create instances
        cur_iter = 0
        print('cur_iter ', cur_iter)
        optimizer, scheduler = models.ssl.configure_optimizers(args, model, cur_iter - 1, LARS_optimizer_weights = None)
        print('No optimzer state to load')
        
    cudnn.benchmark = True

    print('args.iters ', args.iters)
    continue_training = args.iters != 0
    data_time, it_time = 0, 0
    if 0 == dist.get_rank():
        if args.ckpt != '':
            send_telegram(experiment= args.name, message=' Hello Prakash! Mobius Model Training Resumed.')
        else:
            send_telegram(experiment= args.name, message=' Hello Prakash! Mobius Model Training Started.')
            
            
    test_logs = []
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            batch = [x.to(device) for x in batch]
            # forward pass
            
            logs = model.test_step(batch)
            # save logs for the batch
            test_logs.append(logs)
    test_logs = utils.agg_all_metrics(test_logs)
    test_logs['name'] = args.name
    print(test_logs)
    append_record_to_file('/home/prachh/adversarial_pretraining/perspective/classification/ssl_eval_results/ssl_eval.txt', test_logs)
    logger.add_logs(cur_iter, test_logs, pref='test_')
   

    



def send_telegram(chat_id=-4047549936, experiment= None, message=" Hello Prakash! SimCLR SSL Model Training Completed."):
    TOKEN = "5819045340:AAE0i1bkfFtSYNIix1nkNNRjR7ECttxAGYQ"
    base_url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"

    #base_url = f"https://api.telegram.org/bot{5819045340:AAE0i1bkfFtSYNIix1nkNNRjR7ECttxAGYQ}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": f'{experiment}: {message}'
    }
    response = requests.post(base_url, data=payload)
    return response.json()

if __name__ == '__main__':
    
    main()
    
        
