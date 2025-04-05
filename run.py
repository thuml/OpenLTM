import os
import argparse
import random
import numpy as np
import torch
import torch.distributed as dist
from exp.exp_forecast import Exp_Forecast

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Timer-XL')

    # basic config
    parser.add_argument('--task_name', type=str, required=True, default='forecast', help='task name, options:[forecast]')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='timer_xl', help='model name, options: [timer_xl, timer, moirai, moment]')
    parser.add_argument('--seed', type=int, default=2021, help='seed')
    
    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETTh1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./dataset/ETT-small/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--test_flag', type=str, default='T', help='test domain')
    parser.add_argument('--load_time_stamp', type=str, default='none', help='type of load time stamp')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=672, help='input sequence length')
    parser.add_argument('--input_token_len', type=int, default=576, help='input token length')
    parser.add_argument('--output_token_len', type=int, default=96, help='output token length')
    parser.add_argument('--test_seq_len', type=int, default=672, help='test seq len')
    parser.add_argument('--test_pred_len', type=int, default=96, help='test pred len')

    # model define
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--e_layers', type=int, default=1, help='encoder layers')
    parser.add_argument('--d_model', type=int, default=512, help='d model')
    parser.add_argument('--n_heads', type=int, default=8, help='n heads')
    parser.add_argument('--d_ff', type=int, default=2048, help='d ff')
    parser.add_argument('--activation', type=str, default='relu', help='activation')
    parser.add_argument('--covariate', action='store_true', help='use cov', default=False)
    parser.add_argument('--node_num', type=int, default=100, help='number of nodes')
    parser.add_argument('--node_list', type=str, default='23,37,40', help='number of nodes for a tree')
    parser.add_argument('--use_norm', action='store_true', help='use norm', default=False)
    parser.add_argument('--nonautoregressive', action='store_true', help='nonautoregressive', default=False)
    parser.add_argument('--test_dir', type=str, default='./test', help='test dir')
    parser.add_argument('--test_file_name', type=str, default='checkpoint.pth', help='test file')
    parser.add_argument('--output_attention', action='store_true', help='output attention', default=False)
    parser.add_argument('--visualize', action='store_true', help='visualize', default=False)
    parser.add_argument('--flash_attention', action='store_true', help='flash attention', default=False)

    # adaptation
    parser.add_argument('--adaptation', action='store_true', help='adaptation', default=False)
    parser.add_argument('--pretrain_model_path', type=str, default='pretrain_model.pth', help='pretrain model path')
    parser.add_argument('--subset_rand_ratio', type=float, default=1, help='few shot ratio')
    
    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--cosine', action='store_true', help='use cosine annealing lr', default=False)
    parser.add_argument('--tmax', type=int, default=10, help='tmax in cosine anealing lr')
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--valid_last', action='store_true', help='valid last', default=False)
    parser.add_argument('--last_token', action='store_true', help='last token', default=False)
    
    # GPU
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--ddp', action='store_true', help='Distributed Data Parallel', default=False)
    parser.add_argument('--dp', action='store_true', help='Data Parallel', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')    
    
    # LLM-based model
    parser.add_argument('--gpt_layers', type=int, default=6)
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--kernel_size', type=int, default=25)
    parser.add_argument('--stride', type=int, default=8)
    
    # TTM
    parser.add_argument("--n_vars", type=int, default=7, help='number of variables')
    parser.add_argument("--factor", type=int, default=2, help='expansion factor of hidden layer')
    parser.add_argument("--mode", type=str, default="mix_channel", help="allowed values: common_channel, mix_channel")
    parser.add_argument("--AP_levels", type=int, default=0, help="number of attention patching levels")
    parser.add_argument("--use_decoder", action="store_true", help="use decoder", default=True)
    parser.add_argument("--d_mode", type=str, default="common_channel", help="allowed values: common_channel, mix_channel")
    parser.add_argument("--d_layers", type=int, default=8, help="number of layers in decoder")
    parser.add_argument("--d_d_model", type=int, default=16, help="d_model in decoder")
    
    # Time-LLM
    parser.add_argument("--ts_vocab_size", type=int, default=1000, help="size of a small collection of text prototypes in llm")
    parser.add_argument("--domain_des", type=str, default="The Electricity Transformer Temperature (ETT) is a crucial indicator in the electric power long-term deployment.", help="domain description")
    # @llm_model: 
    # for Time-LLM: Currently, LLAMA, GPT2, and BERT are supported.
    # for AutoTimes: Currently, LLAMA, GPT2, and OPT are supported.
    parser.add_argument("--llm_model", type=str, default="LLAMA", help="LLM model, LLAMA, GPT2, BERT, OPT are supported") 
    parser.add_argument("--llm_layers", type=int, default=6, help="number of layers in llm")
    
    # AutoTimes
    parser.add_argument("--mix_embeds", action="store_true", help="load time stamp embedding from file, mix time series and textual embeddings of time stamp", default=False)
    
    args = parser.parse_args()
    fix_seed = args.seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    args.node_list = [int(x) for x in args.node_list.split(',')]
    
    if args.dp:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]
    elif args.ddp:
        ip = os.environ.get("MASTER_ADDR", "127.0.0.1")
        port = os.environ.get("MASTER_PORT", "64209")
        hosts = int(os.environ.get("WORLD_SIZE", "8"))
        rank = int(os.environ.get("RANK", "0")) 
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        gpus = torch.cuda.device_count()
        args.local_rank = local_rank
        print(ip, port, hosts, rank, local_rank, gpus)
        dist.init_process_group(backend="nccl", init_method=f"tcp://{ip}:{port}", world_size=hosts,
                                rank=rank)
        torch.cuda.set_device(local_rank)
    
    if args.task_name == 'forecast':
        Exp = Exp_Forecast
    else:
        Exp = Exp_Forecast

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            exp = Exp(args)  # set experiments
            setting = '{}_{}_{}_{}_sl{}_it{}_ot{}_lr{}_bt{}_wd{}_el{}_dm{}_dff{}_nh{}_cos{}_{}_{}'.format(
                args.task_name,
                args.model_id,
                args.model,
                args.data,
                args.seq_len,
                args.input_token_len,
                args.output_token_len,
                args.learning_rate,
                args.batch_size,
                args.weight_decay,
                args.e_layers,
                args.d_model,
                args.d_ff,
                args.n_heads,
                args.cosine,
                args.des, ii)
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            if not args.ddp and not args.dp:
                exp.test(setting)
            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_{}_sl{}_it{}_ot{}_lr{}_bt{}_wd{}_el{}_dm{}_dff{}_nh{}_cos{}_{}_{}'.format(
            args.task_name,
            args.model_id,
            args.model,
            args.data,
            args.seq_len,
            args.input_token_len,
            args.output_token_len,
            args.learning_rate,
            args.batch_size,
            args.weight_decay,
            args.e_layers,
            args.d_model,
            args.d_ff,
            args.n_heads,
            args.cosine,
            args.des, ii)
        exp = Exp(args)  # set experiments
        exp.test(setting, test=1)
        torch.cuda.empty_cache()
