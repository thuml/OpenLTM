from data_provider.data_loader import UnivariateDatasetBenchmark, MultivariateDatasetBenchmark, Global_Temp, Global_Wind, Dataset_ERA5_Pretrain, Dataset_ERA5_Pretrain_Test, UTSD, UTSD_Npy
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

data_dict = {
    'UnivariateDatasetBenchmark': UnivariateDatasetBenchmark,
    'MultivariateDatasetBenchmark': MultivariateDatasetBenchmark,
    'Global_Temp': Global_Temp,
    'Global_Wind': Global_Wind,
    'Era5_Pretrain': Dataset_ERA5_Pretrain,
    'Era5_Pretrain_Test': Dataset_ERA5_Pretrain_Test,
    'Utsd': UTSD,
    'Utsd_Npy': UTSD_Npy
}


def data_provider(args, flag):
    Data = data_dict[args.data]

    if flag in ['test', 'val']:
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
    else:
        shuffle_flag = True
        drop_last = False
        batch_size = args.batch_size

    if flag in ['train', 'val']:
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.input_token_len, args.output_token_len],
            nonautoregressive=args.nonautoregressive,
            test_flag=args.test_flag,
            subset_rand_ratio=args.subset_rand_ratio,
            load_time_stamp=args.load_time_stamp,
        )
    else:
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.test_seq_len, args.input_token_len, args.test_pred_len],
            nonautoregressive=args.nonautoregressive,
            test_flag=args.test_flag,
            subset_rand_ratio=args.subset_rand_ratio,
            load_time_stamp=args.load_time_stamp,
        )
    print(flag, len(data_set))
    if args.ddp:
        train_datasampler = DistributedSampler(data_set, shuffle=shuffle_flag)
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            sampler=train_datasampler,
            num_workers=args.num_workers,
            persistent_workers=True,
            pin_memory=True,
            drop_last=drop_last,
        )
    else:
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            persistent_workers=True,
            pin_memory=True,
            drop_last=drop_last
        )
    return data_set, data_loader
