import argparse
import logging
import os
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.runner import Runner
from opencd.registry import RUNNERS

import opencd_custom  # noqa: F401,F403


def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor') # 实例化 添加描述
    parser.add_argument('config', help='train config file path') # 配置文件路径
    parser.add_argument('--work-dir', help='the dir to save logs and models') # 模型和日志保存目录
    parser.add_argument(
        '--resume',
        action='store_true',
        default=False,
        help='resume from the latest checkpoint in the work_dir automatically') # 布尔标志，如果被设置，则表示从最近的检查点自动恢复训练。
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training') # 布尔标志，如果被设置，则表示启用自动混合精度训练
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.') # 用于覆盖配置文件中的一些设置。它接受一个或多个键值对作为参数，这些键值对会被合并到配置文件中。
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher') # 用于指定作业启动器的选择，可以是none、pytorch、slurm或mpi之一。
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    '''
    这几行注释解释了在使用PyTorch版本大于等于2.0.0时的一个行为变化。
    在以前的版本中 使用torch.distributed.launch启动分布式训练时 
    它会将--local_rank参数传递给tools/train.py脚本
    而在PyTorch版本大于等于2.0.0时，它会将--local-rank参数传递给相同的脚本。
    '''
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0) # 它用于指定本地排名，用于分布式训练。
    args = parser.parse_args() # 解析命令行参数并将其存储在args对象中
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank) # 如果环境变量中没有定义LOCAL_RANK，则将args.local_rank的值转换为字符串，并将其设置为环境变量LOCAL_RANK的值。

    return args


def main():
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    # enable automatic-mixed-precision training
    if args.amp is True:
        optim_wrapper = cfg.optim_wrapper.type
        if optim_wrapper == 'AmpOptimWrapper':
            print_log(
                'AMP training is already enabled in your config.',
                logger='current',
                level=logging.WARNING)
        else:
            assert optim_wrapper == 'OptimWrapper', (
                '`--amp` is only supported when the optimizer wrapper type is '
                f'`OptimWrapper` but got {optim_wrapper}.')
            cfg.optim_wrapper.type = 'AmpOptimWrapper'
            cfg.optim_wrapper.loss_scale = 'dynamic'

    # resume training
    cfg.resume = args.resume

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # start training
    runner.train()


if __name__ == '__main__':
    main()
