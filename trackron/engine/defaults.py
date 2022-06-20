import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import torch
from trackron.utils import comm
from trackron.utils.collect_env import collect_env_info
from trackron.utils.env import seed_all_rng
from trackron.utils.events import (CommonMetricPrinter, JSONWriter,
                                   TensorboardXWriter)
from trackron.utils.logger import setup_logger

__all__ = [
    "default_argument_parser",
    "default_setup",
    "default_writers"
]


def default_argument_parser(epilog=None):
    """
    Create a parser with some common arguments used by trackron users.

    Args:
        epilog (str): epilog passed to ArgumentParser describing the usage.

    Returns:
        argparse.ArgumentParser:
    """
    parser = argparse.ArgumentParser(
        epilog=epilog or f"""
Examples:

Run on single machine:
    $ {sys.argv[0]} --num-gpus 8 --config-file cfg.yaml

Change some config options:
    $ {sys.argv[0]} --config-file cfg.yaml MODEL.WEIGHTS /path/to/weight.pth SOLVER.BASE_LR 0.001

Run on multiple machines:
    (machine0)$ {sys.argv[0]} --machine-rank 0 --num-machines 2 --dist-url <URL> [--other-flags]
    (machine1)$ {sys.argv[0]} --machine-rank 1 --num-machines 2 --dist-url <URL> [--other-flags]
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config-file",
                        default="",
                        metavar="FILE",
                        help="path to config file")
    parser.add_argument("--config-func",
                        default="s3t",
                        help="function for setting configs")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Whether to attempt to resume from the checkpoint directory. "
        "See documentation of `DefaultTrainer.resume_or_load()` for what it means.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="sot",
        help="tracking mode for training and inference"
    )
    parser.add_argument("--eval-only",
                        action="store_true",
                        help="perform evaluation only")
    parser.add_argument("--num-gpus",
                        type=int,
                        default=1,
                        help="number of gpus *per machine*")
    parser.add_argument("--debug",
                        type=int,
                        default=0,
                        help="debug level")
    parser.add_argument("--num-machines",
                        type=int,
                        default=1,
                        help="total number of machines")
    parser.add_argument("--unused-parameter",
                        type=bool,
                        default=True,
                        help="model contain unused parameters")
    parser.add_argument("--machine-rank",
                        type=int,
                        default=0,
                        help="the rank of this machine (unique per machine)")

    # PyTorch still may leave orphan processes in multi-gpu training.
    # Therefore we use a deterministic way to obtain port,
    # so that users are aware of orphan processes by seeing the port occupied.
    port = 2 ** 15 + 2 ** 14 + \
        hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    parser.add_argument(
        "--dist-url",
        default="tcp://127.0.0.1:{}".format(port),
        help="initialization URL for pytorch distributed backend. See "
        "https://pytorch.org/docs/stable/distributed.html for details.",
    )
    parser.add_argument(
        "opts",
        help="""
Modify config options at the end of the command. For Yacs configs, use
space-separated "PATH.KEY VALUE" pairs.
For python-based LazyConfig, use "path.key=value".
        """.strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser

def default_writers(output_dir: str, max_iter: Optional[int] = None):
    """
    Build a list of :class:`EventWriter` to be used.
    It now consists of a :class:`CommonMetricPrinter`,
    :class:`TensorboardXWriter` and :class:`JSONWriter`.

    Args:
        output_dir: directory to store JSON metrics and tensorboard events
        max_iter: the total number of iterations

    Returns:
        list[EventWriter]: a list of :class:`EventWriter` objects.
    """
    return [
        # It may not always print what you want to see, since it prints "common" metrics only.
        CommonMetricPrinter(max_iter),
        JSONWriter(os.path.join(output_dir, "metrics.json")),
        TensorboardXWriter(output_dir),
    ]

def default_setup(cfg, args):
    """
    Perform some basic common setups at the beginning of a job, including:

    1. Set up the trackron logger
    2. Log basic information about environment, cmdline arguments, and config
    3. Backup the config to the output directory

    Args:
        cfg (CfgNode): the full config to be used
        args (argparse.NameSpace): the command line arguments to be logged
    """
    output_dir = Path(cfg.OUTPUT_DIR)
    if comm.is_main_process() and output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    rank = comm.get_rank()
    setup_logger(output_dir, distributed_rank=rank, name="fvcore")
    logger = setup_logger(output_dir, distributed_rank=rank)

    logger.info("Rank of current process: {}. World size: {}".format(
        rank, comm.get_world_size()))
    logger.info("Environment info:\n" + collect_env_info())

    logger.info("Command line arguments: " + str(args))
    if hasattr(args, "config_file") and args.config_file != "":
        logger.info("Contents of args.config_file={}:\n{}".format(
            args.config_file,
            open(args.config_file, "r").read()))

    if comm.is_main_process() and output_dir:
        # Note: some of our scripts may expect the existence of
        # config.yaml in output directory
        logger.info("Running with full config:\n{}".format(cfg))
        path = output_dir / "config.yaml"
        if not args.eval_only:
            with open(path, "w") as f:
                f.write(cfg.dump())
            logger.info("Full config saved to {}".format(path))

    # make sure each worker has a different, yet deterministic seed if specified
    seed_all_rng(None if cfg.SEED < 0 else cfg.SEED + rank)

    # cudnn benchmark has large overhead. It shouldn't be used considering the small size of
    # typical validation set.
    if not (hasattr(args, "eval_only") and args.eval_only):
        torch.backends.cudnn.benchmark = cfg.CUDNN_BENCHMARK
