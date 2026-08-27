"""ML Job entry point: launch brats.train across the node's 4 GPUs.

Kept deliberately thin. PyTorchDistributor owns process/rank setup; all training
logic lives in brats.train, which stays runnable under plain torchrun so the same
code path is exercised locally and remotely.
"""

import argparse
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/segresnet_base.yaml")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lambda-cls", type=float, default=None)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--stage-uri", default="")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    from snowflake.ml.modeling.distributors.pytorch import (
        PyTorchDistributor,
        PyTorchScalingConfig,
        WorkerResourceConfig,
    )

    def train_func():
        import os

        import torch.distributed as dist
        from snowflake.ml.modeling.distributors.pytorch import get_context

        ctx = get_context()
        # Bridge Snowflake's context into the env vars torch DDP expects, so
        # brats.train needs no Snowflake-specific branch.
        os.environ["RANK"] = str(ctx.get_rank())
        os.environ["LOCAL_RANK"] = str(ctx.get_local_rank())
        os.environ["WORLD_SIZE"] = str(ctx.get_world_size())

        from brats.config import DataConfig
        from brats.train import TrainConfig, train

        cfg = TrainConfig.from_yaml(args.config)
        if args.epochs is not None:
            cfg.epochs = args.epochs
        if args.lambda_cls is not None:
            cfg.lambda_cls = args.lambda_cls
        if args.run_name:
            cfg.run_name = args.run_name
        if args.stage_uri:
            cfg.stage_uri = args.stage_uri

        train(cfg, DataConfig.load(), resume=args.resume)
        if dist.is_initialized():
            dist.destroy_process_group()

    distributor = PyTorchDistributor(
        train_func=train_func,
        scaling_config=PyTorchScalingConfig(
            num_nodes=1,                # GPU_NV_M is a single 4-GPU node
            num_workers_per_node=4,     # one worker per A10G
            resource_requirements_per_worker=WorkerResourceConfig(
                num_cpus=10,            # 44 vCPU / 4 ranks, leaving headroom
                num_gpus=1,
            ),
        ),
    )
    # No dataset_map: our data is .npz on a stage, not a Snowflake table, so
    # sharding is handled by DistributedSampler rather than ShardedDataConnector.
    distributor.run()
    return 0


if __name__ == "__main__":
    __return__ = main()
