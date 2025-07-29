import json
import subprocess
import argparse
import time
import numpy as np
import nemo_run as run
from nemo import lightning as nl
from nemo.collections import llm
from nemo.collections.llm.recipes.precision.mixed_precision import (
    fp16_mixed,
    fp16_with_fp8_mixed
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_size", type=str, default="8b", choices=["8b", "3b"])
    args, _ = parser.parse_known_args()
    return args

args = parse_args()

def load_config():
    with open("config.json") as f:
        return json.load(f)["LLAMA3Pretraining"]


def configure_recipe(cfg, nodes=1, gpus_per_node=4):
    precision = cfg.get("precision", "fp16").lower()
    plugin = fp16_with_fp8_mixed() if precision == "fp8" else fp16_mixed()

    model_size = args.model_size
    if model_size == "3b":
        recipe_fn = llm.llama32_3b.pretrain_recipe
    else:
        recipe_fn = llm.llama3_8b.pretrain_recipe

    recipe = recipe_fn(
        dir="/checkpoints/llama3_{model_size}",
        name=f"llama3_{model_size}_pretraining",
        num_nodes=nodes,
        num_gpus_per_node=gpus_per_node,
    )

    # Parallelism
    tp = cfg["parallelism"]["tp"]
    pp = cfg["parallelism"]["pp"]
    vp = cfg["parallelism"].get("vp")
    cp = cfg["parallelism"]["cp"]

    recipe.model.config.tensor_model_parallel_size = tp
    recipe.model.config.pipeline_model_parallel_size = pp
    recipe.model.config.virtual_pipeline_model_parallel_size = vp

    recipe.trainer.strategy.tensor_model_parallel_size = tp
    recipe.trainer.strategy.pipeline_model_parallel_size = pp
    recipe.trainer.strategy.virtual_pipeline_model_parallel_size = vp
    recipe.trainer.strategy.context_parallel_size = cp

    recipe.data.micro_batch_size = cfg["micro_batch_size"]

    recipe.trainer.plugins = plugin
    recipe.trainer.accelerator = "gpu"
    recipe.trainer.devices = gpus_per_node

    recipe.trainer.max_time = "0:04:00:00" # stop after 4 hours
    return recipe


def local_executor_torchrun(nodes=1, devices=4):
    env_vars = {
        "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
        "NCCL_NVLS_ENABLE": "0",
        "NVTE_DP_AMAX_REDUCE_INTERVAL": "0",
        "NVTE_ASYNC_AMAX_REDUCTION": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }
    return run.LocalExecutor(
        ntasks_per_node=devices,
        launcher="torchrun",
        env_vars=env_vars
    )


def run_pretraining():
    cfg = load_config()
    recipe = configure_recipe(cfg)

    executor = local_executor_torchrun(
        nodes=recipe.trainer.num_nodes,
        devices=recipe.trainer.devices
    )

    run.run(recipe, executor=executor, name=f"llama3_{args.model_size}_pretraining")


if __name__ == "__main__":
    run_pretraining()