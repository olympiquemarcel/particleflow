"""
Developing a PyTorch Geometric supervised training of MLPF using DistributedDataParallel.

Authors: Farouk Mokhtar, Joosep Pata, Eric Wulff
"""

import argparse
import logging
from pathlib import Path

# comet needs to be imported before torch
from comet_ml import OfflineExperiment, Experiment  # noqa: F401, isort:skip

import yaml
from pyg.training import device_agnostic_run, override_config, run_hpo
from utils import create_experiment_dir

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()

# add default=None to all arparse arguments to ensure they do not override
# values loaded from the config file given by --config unless explicitly given
parser.add_argument("--config", type=str, default=None, help="yaml config")
parser.add_argument("--prefix", type=str, default=None, help="prefix appended to result dir name")
parser.add_argument("--data-dir", type=str, default=None, help="path to `tensorflow_datasets/`")
parser.add_argument("--gpus", type=str, default=None, help="to use CPU set to empty string; else e.g., `0,1`")
parser.add_argument(
    "--gpu-batch-multiplier", type=int, default=None, help="Increase batch size per GPU by this constant factor"
)
parser.add_argument(
    "--dataset", type=str, default=None, choices=["clic", "cms", "delphes"], required=False, help="which dataset?"
)
parser.add_argument("--num-workers", type=int, default=None, help="number of processes to load the data")
parser.add_argument("--prefetch-factor", type=int, default=None, help="number of samples to fetch & prefetch at every call")
parser.add_argument("--load", type=str, default=None, help="dir from which to load a saved model")
parser.add_argument("--train", action="store_true", default=None, help="initiates a training")
parser.add_argument("--test", action="store_true", default=None, help="tests the model")
parser.add_argument("--num-epochs", type=int, default=None, help="number of training epochs")
parser.add_argument("--patience", type=int, default=None, help="patience before early stopping")
parser.add_argument("--lr", type=float, default=None, help="learning rate")
parser.add_argument(
    "--conv-type", type=str, default=None, help="which graph layer to use", choices=["gravnet", "attention", "gnn_lsh"]
)
parser.add_argument("--make-plots", action="store_true", default=None, help="make plots of the test predictions")
parser.add_argument("--export-onnx", action="store_true", default=None, help="exports the model to onnx")
parser.add_argument("--ntrain", type=int, default=None, help="training samples to use, if None use entire dataset")
parser.add_argument("--ntest", type=int, default=None, help="training samples to use, if None use entire dataset")
parser.add_argument("--nvalid", type=int, default=None, help="validation samples to use")
parser.add_argument("--checkpoint-freq", type=int, default=None, help="epoch frequency for checkpointing")
parser.add_argument("--hpo", type=str, default=None, help="perform hyperparameter optimization, name of HPO experiment")
parser.add_argument("--local", action="store_true", default=None, help="perform HPO locally, without a Ray cluster")
parser.add_argument("--ray-cpus", type=int, default=None, help="CPUs per trial for HPO")
parser.add_argument("--ray-gpus", type=int, default=None, help="GPUs per trial for HPO")
parser.add_argument(
    "--load-checkpoint", type=str, default=None, help="which checkpoint to load. if None then will load best weights"
)
parser.add_argument("--comet", action="store_true", help="use comet ml logging")
parser.add_argument("--comet-offline", action="store_true", help="save comet logs locally")
parser.add_argument("--comet-step-freq", type=int, default=None, help="step frequency for saving comet metrics")
parser.add_argument("--experiments-dir", type=str, default=None, help="base directory within which trainings are stored")


def main():
    # torch.multiprocessing.set_start_method('spawn')

    args = parser.parse_args()
    world_size = len(args.gpus.split(","))  # will be 1 for both cpu ("") and single-gpu ("0")

    with open(args.config, "r") as stream:  # load config (includes: which physics samples, model params)
        config = yaml.safe_load(stream)

    # override loaded config with values from command line args
    config = override_config(config, args)

    if args.hpo:
        run_hpo(args, config)
    else:
        outdir = create_experiment_dir(
            prefix=(args.prefix or "") + Path(args.config).stem + "_",
            experiments_dir=args.experiments_dir if args.experiments_dir else "experiments",
        )
        device_agnostic_run(config, args, world_size, outdir)


if __name__ == "__main__":
    main()
