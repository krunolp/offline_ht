import datetime
import json
import os
from pathlib import Path
from tqdm import tqdm
import fire
import numpy as np
from loguru import logger
from pydantic import BaseModel

from ht_nns.exp_nn_run.train import main as train


def reset_wandb_env():
    exclude = {
        "WANDB_PROJECT",
        "WANDB_ENTITY",
        "WANDB_API_KEY",
    }
    for k, v in os.environ.items():
        if k.startswith("WANDB_") and k not in exclude:
            del os.environ[k]


class AnalysisOptions(BaseModel):
    iterations: int = 10000
    log_weights: bool = True
    batch_size_eval: int = 5000
    lrmin: float = 0.005
    lrmax: float = 0.1
    bs_min: int = 32
    bs_max: int = 256
    eval_freq: int = 3000
    risk_eval_freq: int = 1
    dataset: str = "mnist"
    data_path: str = "~/data/"
    model: str = "fc"
    save_folder: str = "./results_temp"
    depth: int = 3
    width: int = 128
    optim: str = "SGD"
    num_exp_lr: int = 6
    num_exp_bs: int = 6
    project_name: str = "mnist_ht"
    initial_weights: str = None
    resize: int = 28
    seed: int = 1234
    stopping_criterion: float = 1.e-2

    def __call__(self):
        save_folder = Path(self.save_folder)

        exp_folder = save_folder / str(datetime.datetime.now()).replace(" ", "_").replace(":", "_")
        exp_folder.mkdir(parents=True, exist_ok=True)

        log_file = exp_folder / "parameters.log.json"
        log_file.touch()

        logger.info(f"Saving log file in {log_file}")
        with open(log_file, "w") as log:
            json.dump(self.dict(), log, indent=2)

        if self.log_weights:
            weights_dir = exp_folder / "weights"
            weights_dir.mkdir(parents=True, exist_ok=True)
        else:
            weights_dir = None

        if self.lrmin > self.lrmax:
            raise ValueError(f"lrmin ({self.lrmin}) should be smaller than or equal to lmax ({self.lrmax})")

        eta = np.linspace(0.0001, 0.1, 20)
        bs = [1, 5, 10]
        perc = [0.25, 0.5, 0.75, 1.]

        logger.info(f"Launching {self.num_exp_lr * self.num_exp_bs} experiences")

        exp_num = 0
        experiment_results = {}

        for lr_ in tqdm(eta):
            for bs_ in bs:
                for perc_ in perc:

                    # Initial weights should be stored in
                    if self.log_weights:
                        str_ = "weights_" + str(exp_num) + "_lr_" + str(lr_) + "_bs_" + str(
                            bs_) + "_perc_" + str(perc_) + ".pth"
                        save_weights_file = weights_dir / str_
                        logger.info("Will save weights at " + str(save_weights_file))
                    else:
                        save_weights_file = None

                    exp_dict = train(
                        self.iterations,
                        bs_,
                        self.batch_size_eval,
                        lr_,
                        self.eval_freq,
                        self.dataset,
                        self.data_path,
                        self.model,
                        str(exp_folder),
                        self.depth,
                        self.width,
                        self.optim,
                        self.seed,
                        save_weights_file,
                        resize=self.resize,
                    )

                    experiment_results[exp_num] = exp_dict

                    save_path = Path(exp_folder) / f"results_{exp_num}.json"
                    with open(str(save_path), "w") as save_file:
                        json.dump(experiment_results, save_file, indent=2)

                    # Remove previously saved file
                    if exp_num >= 1:
                        if (Path(exp_folder) / f"results_{exp_num - 1}.json").exists():
                            os.remove(Path(exp_folder) / f"results_{exp_num - 1}.json")

                    exp_num += 1

        return str(exp_folder)


if __name__ == "__main__":
    fire.Fire(AnalysisOptions)
