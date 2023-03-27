from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from loguru import logger

from ht_nns.models import alexnet, fc_mnist, lenet
from ht_nns.exp_nn_run.dataset import get_data_
from ht_nns.exp_nn_run.eval import eval


def accuracy(out, y):
    _, pred = out.max(1)
    correct = pred.eq(y)
    return 100 * correct.sum().float() / y.size(0)


def get_weights(net):
    with torch.no_grad():
        w = []

        for p in net.parameters():
            w.append(p.view(-1).detach().to(torch.device('cpu')))
        return torch.cat(w).cpu().numpy()


def get_weights_not_concat(net):
    with torch.no_grad():
        w = []

        for p in net.parameters():
            w.append(p.view(-1).detach().to(torch.device('cpu')))

        w = [x.cpu().numpy() for x in w]
        return w


def main(iterations: int = 10000,
         batch_size_train: int = 100,
         batch_size_eval: int = 1000,
         lr: float = 1.e-1,
         eval_freq: int = 1000,
         dataset: str = "mnist",
         data_path: str = "~/data/",
         model: str = "fc",  # TODO: use this arg (CNN,...)
         save_folder: str = "results_temp",
         depth: int = 3,
         width: int = 128,
         optim: str = "SGD",
         seed: int = 42,
         save_weights_file: str = None,
         weight_file: str = None,
         resize: int = None,
         burn_in: float = 0.9,
         ):
    # Creating files to save results_temp
    save_folder = Path(save_folder)
    assert save_folder.exists(), str(save_folder)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logger.info(f"on device {str(device)}")
    logger.info(f"Random seed ('torch.manual_seed'): {seed}")
    torch.manual_seed(seed)

    # training setup
    if dataset not in ["mnist", "cifar10"]:
        raise NotImplementedError(f"Dataset {dataset} not implemented, should be in ['mnist', 'cifar10']")
    _, _, _, num_classes = get_data_(dataset, data_path,
                                           batch_size_train,
                                           batch_size_eval,
                                           resize)

    SCALE = 64
    if model == 'fc':
        if dataset == 'mnist':
            input_size = resize ** 2 if resize is not None else 28 ** 2
            net = fc_mnist(input_dim=input_size, width=width, depth=depth, num_classes=num_classes).to(device)
        elif dataset == 'cifar10':
            net = fc_mnist(input_dim=32 * 32 * 3, width=width, depth=depth, num_classes=num_classes).to(device)
    elif model == 'alexnet':
        if dataset == 'mnist':
            net = alexnet(input_height=28, input_width=28, input_channels=1, num_classes=num_classes).to(device)
        else:
            net = alexnet(ch=SCALE, num_classes=num_classes).to(device)
    elif model == "lenet":
        if dataset == "mnist":
            net = lenet(input_channels=1, height=28, width=28).to(device)
        else:
            net = lenet().to(device)
    else:
        raise NotImplementedError

    logger.info("Network: ")

    if weight_file is not None:
        logger.info(f"Loading weights from {str(weight_file)}")

        if Path(weight_file).suffix == ".pth":
            net.load_state_dict(torch.load(str(weight_file)))
        elif Path(weight_file).suffix == ".pyT":
            net = torch.load(str(weight_file))
        net = net.to(device)

    crit = nn.CrossEntropyLoss().to(device)
    crit_unreduced = nn.CrossEntropyLoss(reduction="none").to(device)

    def cycle_loader(dataloader):
        while 1:
            for data in dataloader:
                yield data

    # training logs per iteration
    training_history = []

    # eval logs less frequently
    evaluation_history_TEST = []
    evaluation_history_TRAIN = []

    # initialize results_temp of the experiment, returned if didn't work
    exp_dict = {}

    # weights
    weights_history = deque([])

    # Defining optimizer
    opt = getattr(torch.optim, optim)(
        net.parameters(),
        lr=lr,
    )

    # start of one exp
    n_iters = 1
    weight_samples = []
    for out_iter in range(n_iters):
        CONVERGED = False
        saving_dict = dict({"last_iterate": net.state_dict(), "run_average": net.state_dict()})

        torch.manual_seed(seed + out_iter)
        # training setup
        if dataset not in ["mnist", "cifar10"]:
            raise NotImplementedError(f"Dataset {dataset} not implemented, should be in ['mnist', 'cifar10']")
        train_loader, test_loader_eval, train_loader_eval, num_classes = get_data_(dataset,
                                                                                         data_path,
                                                                                         batch_size_train,
                                                                                         batch_size_eval,
                                                                                         resize)
        circ_train_loader = cycle_loader(train_loader)

        logger.info("Starting training for: " + str(iterations) + " iterations")
        for i, (x, y) in enumerate(circ_train_loader):

            if i % eval_freq == 0 and (not CONVERGED):
                logger.info(f"Evaluation at iteration {i}, step " + str(out_iter) + "out of" + str(n_iters))
                te_hist, *_ = eval(test_loader_eval, net, crit_unreduced, opt)
                evaluation_history_TEST.append([i, *te_hist])
                logger.info(f"Evaluation on test set at iteration {i} finished ✅, accuracy: {round(te_hist[1], 3)}")

                tr_hist, losses, outputs = eval(train_loader_eval, net, crit_unreduced, opt)
                logger.info(f"Training accuracy at iteration {i}: {round(tr_hist[1], 3)}%")

            net.train()

            x, y = x.to(device), y.to(device)

            opt.zero_grad()
            out = net(x)
            loss = crit(out, y)

            if i % 1000 == 0:
                logger.info(f"Loss at iteration {i}: {loss.item()}")

            if torch.isnan(loss):
                logger.error('Loss has gone nan')
                break

            # calculate the gradients
            loss.backward()

            # record training history (starts at initial point)
            training_history.append([i, loss.item(), accuracy(out, y).item()])

            # take the step
            opt.step()

            if i > iterations:
                CONVERGED = True
            nan_flag = "False"
            if i > int(burn_in * iterations - 1):
                k = i - int(burn_in * iterations)
                temp_list = []
                for key in saving_dict['run_average'].keys():
                    saving_dict['run_average'][key] = (saving_dict['run_average'][key] * k + net.state_dict()[key]) / (
                            k + 1)
                    temp_list.append(net.state_dict()[key])
                    nan_flag = "True" if any(np.isnan(np.concatenate([x.flatten() for x in temp_list]))) else "False"
            if nan_flag == "True":
                logger.info(str(nan_flag) + str(i) + str(batch_size_train) + str(lr))

            # clear cache
            torch.cuda.empty_cache()

            # final evaluation and saving results_temp
            if CONVERGED:

                weights = get_weights_not_concat(net)
                weight_samples.append(weights)

                logger.debug('eval time {}'.format(i))
                te_hist, *_ = eval(test_loader_eval, net, crit_unreduced, opt)
                tr_hist, *_ = eval(train_loader_eval, net, crit_unreduced, opt)

                evaluation_history_TEST.append([i + 1, *te_hist])
                evaluation_history_TRAIN.append([i + 1, *tr_hist])

                test_acc = evaluation_history_TEST[-1][2]
                train_acc = evaluation_history_TRAIN[-1][2]

                exp_dict = {
                    "train_acc": train_acc,
                    "eval_acc": test_acc,
                    "acc_gap": train_acc - test_acc,
                    "loss_gap": te_hist[0] - tr_hist[0],
                    "test_loss": te_hist[0],
                    "learning_rate": lr,
                    "batch_size": int(batch_size_train),
                    "LB_ratio": lr / batch_size_train,
                    "depth": depth,
                    "width": width,
                    "model": model,
                    "last_losses": f"outputs_loss_{lr}_{batch_size_train}.npy",
                    "iterations": i
                }

                # Saving weights
                if save_weights_file is not None:
                    logger.info(f"Saving last weights in {str(save_weights_file)}")
                    torch.save(saving_dict, str(save_weights_file))
                    exp_dict["saved_weights"] = str(save_weights_file)
                break

    return exp_dict
