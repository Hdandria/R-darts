import argparse
import glob
import logging
import os
import sys
import time

from architect import Architect
import genotypes as genotypes_module  # To look up genotypes dynamically
from genotypes import PRIMITIVES
from model_search import Network
import numpy as np
from operations import OPS, CellOp
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset
import utils


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("cifar")
    parser.add_argument("--data", type=str, default="../data", help="location of the data corpus")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=0.025, help="init learning rate")
    parser.add_argument("--learning_rate_min", type=float, default=0.001, help="min learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum")
    parser.add_argument("--weight_decay", type=float, default=3e-4, help="weight decay")
    parser.add_argument("--report_freq", type=float, default=50, help="report frequency")
    parser.add_argument("--gpu", type=int, default=0, help="gpu device id")
    parser.add_argument("--epochs", type=int, default=50, help="num of training epochs")
    parser.add_argument("--init_channels", type=int, default=16, help="num of init channels")
    parser.add_argument("--layers", type=int, default=8, help="total number of layers")
    parser.add_argument(
        "--model_path", type=str, default="saved_models", help="path to save the model"
    )
    parser.add_argument("--cutout", action="store_true", default=False, help="use cutout")
    parser.add_argument("--cutout_length", type=int, default=16, help="cutout length")
    parser.add_argument("--drop_path_prob", type=float, default=0.3, help="drop path probability")
    parser.add_argument("--save", type=str, default="EXP", help="experiment name")
    parser.add_argument("--seed", type=int, default=2, help="random seed")
    parser.add_argument("--grad_clip", type=float, default=5, help="gradient clipping")
    parser.add_argument("--train_portion", type=float, default=0.5, help="portion of training data")
    parser.add_argument(
        "--unrolled", action="store_true", default=False, help="use one-step unrolled validation loss"
    )
    parser.add_argument(
        "--arch_learning_rate", type=float, default=3e-4, help="learning rate for arch encoding"
    )
    parser.add_argument(
        "--arch_weight_decay", type=float, default=1e-3, help="weight decay for arch encoding"
    )
    parser.add_argument(
        "--recursive_genotypes", type=str, default=None,
        help="Comma-separated list of genotypes to use as primitives for recursive search (e.g., 'DARTS_V1,NASNET')"
    )
    parser.add_argument(
        "--n_samples", type=int, default=8,
        help="Number of genotypes to sample from the final distribution (Top-k Sampling)"
    )
    return parser


def setup_logging(save_path: str) -> None:
    log_format = "%(asctime)s %(message)s"
    logging.basicConfig(
        stream=sys.stdout, level=logging.INFO, format=log_format, datefmt="%m/%d %I:%M:%S %p"
    )
    fh = logging.FileHandler(os.path.join(save_path, "log.txt"))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)


CIFAR_CLASSES = 10


def main():
    global args  # to allow access in train/infer
    parser = build_parser()
    args = parser.parse_args()
    args.save = "search-{}-{}".format(args.save, time.strftime("%Y%m%d-%H%M%S"))
    utils.create_exp_dir(args.save, scripts_to_save=glob.glob("*.py"))
    setup_logging(args.save)

    if not torch.cuda.is_available():
        logging.info("no gpu device available. Switching to CPU.")
        device = torch.device("cpu")
    else:
        logging.info("gpu device = %d", args.gpu)
        torch.cuda.set_device(args.gpu)
        cudnn.benchmark = True
        torch.cuda.manual_seed(args.seed)
        cudnn.enabled = True
        device = torch.device("cuda")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    logging.info("args = %s", args)

    # --------------------------------------------------------------------------
    # R-DARTS Logic: Setup Primitives
    # --------------------------------------------------------------------------
    current_primitives = PRIMITIVES
    current_ops = OPS

    if args.recursive_genotypes:
        logging.info(f"[R-DARTS] Recursive mode enabled. Loading genotypes: {args.recursive_genotypes}")
        try:
            # Create a copy of OPS to avoid polluting the global state for subsequent runs (if any)
            current_ops = OPS.copy()
            current_primitives = ["none"] # Start with 'none' (required for sparsity)

            # Parse the comma-separated list of genotype names
            genotype_names = [name.strip() for name in args.recursive_genotypes.split(',')]

            for name in genotype_names:
                if not name:
                    continue  # Skip empty strings

                # Dynamically load the genotype from the genotypes module
                loaded_genotype = getattr(genotypes_module, name)

                # Register the new recursive primitive
                # We use the lower-case name as the primitive key
                primitive_key = name.lower()

                # Capture loaded_genotype in the lambda default arg to avoid late binding issues in loop
                current_ops[primitive_key] = lambda C, stride, affine, g=loaded_genotype: CellOp(g, C, stride, affine)

                # Add to the primitives list
                current_primitives.append(primitive_key)

            logging.info(f"[R-DARTS] New search space primitives (STRICT): {current_primitives}")

        except AttributeError as e:
            logging.error(f"Error loading genotypes: {e}")
            sys.exit(1)
    else:
        logging.info("[R-DARTS] Standard mode (Level 0). Using default primitives.")

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)

    # Initialize Network with potentially recursive primitives
    model = Network(
        args.init_channels,
        CIFAR_CLASSES,
        args.layers,
        criterion,
        primitives=current_primitives,
        ops=current_ops
    )

    model = model.to(device)
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    train_transform, valid_transform = utils._data_transforms_cifar10(args)
    train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))

    train_queue = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(indices[:split]),
        pin_memory=True,
        num_workers=2,
    )

    valid_queue = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(indices[split:num_train]),
        pin_memory=True,
        num_workers=2,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs, eta_min=args.learning_rate_min
    )

    architect = Architect(model, args)

    for epoch in range(args.epochs):
        lr = scheduler.get_last_lr()[0]

        logging.info("epoch %d lr %e", epoch, lr)

        genotype = model.genotype()
        logging.info("genotype = %s", genotype)

        # print(F.softmax(model.alphas_normal, dim=-1))
        # print(F.softmax(model.alphas_reduce, dim=-1))

        # training
        print(lr)
        train_acc, train_obj = train(
            train_queue, valid_queue, model, architect, criterion, optimizer, lr, device
        )
        logging.info("train_acc %f", train_acc)
        scheduler.step()

        # validation
        valid_acc, valid_obj = infer(valid_queue, model, criterion, device)
        logging.info("valid_acc %f", valid_acc)

        utils.save(model, os.path.join(args.save, "weights.pt"))

    # Final deterministic genotype (argmax over alphas)
    best_genotype = model.genotype()
    logging.info("Best (argmax) Genotype: %s", best_genotype)

    # Final Sampling of Genotypes (Top-k Sampling)
    logging.info("Sampling %d genotypes from the final architecture distribution...", args.n_samples)
    sampled_genotypes = model.sample_genotypes(args.n_samples)
    for i, g in enumerate(sampled_genotypes):
        logging.info("Sampled Genotype %d: %s", i + 1, g)


def train(
    train_queue: torch.utils.data.DataLoader,
    valid_queue: torch.utils.data.DataLoader,
    model: Network,
    architect: Architect,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    lr: float,
    device: torch.device,
) -> tuple[float, float]:
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()

    for step, (input, target) in enumerate(train_queue):
        model.train()
        n = input.size(0)

        input = input.to(device)
        target = target.to(device, non_blocking=True)

        # get a random minibatch from the search queue with replacement
        input_search, target_search = next(iter(valid_queue))
        input_search = input_search.to(device)
        target_search = target_search.to(device, non_blocking=True)

        architect.step(
            input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled
        )

        optimizer.zero_grad()
        logits = model(input)
        loss = criterion(logits, target)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        if step % args.report_freq == 0:
            logging.info(
                "train: step= %03d, loss = %e, top1_accuracy = %f, top 5_accuracy = %f",
                step,
                objs.avg,
                top1.avg,
                top5.avg,
            )

    return top1.avg, objs.avg


def infer(
    valid_queue: torch.utils.data.DataLoader,
    model: Network,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    with torch.no_grad():
        for step, (input, target) in enumerate(valid_queue):
            input = input.to(device)
            target = target.to(device, non_blocking=True)

            logits = model(input)
            loss = criterion(logits, target)

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

            if step % args.report_freq == 0:
                logging.info("valid %03d %e %f %f", step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


if __name__ == "__main__":
    main()
