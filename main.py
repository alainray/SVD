from utils import *
from models import Model
from exp_gen import load_experiment
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("exp_list", help="File with list of experiments to run")
parser.add_argument("--exp_dir", default="exps", help="Directory with experiments")
parser.add_argument("--output_dir", default="metrics", help="File with list of experiments to run")
prog_args = parser.parse_args()

torch.manual_seed(0)    # For reproducibility
# Handle args
experiments = load_experiments(prog_args.exp_list)
print(experiments)
for exp_name in experiments:
    if not check_if_finished(exp_name, prog_args.output_dir):
        args = load_experiment(exp_name, prog_args.exp_dir)
        print("Loaded arguments for experiment are:")
        for arg_name, arg_value in args.items():
            print("{}: {}".format(arg_name, arg_value))
        # Preparing our data
        train_ds, input_size, n_classes = get_dataset(args.dataset, 'train', ".", args.bs_train)
        test_ds, *_ = get_dataset(args.dataset, 'test', ".", args.bs_val)
        dls = {'train': train_ds, 'test': test_ds}
        # Model
        model = Model(args.model, input_size, args.hidden_dim, n_classes)
        model.to('cuda')

        # Optimizer
        loss = get_loss(args.loss)
        opt = get_optimizer(args.optimizer)(model.parameters(), lr=args.lr, weight_decay=args.l2)
        _, tt = run_experiment(exp_name, model, dls, opt, loss, args)
        print("Total experiment time: {:.2f}s".format(tt))
