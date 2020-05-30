from easydict import EasyDict as edict
from os.path import exists, join
from itertools import product
import pickle
from os import makedirs

args = edict()
args.optimizers = ['sgd', 'adam']
args.activations = ['relu', 'tanh', 'sigmoid']
args.l1s = [True, False]
args.l2s = [0.00, 0.02]
args.lrs = [0.001]
args.losses = ['cross_entropy']
args.datasets = ['mnist', 'stl10', 'cifar10']
args.dropouts = [0.0, 0.2]
args.epochs = [100]
args.modes = ['baseline', 'spectral', 'thresh']
args.models = ['cnn', 'mlp']
args.hidden_dims = [100]
args.bs_trains = [128]
args.bs_vals = [512]
dict_labels = ['optimizer', 'activation', 'l1', 'l2', 'lr', 'loss', 'dataset',
               'dropout', 'epochs', 'mode', 'model', 'hidden_dim', 'bs_train', 'bs_val']


def get_exp_name(params):
    (opt, act, l1, l2, lr, loss, ds, drop, epoch, mode, model, h_dim, bs_train, bs_val) = params

    name = "{}-{}-{}-dr_{}-{}-{}-{}-{}-lr_{}-l1_{}-l2_{}".format(model, act, h_dim, drop, mode, ds,
                                                                 loss, opt, lr, int(l1), l2)
    return name


def create_single_exp(params):
    result = edict()
    for i, p in enumerate(params):
        result[dict_labels[i]] = p
    return result


def create_all_exps(args, output_dir="exps"):
    if not exists(output_dir):
        makedirs(output_dir)

    for i, comb in enumerate(product(*[v for v in args.values()])):
        exp_name = get_exp_name(comb)
        exp = create_single_exp(comb)
        output_path = join(output_dir,exp_name)
        print("Guardando experimento N° {} en {}".format(i+1, output_path))
        pickle.dump(exp, open(output_path, "wb"))


def load_experiment(exp_name, exp_dir='exps'):
    file_path = join(exp_dir, exp_name)
    print("Loading experiment from {}...".format(file_path))
    result = pickle.load(open(file_path, "rb"))
    return result



if __name__ == "__main__":
    create_all_exps(args)
