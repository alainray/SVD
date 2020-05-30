import torch.nn as nn
import torch
from torch.optim import Adam, SGD
from torchvision.datasets import MNIST, STL10, CIFAR10
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from os.path import exists, join
from os import makedirs, listdir
import time


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        print('{}: {:.2f}s'.format(method.__name__, te - ts))
        return result, te - ts

    return timed


def get_loss(loss):
    losses = {'cross_entropy': nn.CrossEntropyLoss,
              'mse': nn.MSELoss,
              'bce': nn.BCELoss}
    return losses[loss.lower()]()


def get_optimizer(opt):
    opts = {'adam': Adam, 'sgd': SGD }
    return opts[opt.lower()]


def get_dataset(d, split='train', root=".", batch_size=32):
    train = split == 'train'
    t = ToTensor()

    if d == 'mnist':
        ds = MNIST(root, train=train, download=True, transform=t)
    elif d == 'cifar10':
        ds = CIFAR10(root, train=train, download=True, transform=t)
    else:
        ds = STL10(root, split=split, download=True, transform=t)

    dl = DataLoader(ds, batch_size=batch_size, shuffle=train)
    input_sizes = {'mnist': (1, 28, 28), 'cifar10': (3, 32, 32), 'stl10': (3, 96, 96)}
    n_classes = {'mnist': 10, 'cifar10': 10, 'stl10': 10}
    return dl, input_sizes[d], n_classes[d]


def init_metrics(exp_name):
    def init_sub_metrics():
        return {'min': [], 'max': [], 'mean': [], 'std_dev_max': [], 'std_dev_min': []}
    metrics = {'experiment': exp_name,
               'fc1': init_sub_metrics(),
               'fc2': init_sub_metrics(),
               'cls': init_sub_metrics(),
               'conv1': init_sub_metrics(),
               'conv2': init_sub_metrics(),
               'acc': {'train': [], 'test': []},
               'loss': {'train': [], 'test': []},
               'epoch': []
               }
    return metrics


def append_svd_metrics(series, new_data):
    for layer, metrics in new_data.items():
        for metric, value in metrics.items():
            series[layer][metric].append(value)
    return series


def append_regular_metrics(series, loss, acc, epoch):
    for split in ['train', 'test']:
        series['loss'][split].append(loss[split])
        series['acc'][split].append(acc[split])
    series['epoch'].append(epoch)
    return series


def save_metrics(metrics, output_dir="metrics"):
    if not exists(output_dir):
        makedirs(output_dir)
    filename = "{}.pkl".format(metrics['experiment'])
    filename = join(output_dir, filename)
    torch.save(metrics, filename)


def load_experiments(filename):
    experiments = []
    with open(filename, "r") as f:
        for line in f:
            experiments.append(line.replace("\n", ""))
        f.close()
    return experiments


def check_if_finished(experiment, done_experiments_dir):
    done_experiments = [exp.replace(".pkl", "") for exp in listdir(done_experiments_dir)]
    return experiment in done_experiments


@timeit
def run_epoch(model, dls, opt, criterion, args, metrics, epoch=1):
    l1 = nn.L1Loss()

    # Data for metrics
    total_loss = {'train': 0.0, 'test': 0.0}
    total_acc = {'train': 0.0, 'test': 0.0}
    n_batches = {'train': len(dls['train']), 'test': len(dls['test'])}

    for phase in ['train', 'test']:
        n_samples = 0.0
        for i, (imgs, labels) in enumerate(dls[phase]):
            labels = labels.cuda()
            imgs = imgs.cuda()
            opt.zero_grad()
            if phase == 'train':
                model.train()
            else:
                model.eval()
            n_samples += imgs.shape[0]
            with torch.set_grad_enabled(phase == 'train'):

                result = model(imgs)
                preds = result.argmax(axis=1)
                acc = preds == labels
                total_acc[phase] += acc.sum()
                loss_f = criterion(result, labels)
                if args.l1 == 1:
                    loss_f += l1(model.parameters())
                if phase == 'train':
                    loss_f.backward()
                    opt.step()
                    append_svd_metrics(metrics, model.get_sv_stats(args.mode))

            total_loss[phase] += loss_f.detach().cpu()
            texto = "Epoch: {} - ".format(epoch)
            print("\r" + texto + "Batch {}/{} {} Loss: {:.6f} Acc: {:.2f}%".format(i + 1,
                                                                                   n_batches[phase],
                                                                                   phase.upper(),
                                                                                   total_loss[phase] / n_samples,
                                                                                   100 * total_acc[phase] / n_samples)
                  , end="")
        total_loss[phase] = (total_loss[phase]/n_samples).item()
        total_acc[phase] = (100 * total_acc[phase]/n_samples).item()
        print("")
    append_regular_metrics(metrics, total_loss, total_acc, epoch)

    return metrics


@timeit
def run_experiment(exp_name, model, dls, opt, criterion, args):
    metrics = init_metrics(exp_name)
    print("Initiating main loop...")
    for epoch in range(1, args.epochs+1):
        print("Epoch {}/{}:".format(epoch, args.epochs))
        metrics, tt = run_epoch(model, dls, opt, criterion, args, metrics, epoch)
    save_metrics(metrics)


if __name__ == '__main__':
    a = torch.load("metrics/cnn-relu-100-dr_0.0-baseline-stl10-cross_entropy-adam-lr_0.001-l1_0-l2_0.0.pkl")
    print(a)

