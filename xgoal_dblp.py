import argparse
from models import XGOAL


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', nargs='?', default='dblp')
    parser.add_argument('--model', type=str, default='xgoal')

    parser.add_argument('--hid_units', type=int, default=128, help='hidden dimension')
    parser.add_argument('--nb_epochs', type=int, default=20000, help='the maximum number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--patience', type=int, default=100, help='patience for early stopping')
    parser.add_argument('--gpu_num', type=int, default=0, help='the id of gpu to use')

    # path
    parser.add_argument('--save_root', type=str, default="./saved_models", help='root for saving the model')
    parser.add_argument('--pretrained_model_path', type=str, default="./example_ckpts/warmup_dblp_xgoal.pkl",
                        help='path to the pretrained model')

    # hyper-parameters for info-nce
    parser.add_argument('--p_drop', type=float, default=0.5, help='dropout rate for attributes')

    # hyper-parameters for clusters
    parser.add_argument('--k', type=list, default=[4, 4, 4], help='the numbers of clusters')
    parser.add_argument('--tau', type=list, default=[1, 1, 1], help='the temperature of clusters')
    parser.add_argument('--w_cluster', type=list, default=1e-2, help='weight for cluster loss')
    parser.add_argument('--cluster_step', type=int, default=5, help='every n steps to perform clustering')

    # hyper-parameters for alignment
    parser.add_argument('--w_reg_n', type=float, default=1e-5, help='weight for node level alignment regularization')
    parser.add_argument('--w_reg_c', type=float, default=1e-4, help='weight for cluster level alignment regularization')

    # hyper-parameters for differnet layers
    parser.add_argument('--w_list', type=list, default=[1, 1e-5, 1], help="weights for different layers")

    # warm-up
    parser.add_argument('--is_warmup', type=bool, default=False, help='whether to warm up or not')
    parser.add_argument('--warmup_lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--warmup_w_reg_n', type=float, default=1e-5, help='weight for node level alignment regularization')

    return parser.parse_known_args()


def printConfig(args):
    arg2value = {}
    for arg in vars(args):
        arg2value[arg] = getattr(args, arg)
    print(arg2value)


def main():
    args, unknown = parse_args()
    printConfig(args)

    model = XGOAL(args)
    model.train()
    model.evaluate()


def evaluate(path="./example_ckpts/dblp_xgoal.pkl"):
    args, unknown = parse_args()
    printConfig(args)

    model = XGOAL(args)
    model.evaluate(path)


if __name__ == '__main__':
    main()
    # evaluate()
