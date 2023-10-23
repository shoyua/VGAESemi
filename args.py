import argparse


def read_args():
    parser = argparse.ArgumentParser(description='Variant Graph Auto Encoder')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Initial learning rate.')
    parser.add_argument('--epochs', '-e', type=int,
                        default=200, help='Number of epochs to train.')
    parser.add_argument('--sf_epochs', '-sf', type=int,
                        default=3000, help='Number of epochs to train.')
    parser.add_argument('--vae_hidden1', '-vh1', type=int,
                        default=1024, help='Number of units in hidden layer 1.')
    parser.add_argument('--vae_hidden2', '-vh2', type=int,
                        default=64, help='Number of units in hidden layer 2.')
    parser.add_argument('--classifier_dim1', '-cd1', type=int,
                        default=1024, help='Number of units in hidden layer 1.')
    parser.add_argument('--classifier_dim2', '-cd2', type=int,
                        default=512, help='Number of units in hidden layer 2.')
    parser.add_argument('--num_heads', '-nh', type=int,
                        default=8, help='Number of heads.')

    parser.add_argument('--repeats', '-r', type=int,
                        default=5, help='repeat num.')
    parser.add_argument('--datasrc', '-s', type=str, default='dgl',
                        help='Dataset download from dgl Dataset or website.')

    parser.add_argument('--dataset', '-d', type=str, default='pubmed',
                        help='choose form cora|pubmed|citeseer|coacs|coaphysics|coaphoto|coacomputer')

    parser.add_argument('--backbone', '-b', type=str,
                        default='gcn', help='choose from gat|gcn')

    parser.add_argument('--seed', type=int, default=42, help='seed num')

    parser.add_argument('--alpha', type=float, default=1,
                        help='cofficient alpha')
    parser.add_argument('--beta', type=float, default=0.1,
                        help='cofficient beta')
    parser.add_argument('--gamma', type=float, default=0.5,
                        help='cofficient gamma')
    parser.add_argument('--eta', type=float, default=0.1,
                        help='cofficient eta')
    parser.add_argument('--tau', type=float, default=0.8,
                        help='temperature coefficient')

    parser.add_argument('--feat_dp', type=float, default=0.6,
                        help='feature dropout ratio')
    parser.add_argument('--attn_dp', type=float, default=0.6,
                        help='attention dropout ratio')
    parser.add_argument('--dp_rate', type=float,
                        default=0.6, help='dropout ratio')

    parser.add_argument('--device', type=int, default=0, help='GPU id to use.')
    parser.add_argument("--use_cfg", action="store_true",
                        help="Set to True to read config file")
    parser.add_argument('--resi', type=int, default=1,
                        help='resisual connection')
    parser.add_argument('--debug', type=bool, default=True, help='debug mode')

    args = parser.parse_args()
    return args
