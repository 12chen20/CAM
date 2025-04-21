def parser_add_main_args(parser):


    #  lambda_sparse 
    parser.add_argument('--lambda_sparse', type=float, default=0.1, help="Sparse regularization strength")
    #  theta 
    parser.add_argument('--theta', type=float, default=0.2, help="Pruning threshold for sparse regularization")
    
    # setup and protocol
    parser.add_argument('--dataset', type=str, default='elliptic')# citeseer, pubmed,elliptic,twitch,arxiv
    parser.add_argument('--data_dir', type=str,
                        default='/content/drive/MyDrive/CAM/data/')
                        
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--seed', type=int, default=3407)#3407
    parser.add_argument('--runs', type=int, default=5,
                        help='number of distinct runs')
    parser.add_argument('--epochs', type=int, default=500)

    # model network
    parser.add_argument('--hidden_channels', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layers for deep methods')
    
    # 
    parser.add_argument('--model', type=str, default='cam',
                        choices=['cam', 'odin', 'eerm'])
    parser.add_argument('--backbone', type=str, default='gcn')
    

    # CaNet
    parser.add_argument('--backbone_type', type=str, default='gcn', choices=['gcn', 'gat'])
    parser.add_argument('--K', type=int, default=2,
                        help='num of domains, each for one graph convolution filter')
    parser.add_argument('--tau', type=float, default=1,
                        help='temperature for Gumbel Softmax')
    parser.add_argument('--env_type', type=str, default='node', choices=['node', 'graph'])
    parser.add_argument('--lamda', type=float, default=1.0,
                        help='weight for regularlization')
    parser.add_argument('--variant', action='store_true',help='set to use variant')

    # training
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--use_bn', action='store_true', help='use batch norm')

    # display and utility
    parser.add_argument('--display_step', type=int,
                        default=1, help='how often to print')
    parser.add_argument('--store_result', action='store_true',
                        help='whether to store results')
    parser.add_argument('--combine_result', action='store_true',
                        help='whether to combine all the ood environments')
