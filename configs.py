# Time: 2019-3-7 20:15:18
# Author: gaozhengjie

import argparse
from datetime import datetime

def get_config():
    parser = argparse.ArgumentParser()
    TIMESTAMP = "{0:%Y-%m-%d--%H-%M-%S/}".format(datetime.now())

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_config_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The config json file corresponding to the pre-trained BERT model. \n"
                             "This specifies the model architecture.")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--vocab_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The vocabulary file that the BERT model was trained on.")
    parser.add_argument("--output_dir",
                        default='log/'+TIMESTAMP,
                        type=str,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--model_save_path",
                        default='save_model/' + TIMESTAMP,
                        type=str,
                        help="The output directory where the model checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--init_checkpoint",
                        default=None,
                        type=str,
                        help="Initial checkpoint (usually from a pre-trained BERT model).")
    parser.add_argument("--do_lower_case",
                        default=False,
                        action='store_true',
                        help="Whether to lower case the input text. True for uncased models, False for cased models.")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        default=True,
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        default=True,
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict",
                        default=False,
                        help="Whether to run eval on the test set.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--save_checkpoints_steps",
                        default=1000,
                        type=int,
                        help="How often to save the model checkpoint.")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumualte before performing a backward/update pass.")
    parser.add_argument('--optimize_on_cpu',
                        default=False,
                        action='store_true',
                        help="Whether to perform optimization and keep the optimizer averages on CPU")
    parser.add_argument('--fp16',
                        default=False,
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=128,
                        help='Loss scaling, positive power of 2 values can improve fp16 convergence.')
    parser.add_argument('--gpu_id',
                        default=[],
                        type=list,
                        help=u'输入要指定的 GPU 编号')

    # parser.add_argument('--model_name', default='lstm', type=str)
    parser.add_argument('--model_name', default='fc', type=str)  # 全连接模型，即bert的输出后面加全连接
    parser.add_argument('--embed_dim', default=768, type=int)
    parser.add_argument('--n_filters', default=100, type=int)
    parser.add_argument('--filter_sizes', default=[1, 2, 3, 4], type=list)
    parser.add_argument('--dropout', default=0, type=float)
    # parser.add_argument('--output_dim', default=3, type=int)  # CNN里面的输出维度，也就是标签的个数
    parser.add_argument('--hidden_dim', default=300, type=int)  # 以前是150，跑aen_bert改的300
    parser.add_argument('--lstm_layers', default=1, type=int)
    parser.add_argument('--lstm_mean', default='maxpool', type=str)
    parser.add_argument('--keep_dropout', default=0.1, type=float)
    parser.add_argument('--hops', default=3, type=int)
    parser.add_argument('--para_LSR', default=0.2, type=float)

    return parser.parse_args()
