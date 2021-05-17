import subprocess
import pathlib
import os
import sys

sys.path.append('/home/gzj/anaconda3/lib/python3.6/site-packages/tqdm/')
if __name__ == '__main__':
    print("*"*100)
    print(sys.path)
    print("*"*100)
    # main_path = pathlib.Path('/home/gzj/sentiment/aspect_sentiment/')
    # os.chdir(main_path)
    print(sys.path)
    print(os.getcwd())
    subprocess.run(['python', '/home/gzj/sentiment/aspect_sentiment/run_classifier_word.py',
                    "--task_name=restaurant"
                    "--data_dir=/home/gzj/sentiment/aspect_sentiment/datasets/semeval14/restaurants/4way"
                    "--vocab_file=/home/gzj/sentiment/bert/uncased_L-12_H-768_A-12/vocab.txt"
                    "--bert_config_file=/home/gzj/sentiment/bert/uncased_L-12_H-768_A-12/bert_config.json"
                    "--init_checkpoint=/home/gzj/sentiment/bert/uncased_L-12_H-768_A-12/pytorch_model.bin"
                    "--max_seq_length=128"
                    "--train_batch_size 20"
                    "--eval_batch_size 20"
                    "--learning_rate 2e-5"
                    "--num_train_epochs 1.0"
                    "--model_name fc"
                    "--local_rank 4"
                    "--gpu_id 0,1,2"
                    "--output_dir log/temp"])
