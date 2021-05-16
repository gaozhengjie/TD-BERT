# Experimental environment
- Python 3.7.4
- PyTorch 0.4.0


# How to run it?

## Step 1
Download the pretrained TensorFlow model: [uncased_L-12_H-768_A-12](https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip)

## Step 2
Change the TensorFlow Pretrained Model into Pytorch
```shell
cd  convert_tf_to_pytorch
```

```shell
export BERT_BASE_DIR=./uncased_L-12_H-768_A-12

python3 convert_tf_checkpoint_to_pytorch.py \
  --tf_checkpoint_path $BERT_BASE_DIR/bert_model.ckpt \
  --bert_config_file $BERT_BASE_DIR/bert_config.json \
  --pytorch_dump_path $BERT_BASE_DIR/pytorch_model.bin
```


## Step 3
Just run the command `sh run.sh` in terminal.

- In the file 'run.sh', you can set the number of times the program runs.
- task_name: has three options, 'laptop', 'restaurant' and 'tweet'.
- data_dir: dataset directory


# PS
if you have some questions, you can contact with me by E-mail: gaozhengj@foxmail.com