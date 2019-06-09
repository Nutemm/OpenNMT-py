# GitHub repository for my CS224U project: adapting Transformer to text classification.


## For more indications about how the code works, see OpenNMT-py source repository: https://github.com/OpenNMT/OpenNMT

Below is the main workflow to make the code run and get results:

To make the code run on SST dataset, you first need to preprocess it:

```bash
python preprocess.py -train_src SST/src_train.txt -train_tgt SST/tgt_train.txt -valid_src SST/src_val.txt -valid_tgt SST/tgt_val.txt -save_data SST/SST --src_vocab_size 50000 --tgt_vocab_size 10 --src_seq_length 100000 --tgt_seq_length 100000
```

Or for the character-level version:

```bash
python preprocess.py -train_src SST/char/src_train.txt -train_tgt SST/char/tgt_train.txt -valid_src SST/char/src_val.txt -valid_tgt SST/char/tgt_val.txt -save_data SST/char/SST --src_vocab_size 50000 --tgt_vocab_size 10 --src_seq_length 100000 --tgt_seq_length 100000
```

Then you need to train your model on this dataset. Here is the code for a classic train with the vanilla transformer:


```bash
python train.py -data SST/SST -save_model saved_models/sst_transfo\
        -layers 6 -rnn_size 512 -word_vec_size 512 -transformer_ff 2048 -heads 8  \
        -encoder_type transformer -decoder_type transformer -position_encoding \
        -train_steps 20000 -early_stopping 50 -max_generator_batches 2 -dropout 0.1 \
        -batch_size 1024 -batch_type tokens -normalization tokens  -accum_count 2 \
        -optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 8000 -learning_rate 2 \
        -max_grad_norm 0 -param_init 0  -param_init_glorot -valid_batch_size 8\
        -label_smoothing 0.1 -valid_steps 500 -save_checkpoint_steps 500 \
        -tensorboard -tensorboard_log_dir runs/onmt/sst_transfo \
        -world_size 1 -gpu_ranks 0 \
```

Once you trained your model, you can run it on a test dataset:

```bash
python translate.py -model saved_models/sst_transfo_step_1000.pt -src SST/src_test.txt -output SST/predictions.txt -verbose -gpu 0
```

Finally, to compute the accuracy:

```bash
python compute_accuracy_from_file.py SST/tgt_test.txt SST/predictions.txt
```

For the AG-news dataset, you first need to download it and to put all the files in the folder ag-news. Then run the jupyter notebooks to create the datasets. Finally, you can apply the workflow described just above.


## Acknowledgements

OpenNMT-py is run as a collaborative open-source project.
The original code was written by [Adam Lerer](http://github.com/adamlerer) (NYC) to reproduce OpenNMT-Lua using Pytorch.

Major contributors are:
[Sasha Rush](https://github.com/srush) (Cambridge, MA)
[Vincent Nguyen](https://github.com/vince62s) (Ubiqus)
[Ben Peters](http://github.com/bpopeters) (Lisbon)
[Sebastian Gehrmann](https://github.com/sebastianGehrmann) (Harvard NLP)
[Yuntian Deng](https://github.com/da03) (Harvard NLP)
[Guillaume Klein](https://github.com/guillaumekln) (Systran)
[Paul Tardy](https://github.com/pltrdy) (Ubiqus / Lium)
[François Hernandez](https://github.com/francoishernandez) (Ubiqus)
[Jianyu Zhan](http://github.com/jianyuzhan) (Shanghai)
[Dylan Flaute](http://github.com/flauted (University of Dayton)
and more !

OpentNMT-py belongs to the OpenNMT project along with OpenNMT-Lua and OpenNMT-tf.
