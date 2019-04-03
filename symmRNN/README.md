# Symmetric RNN language model for PennTree and Wikitext-2

The code is adapted from Pytorch's RNN example, 
which is tested with Pytorch 0.3.0.

During training, if a keyboard interrupt (Ctrl-C) is received,
training is stopped and the current model is evaluated against the test dataset.

The `main.py` script accepts the following arguments:

```bash
optional arguments:
  -h, --help         show this help message and exit
  --data DATA        location of the data corpus
  --model MODEL      type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)
  --emsize EMSIZE    size of word embeddings
  --nhid NHID        number of hidden units per layer
  --nlayers NLAYERS  number of layers
  --lr LR            initial learning rate
  --clip CLIP        gradient clipping
  --epochs EPOCHS    upper epoch limit
  --batch-size N     batch size
  --bptt BPTT        sequence length
  --dropout DROPOUT  dropout applied to layers (0 = no dropout)
  --decay DECAY      learning rate decay per epoch
  --tied             tie the word embedding and softmax weights
  --seed SEED        random seed
  --cuda             use CUDA
  --log-interval N   report interval
  --save SAVE        path to save the final model
  --symm_hh          turn on symmetry on hh module
  --symm_ih          turn on symmetry on ih module
  --symm_type        which symmetry parameterization to be used
```

With these arguments, a variety of models can be tested.

# Commands to reproduce our experiments:
* full model 
```
CUDA_VISIBLE_DEVICES=0 python main.py --model LSTM --cuda --emsize 650 --nhid 650 --dropout 0.5 --epochs 50
```
| card | valid ppl | test ppl | time per epoch | dataset    |
|------|-----------|----------|:---------------|------------|
| T-B  | 99.72     | 94.47    | 288s           | wikitext-2 |
| T-B  | 84.65     | 81.40    | 79s            | penn-tree  |

* symmetry on `hh`
```
CUDA_VISIBLE_DEVICES=0 python main.py --model symmLSTM --symm_hh 1111 --symm_ih 0000 --symm_type wwt --cuda --emsize 650 --nhid 650 --dropout 0.5 --epochs 50
```
| card | valid ppl | test ppl | time per epoch | dataset    |
|------|-----------|----------|:---------------|------------|
| T-X  | 100.90    | 95.11    | 380s           | wikitext-2 |
| T-B  | 84.19     | 79.89    | 291s           | penn-tree  |


* symmetric both `hh` and `ih`
```
CUDA_VISIBLE_DEVICES=0 python main.py --model symmLSTM --symm_hh 1111 --symm_ih 1111 --symm_type wwt --cuda --emsize 650 --nhid 650 --dropout 0.5 --epochs 50
```

| card | valid ppl | test ppl | time per epoch | dataset    |
|------|-----------|----------|:---------------|------------|
| P100 | 103.23    | 98.40    | 307            | wikitext-2 |
| T-B  | 86.35     | 82.97    | 379s           | penn-tree  |

