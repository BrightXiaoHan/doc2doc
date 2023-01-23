# G-Transformer

This example is for ACL 2021 paper [G-Transformer for Document-level Machine Translation](https://aclanthology.org/2021.acl-long.267/).

## Train sentence level model
Environment variable. Create a file `.env` and `source .env`.
```
FAIRSEQ_SRC=/path/to/fairseq
RAW_DATA_DIR=raw
DATA_DIR=data
DATA_BIN=databin
CHECKPOINTS=checkpoints
CUDA_VISIBLE_DEVICES=0
MAX_TOKENS=4096  # decide by gpu memory
SRC_LANG=en
TGT_LANG=zh
```
Put train, valid and test data into `RAW_DATA_DIR` like this:
```
├── raw_data_dir
│   ├── en.test
│   ├── zh.test
│   ├── en.train
│   ├── zh.train
│   ├── en.valid
│   └── zh.valid
```

Train sentencepiece model and tokenize data
```
python $FAIRSEQ_SRC/scripts/spm_train.py \
    --input=$RAW_DATA_DIR/$SRC_LANG.train,$RAW_DATA_DIR/$TGT_LANG.train \
    --model_prefix=$DATA_DIR/sentencepiece.bpe \
    --vocab_size=32000 \
    --character_coverage=1.0 \
    --normalization_rule_name=identity \
    --model_type=bpe \
    --input_sentence_size=2000000 \
    --shuffle_input_sentence=true

python $FAIRSEQ_SRC/scripts/spm_encode.py \
    --model $DATA_DIR/sentencepiece.bpe.model \
    --inputs $RAW_DATA_DIR/$SRC_LANG.train $RAW_DATA_DIR/$TGT_LANG.train \
    --outputs $DATA_DIR/train.$SRC_LANG $DATA_DIR/train.$TGT_LANG

python $FAIRSEQ_SRC/scripts/spm_encode.py \
    --model $DATA_DIR/sentencepiece.bpe.model \
    --inputs $RAW_DATA_DIR/en.valid $RAW_DATA_DIR/zh.valid \
    --outputs $DATA_DIR/valid.$SRC_LANG $DATA_DIR/valid.$TGT_LANG

python $FAIRSEQ_SRC/scripts/spm_encode.py \
    --model $DATA_DIR/sentencepiece.bpe.model \
    --inputs $RAW_DATA_DIR/en.test $RAW_DATA_DIR/zh.test \
    --outputs $DATA_DIR/test.$SRC_LANG $DATA_DIR/test.$TGT_LANG
```
Binarize data
```
fairseq-preprocess --source-lang $SRC_LANG --target-lang $TGT_LANG \
    --trainpref $DATA_DIR/train \
    --validpref $DATA_DIR/valid \
    --testpref $DATA_DIR/test \
    --joined-dictionary \
    --thresholdtgt 10 --thresholdsrc 10 \
    --destdir $DATA_BIN \
    --workers 40 
```
Train normal transformer model without adapters.
```
fairseq-train \
    $DATA_BIN \
    --arch transformer --share-all-embeddings \
    --source-lang en --target-lang zh \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.1 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens $MAX_TOKENS \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-remove-bpe sentencepiece \
    --eval-bleu-print-samples \
    --patience 5 \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --fp16 --save-dir $CHECKPOINTS
```
Evalueate bleu score on valid and test data.
```
fairseq-generate $DATA_BIN \
    --path $CHECKPOINTS/checkpoint_best.pt \
    --beam 5 --remove-bpe sentencepiece \
    --replace-unk --sacrebleu \
    --gen-subset test \
    | grep ^H | LC_ALL=C sort -V | cut -f3- | \
    sacrebleu -l $SRC_LANG-$TGT_LANG $RAW_DATA_DIR/$TGT_LANG.train
```

## Continue train document level model

```
```
