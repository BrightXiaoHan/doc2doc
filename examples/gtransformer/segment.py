""" Generate aligned parallel text
      </s> - separator between sentences
    e.g.
      Input: w1 w2 </s> w3 w4 </s> w5 w6 </s> ...
      Y1: w1 w2 </s> w3 w4 </s>
      Y2: w5 w6 </s>
      ...
"""

import argparse
import sys

import sentencepiece as spm
from tqdm import tqdm


def segment(spm_model, max_sent=20, min_sent=2, max_tok=512):

    sp = spm.SentencePieceProcessor()
    sp.Load(spm_model)

    def generate(src_sents, tgt_sents):

        cur_num_sent = 0
        cur_num_tok = 0
        cur_src_sents = []
        cur_tgt_sents = []

        for src_sent, tgt_sent in zip(src_sents, tgt_sents):
            src_len = len(src_sent)
            tgt_len = len(tgt_sent)
            if cur_num_sent >= max_sent or cur_num_tok >= max_tok:
                print(" </s> ".join(cur_src_sents) + "\t" + " </s> ".join(cur_tgt_sents))
                cur_num_sent = 0
                cur_num_tok = 0
                cur_src_sents = []
                cur_tgt_sents = []
            cur_src_sents.append(" ".join(src_sent))
            cur_tgt_sents.append(" ".join(tgt_sent))
            cur_num_sent += 1
            cur_num_tok += src_len + tgt_len

        if cur_num_sent >= min_sent:
            print(" </s> ".join(cur_src_sents) + "\t" + " </s> ".join(cur_tgt_sents))

    cur_src = []
    cur_tgt = []
    for line in tqdm(sys.stdin):
        line = line.strip()
        src, tgt = line.split("\t")

        if src == "<d>":
            assert tgt == "<d>"
            generate(cur_src, cur_tgt)
            cur_src = []
            cur_tgt = []
        else:
            src = sp.encode_as_pieces(src)
            tgt = sp.encode_as_pieces(tgt)
            cur_src.append(src)
            cur_tgt.append(tgt)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Generate aligned parallel text")
    parser.add_argument("--sentencepiece-model", "-m", type=str, required=True, help="SentencePiece model")
    parser.add_argument("--max-sent", type=int, default=20, help="maximum number of sentences per segment")
    parser.add_argument("--min-sent", type=int, default=2, help="minimum number of sentences per segment")
    parser.add_argument("--max-tok", type=int, default=512, help="maximum number of tokens per segment")
    args = parser.parse_args()
    segment(args.sentencepiece_model, args.max_sent, args.min_sent, args.max_tok)


if __name__ == "__main__":
    main()
