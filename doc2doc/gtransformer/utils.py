import torch


def tokens2tags(dict, tokens, eod):
    """
    generate group-tags according token sequence
    """

    def _toks2tags(tokens):
        tags = []
        next_tag = 1
        for tok in tokens:
            if tok in [dict.pad_index, dict.index(eod)]:
                tags.append(0)
            else:
                tags.append(next_tag)
                if tok == dict.eos_index:  # increase tag per </s>
                    next_tag += 1
        return tags

    tok_tags = [_toks2tags(tokens) for tokens in tokens.data.cpu().numpy().tolist()]
    tok_tags = torch.tensor(tok_tags, dtype=tokens.dtype, device=tokens.device)
    return tok_tags


def Linear(in_features, out_features, bias=True):
    m = torch.nn.Linear(in_features, out_features, bias)
    torch.nn.init.xavier_uniform_(m.weight)
    if bias:
        torch.nn.init.constant_(m.bias, 0.0)
    return m
