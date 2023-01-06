# doc2doc
Document level translation built upon [fairseq](https://github.com/facebookresearch/metaseq.git).

## Quick Start
Just install fairseq following [fairseq doc](./third_party/fairseq/README.md).
Follow instructions from [example](./examples/) to train and test different models.

We also have more detailed READMEs to reproduce results from specific papers
- [g-transformer](./examples/gtransformer/): [Paper](https://aclanthology.org/2021.acl-long.267/), [Original repo](https://github.com/baoguangsheng/g-transformer)


## Citation
fairseq
```
@inproceedings{ott2019fairseq,
  title = {fairseq: A Fast, Extensible Toolkit for Sequence Modeling},
  author = {Myle Ott and Sergey Edunov and Alexei Baevski and Angela Fan and Sam Gross and Nathan Ng and David Grangier and Michael Auli},
  booktitle = {Proceedings of NAACL-HLT 2019: Demonstrations},
  year = {2019},
}
```
G-Transformer
```
@article{bao2021g,
  title={G-transformer for document-level machine translation},
  author={Bao, Guangsheng and Zhang, Yue and Teng, Zhiyang and Chen, Boxing and Luo, Weihua},
  journal={arXiv preprint arXiv:2105.14761},
  year={2021}
}
```
