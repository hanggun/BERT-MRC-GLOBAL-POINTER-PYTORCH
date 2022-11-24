## BERT-MRC-GLOBAL-POINTER-PYTORCH
This repository combine the methods of [MRC](https://github.com/ShannonAI/mrc-for-flat-nested-ner) and [GLOBAL POINTER](https://github.com/bojone/GlobalPointer) in 2022.11.24. In original MRC methods, the loss is composed of start, end and match which is not as efficient as single loss in global pointer. therefore this repository is created.

The most challenge part is the reflection between tokens and chars in both chinese and english.

I have down several experiments on msra and genia datasets which also can be download from MRC official website. The performance is under the small trick that is only sentences with entities are trained and tested.

For the msra dataset, it can achieve 0.9649 F1 with BERT encoder.

For the genia dataset, it achieve 0.8209 f1 with BERT and 0.8243 with ROBERTA, when trained with both train and dev set, the test score can reach 0.8347. By using Efficient global pointer, it can achieve 0.8362 f1. Using roberta-large, the best f1 can reach 0.8398

I encountered some small problems, such as using gradient accumulation and make global batch size to 32 which needs more epochs of 5 and the final is 0.8322. Using non acculation with only 8 batch size and 3 epochs, the best is 0.8347. The behaviour is not same with different batch sizes which is not tackled yet.


# Citations
```
@misc{https://doi.org/10.48550/arxiv.2208.03054,
  doi = {10.48550/ARXIV.2208.03054},
  
  url = {https://arxiv.org/abs/2208.03054},
  
  author = {Su, Jianlin and Murtadha, Ahmed and Pan, Shengfeng and Hou, Jing and Sun, Jun and Huang, Wanwei and Wen, Bo and Liu, Yunfeng},
  
  keywords = {Computation and Language (cs.CL), Artificial Intelligence (cs.AI), Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences},
  
  title = {Global Pointer: Novel Efficient Span-based Approach for Named Entity Recognition},
  
  publisher = {arXiv},
  
  year = {2022},
  
  copyright = {Creative Commons Attribution Non Commercial No Derivatives 4.0 International}
}
```
```
@misc{https://doi.org/10.48550/arxiv.1910.11476,
  doi = {10.48550/ARXIV.1910.11476},
  
  url = {https://arxiv.org/abs/1910.11476},
  
  author = {Li, Xiaoya and Feng, Jingrong and Meng, Yuxian and Han, Qinghong and Wu, Fei and Li, Jiwei},
  
  keywords = {Computation and Language (cs.CL), FOS: Computer and information sciences, FOS: Computer and information sciences},
  
  title = {A Unified MRC Framework for Named Entity Recognition},
  
  publisher = {arXiv},
  
  year = {2019},
  
  copyright = {arXiv.org perpetual, non-exclusive license}
}

```