## BERT-MRC-GLOBAL-POINTER-PYTORCH
This repository combine the methods of [MRC](https://github.com/ShannonAI/mrc-for-flat-nested-ner) and [GLOBAL POINTER](https://github.com/bojone/GlobalPointer) in 2022.11.24. In original MRC methods, the loss is composed of start, end and match which is not as efficient as single loss in global pointer. therefore this repository is created.

The most challenge part is the reflection between tokens and chars in both chinese and english.

I have down several experiments on msra and genia datasets which also can be download from MRC official website. The performance is under the small trick that is only sentences with entities are trained and tested.

For the msra dataset, it can achieve 0.9649 F1 with BERT encoder.

For the genia dataset, it achieve 0.8209 f1 with BERT and 0.8243 with ROBERTA, when trained with both train and dev set, the test score can reach 0.8347.

I encountered some small problems, such as using gradient accumulation and make global batch size to 32 which needs more epochs of 5 and the final is 0.8322. Using non acculation with only 8 batch size and 3 epochs, the best is 0.8347. The behaviour is not same with different batch sizes which is not tackled yet.