# Structure-CLIP
![](https://img.shields.io/badge/version-1.0.1-blue)
[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/zjukg/DUET/blob/main/licence)
[![arxiv badge](https://img.shields.io/badge/arxiv-2305.06152-red)](https://arxiv.org/abs/2305.06152)
[![AAAI](https://img.shields.io/badge/AAAI-2024-%23f1592a?labelColor=%23003973&color=%23be1c1a)](https://aaai.org/Conferences/AAAI-24/)
[![Pytorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?e&logo=PyTorch&logoColor=white)](https://pytorch.org/)
 - [*Structure-CLIP: Towards Scene Graph Knowledge to Enhance Multi-modal Structured Representations*](https://arxiv.org/abs/2305.06152)

>This paper introduces an end-to-end framework Structure-CLIP, which integrates **Scene Graph Knowledge** to enhance multi-modal structured representations. 

 ## 🔔 News
- **`2024-02` We preprint our Survey [Knowledge Graphs Meet Multi-Modal Learning: A Comprehensive Survey](http://arxiv.org/abs/2402.05391)  [[`Repo`](https://github.com/zjukg/KG-MM-Survey)].**
- **`2023-12`** Our paper: [Structure-CLIP: Towards Scene Graph Knowledge to Enhance Multi-modal Structured Representations](https://arxiv.org/abs/2305.06152) was accepted by **`AAAI 2024`**
- **`2022-12`** We release the [[Repo](https://github.com/zjukg/DUET)] for our **`AAAI 2023`** paper: [DUET: Cross-modal Semantic Grounding for Contrastive Zero-shot Learning](https://arxiv.org/abs/2207.01328)
 
 ## 🌈 Model Architecture
![Model_architecture](https://github.com/zjukg/Structure-CLIP/blob/main/figure/model.png)

## 📚 Dataset Download
Training datasets are available [`here`](https://drive.google.com/drive/folders/1kmH33IAclyZcHAbchtt2Hk_9YAw6PtJx?usp=sharing) .

## 📕 Code Path

#### Code Structures
There are four parts in the code.
- **model**: It contains the main files for Structure-CLIP network.
- **data**: It contains the pre-training data splits and downstream dataset.
- **checkpoints**: It saves checkpoint for reloading.
- **script**: The training scripts for Structure-CLIP.

## 🔬 Dependencies

- ```Python 3```
- ```PyTorch >= 1.8.0```
- ```Transformers>= 4.11.3```
- ```NumPy```
- All experiments are performed with one A100 GPU.

## 🚀 Train & Eval

The training script:
```shell
bash script/run.sh
```

### [Parameter](#content)
```
[--train_path TRAIN_PATH] [--test_path TEST_PATH] [--nepoch NEPOCH] [--batch_size BATCH_SIZE] [--manualSeed MANUAL_SEED]
[--lr LEARNING-RATE] [--weight_decay WEIGHT_DECAY] [--knowledge_weight KNOWLEDGE_WEIGHT] [--transformer_layer_num NUMBER] [--model_name MODEL_NAME] [--neg_loss_weight NEG_LOSS_WEIGHT] 
```

**Note**: 
- you can open the `.sh` file for <a href="#Parameter">parameter</a> modification.

## 🤝 Cite:
Please consider citing this paper if you use the ```code``` or ```data``` from our work.
Thanks a lot :)

```bigquery
@inproceedings{DBLP:conf/aaai/StructureCLIP,
  author       = {Yufeng Huang and
                  Jiji Tang and
                  Zhuo Chen and
                  Rongsheng Zhang and
                  Xinfeng Zhang and
                  Weijie Chen and
                  Zeng Zhao and
                  Zhou Zhao and
                  Tangjie Lv and
                  Zhipeng Hu and
                  Wen Zhang},
  title        = {Structure-CLIP: Towards Scene Graph Knowledge to Enhance Multi-modal Structured Representations},
  booktitle    = {{AAAI}},
  publisher    = {{AAAI} Press},
  year         = {2024}
}
```

