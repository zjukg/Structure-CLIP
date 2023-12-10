# Structure-CLIP

 - [*Structure-CLIP: Enhance Multi-modal Language Representations with Structure Knowledge*]
 
 ## 🌈 Model Architecture
![Model_architecture](https://github.com/BigHyf/Structure-CLIP/blob/main/figure/model.png)

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

