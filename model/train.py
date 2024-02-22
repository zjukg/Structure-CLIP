import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
from utils import get_args, set_manualSeed, image_transform, WinoLoss, CLIPLoss, MarginLoss
from dataloader import CoCoDataset, CoCoDataset_aug, CoCoDataset_aug_object_attribute, CoCoDataset_aug_update, CoCoDataset_aug_update_withneg
from dataloader_downstream import VG_Relation, VG_Attribution
from clip import load
from model import myTransformer, bert_Transformer, triple_Transformer
from eval import eval_coco, eval_coco_large, eval_coco_batch, test_vg_relation, test_vg_attribution
import wandb
import json
from model_zoo import get_model

args = get_args()

wandb.init(project=args.project, name=args.name + "_lr" +str(args.lr)+ "_weight"+str(args.neg_loss_weight) + "_weight"+str(args.knowledge_weight) + "_layernum" + str(args.transformer_layer_num))
set_manualSeed(args)

idx2id = dict()
with open("data/test_coco_aug_havezero.json", "r") as f:
    infomation = json.load(f)
for idx, item in enumerate(infomation):
    id = item['id']
    idx2id[idx] = id

# CLIP 
clip_model, preprocess = load("ViT-B/32", jit=False)
clip_model = clip_model.cuda()

# Transformer
myTransformer = triple_Transformer().cuda()

# train & eval dataloader
train_vg_dataset = VG_Attribution(data_path=args.train_path, transform =image_transform())
train_vg_dataloader = DataLoader(train_vg_dataset, num_workers=8, batch_size=args.batch_size, shuffle=True)
test_vg_dataset =  VG_Attribution(data_path=args.test_path, transform =image_transform(is_train=False))
test_vg_dataloader = DataLoader(test_vg_dataset,  num_workers=8, batch_size=args.batch_size, shuffle=False)

test_dataset = CoCoDataset_aug_update(data_path="data/test_coco_aug_havezero.json", transform =image_transform(is_train=False))
test_dataloader = DataLoader(test_dataset,  num_workers=8, batch_size=args.batch_size, shuffle=False)
vg_relation_dataset = VG_Relation(transform =image_transform(is_train=False),)
vg_relation_dataloader = DataLoader(vg_relation_dataset, batch_size=128, num_workers=8)

loss = MarginLoss(margin=0.1)
loss_hingo = WinoLoss(margin=0.1)

optimizer = torch.optim.AdamW([
                                {'params': clip_model.parameters()},
                                {'params': myTransformer.parameters()}
                                ], lr=args.lr, weight_decay=args.weight_decay)

best_acc_test_attribution = 0
best_acc_test_relation = 0

for epoch in range(10):
    clip_model.train()
    myTransformer.train()
    for i, batch in enumerate(tqdm(train_vg_dataloader, total=len(train_vg_dataloader))):
        # # eval
        if i % 200 == 0:
            # eval task1
            t1, t5, t10, i1, i5, i10 = eval_coco_large(clip_model, myTransformer, test_dataloader, idx2id, args)
            wandb.log({"TextRank1": t1})
            wandb.log({"TextRank5": t5})
            wandb.log({"TextRank10": t10})
            wandb.log({"ImageRank1": i1})
            wandb.log({"ImageRank5": i5})
            wandb.log({"ImageRank10": i10})

            # eval task2
            acc_test_attribution = test_vg_attribution(clip_model, myTransformer, test_vg_dataloader,test_vg_dataset,  args)
            wandb.log({"acc_test_attribution": acc_test_attribution})           
   
            acc_test_relation = test_vg_relation(clip_model, myTransformer, vg_relation_dataloader, vg_relation_dataset, args)
            wandb.log({"acc_test_relation": acc_test_relation})            
        
        # train
        clip_model.train()
        myTransformer.train()
        optimizer.zero_grad()
        
        img = batch["image_options"][0].cuda()
        text_true = batch["caption_options"][1].squeeze(1).cuda()
        text_false = batch["caption_options"][0].squeeze(1).cuda()
        attention_mask = batch["attention_mask"].cuda()
        reversed_attention_mask = batch["reversed_attention_mask"].cuda()
                
        knowledge_emb = myTransformer(batch["head_inputs"], batch["relation_inputs"], batch["tail_inputs"], 0, attention_mask)
        knowledge_emb = knowledge_emb / knowledge_emb.norm(dim=1, keepdim=True)
        logit_img2text, text_features, image_features = clip_model(img, text_true)
        merge_text_features = text_features + knowledge_emb * args.knowledge_weight
        
        reversed_logit_img2text, reversed_text_features, reversed_image_features = clip_model(img, text_false)
        
        if epoch == 0 and i < 100:
            hingo_loss = loss_hingo(image_features, merge_text_features,1,False)
        else:
            hingo_loss = loss_hingo(image_features, merge_text_features, 1, True)

        neg_loss = loss(image_features, text_features, reversed_text_features, 1) * args.neg_loss_weight

        total_loss = hingo_loss + neg_loss
        total_loss.backward()
        optimizer.step()
        if i % 10 == 0:
            print('Epoch:{},  step:{},  loss:{}'.format(epoch, i, total_loss.item()))
            wandb.log({"Loss": total_loss.item()})

        # if i % 1000 == 0 and i > 0:
        #     torch.save(clip_model, 'checkpoints/triple_pair/epoch_{}_step_{}_WinoLoss.pt'.format(epoch, i))

    print('----------------------this is{}_th epoch----------------------------'.format(epoch))




