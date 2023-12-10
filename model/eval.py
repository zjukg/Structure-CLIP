import time
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.nn import functional as F
from utils import image_transform, compute_logits, WinoLoss
from dataloader import Mydataset
from clip import load, tokenize
from PIL import Image
from tqdm import tqdm
import numpy as np
import clip


def eval_coco(clip_model, dataloader):
    clip_model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    start = time.time()
    text_embedding, img_embedding = torch.tensor([]), torch.tensor([])
    print('loading data')
    for i, batch in enumerate(dataloader):
        id, img, text_true = batch
        text_true = text_true.squeeze(1).to(device)
        img = img.to(device)
        with torch.no_grad():
            text = clip_model.encode_text(text_true)
            text = text / text.norm(dim=1, keepdim=True)
            img = clip_model.encode_image(img)
            img = img / img.norm(dim=1, keepdim=True)
        if i == 0:
            text_embedding, img_embedding = text, img
            continue
        text_embedding = torch.cat((text_embedding, text), 0)
        img_embedding = torch.cat((img_embedding, img), 0)
    print("loading success")
    text_embedding = text_embedding.to('cpu')
    img_embedding = img_embedding.to('cpu')
    text_sim = text_embedding @ img_embedding.T
    img_sim = img_embedding @ text_embedding.T

    TextRank1, TextRank5, TextRank10 = 0, 0, 0
    ImageRank1, ImageRank5, ImageRank10 = 0, 0, 0
    text_sim = torch.tensor(text_sim.to('cpu'))
    for i in range(1000):
        if i % 100 == 0: print(i)
        res_list = sorted(text_sim[i,], reverse=True)
        rank = res_list.index(text_sim[i][i])
        if rank < 1:
            TextRank1 += 1
        if rank < 5:
            TextRank5 += 1
        if rank < 10:
            TextRank10 += 1
    # print("text completed")

    img_sim = img_sim.to('cpu')
    for i in range(1000):
        if i % 100 == 0: print(i)
        res_list = sorted(img_sim[i,], reverse=True)
        rank = res_list.index(img_sim[i][i])
        if rank < 1:
            ImageRank1 += 1
        if rank < 5:
            ImageRank5 += 1
        if rank < 10:
            ImageRank10 += 1
    end = time.time()
    print("Consuming {:.2f} seconds".format(end - start))
    print("TextRank1:{}, TextRank5:{}, TextRank10:{}".format(TextRank1/1000, TextRank5/1000, TextRank10/1000))
    print("ImageRank1:{}, ImageRank5:{}, ImageRank10:{}".format(ImageRank1/1000, ImageRank5/1000, ImageRank10/1000))
    return [TextRank1/1000, TextRank5/1000, TextRank10/1000, ImageRank1/1000, ImageRank5/1000, ImageRank10/1000]

def eval_coco_batch(clip_model, batch):

    clip_model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    start = time.time()

    id, img, text = batch
    num = len(id)
    idx2id = dict()
    for i in range(len(id)):
        idx2id[i] = id[i]
    img = img.to(device)
    text = text.squeeze(1).to(device)
    logit_img2text, text_embedding, img_embedding = clip_model(img, text)
    print("loading success")
    text_embedding = text_embedding.to('cpu')
    img_embedding = img_embedding.to('cpu')
    text_sim = text_embedding @ img_embedding.T
    img_sim = img_embedding @ text_embedding.T

    TextRank1, TextRank5, TextRank10 = 0, 0, 0
    ImageRank1, ImageRank5, ImageRank10 = 0, 0, 0
    text_sim = torch.tensor(text_sim.to('cpu'))


    img_sim = img_sim.to('cpu')
    img_set = set()
    img_unq = []
    img_mask = []
    for i in range(num):
        if i % 1000 == 0: print(i)
        cur_id = idx2id[i]
        if cur_id in img_set:
            img_mask.append(0)
            continue
        img_set.add(cur_id)
        img_unq.append(i)
        img_mask.append(1)
        _, sorted_idx = torch.sort(img_sim[i,], descending=True)
        sorted_idx = sorted_idx.numpy().tolist()
        flag1, flag5, flag10 = 1, 1, 1
        for j in range(10):
            if j < 1 and idx2id[sorted_idx[j]] == cur_id and flag1:
                ImageRank1 += 1
                flag1 = 0
            if j < 5 and idx2id[sorted_idx[j]] == cur_id and flag5:
                ImageRank5 += 1
                flag5 = 0
            if j < 10 and idx2id[sorted_idx[j]] == cur_id and flag10:
                ImageRank10 += 1
                flag10 = 0
    img_num = len(img_set)


    for i in range(len(idx2id)):
        if i % 1000 == 0: print(i)
        cur_id = idx2id[i]
        sorted_score, sorted_idx = torch.sort(text_sim[i, ].mul(torch.tensor(img_mask)), descending=True)
        sorted_idx = sorted_idx.numpy().tolist()
        flag1, flag5, flag10 = 1, 1, 1
        for j in range(10):
            if j < 1 and idx2id[sorted_idx[j]] == cur_id and flag1:
                TextRank1 += 1
                flag1 = 0
            if j < 5 and idx2id[sorted_idx[j]] == cur_id and flag5:
                TextRank5 += 1
                flag5 = 0
            if idx2id[sorted_idx[j]] == cur_id and flag10:
                TextRank10 += 1
                flag10 = 0

    # print("text completed")

    end = time.time()
    print("-----------------------------------eval_batch-------------------------------------------")
    print("Consuming {:.2f} seconds".format(end - start))
    print("TextRank1:{:.4f}, TextRank5:{:.4f}, TextRank10:{:.4f}".format(TextRank1/num, TextRank5/num, TextRank10/num))
    print("ImageRank1:{:.4f}, ImageRank5:{:.4f}, ImageRank10:{:.4f}".format(ImageRank1/img_num, ImageRank5/img_num, ImageRank10/img_num))
    print("-----------------------------------eval_batch-------------------------------------------")
    return [TextRank1/num, TextRank5/num, TextRank10/num, ImageRank1/img_num, ImageRank5/img_num, ImageRank10/img_num]

def eval_coco_large(clip_model, myTransformer, dataloader, idx2id, args):
    num = len(idx2id)
    clip_model.eval()
    myTransformer.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    start = time.time()
    text_embedding, img_embedding = torch.tensor([]), torch.tensor([])
    print('loading data')
    for i, batch in enumerate(dataloader):
        img, text, head_inputs, relation_inputs, tail_inputs, token_type_ids, attention_mask = batch

        
        img = img.cuda()
        text = text.squeeze(1).cuda()
        token_type_ids = token_type_ids.cuda()
        attention_mask = attention_mask.cuda()

        with torch.no_grad():
            text = clip_model.encode_text(text)
            text = text / text.norm(dim=1, keepdim=True)
            knowledge_emb = myTransformer(head_inputs, relation_inputs, tail_inputs, token_type_ids, attention_mask)
            knowledge_emb = knowledge_emb / knowledge_emb.norm(dim=1, keepdim=True)
            text = text + knowledge_emb * args.knowledge_weight

            img = clip_model.encode_image(img)
            img = img / img.norm(dim=1, keepdim=True)

        if i == 0:
            text_embedding, img_embedding = text, img
            continue
        text_embedding = torch.cat((text_embedding, text), 0)
        img_embedding = torch.cat((img_embedding, img), 0)
    print("loading success")
    text_embedding = text_embedding.to('cpu')
    img_embedding = img_embedding.to('cpu')
    text_sim = text_embedding @ img_embedding.T
    img_sim = img_embedding @ text_embedding.T

    TextRank1, TextRank5, TextRank10 = 0, 0, 0
    ImageRank1, ImageRank5, ImageRank10 = 0, 0, 0
    text_sim = torch.tensor(text_sim.to('cpu'))


    img_sim = img_sim.to('cpu')
    img_set = set()
    img_unq = []
    img_mask = []
    for i in tqdm(range(num)):
        # if i % 1000 == 0: print(i)
        cur_id = idx2id[i]
        if cur_id in img_set:
            img_mask.append(0)
            continue
        img_set.add(cur_id)
        img_unq.append(i)
        img_mask.append(1)
        _, sorted_idx = torch.sort(img_sim[i,], descending=True)
        sorted_idx = sorted_idx.numpy().tolist()
        flag1, flag5, flag10 = 1, 1, 1
        for j in range(10):
            if j < 1 and idx2id[sorted_idx[j]] == cur_id and flag1:
                ImageRank1 += 1
                flag1 = 0
            if j < 5 and idx2id[sorted_idx[j]] == cur_id and flag5:
                ImageRank5 += 1
                flag5 = 0
            if j < 10 and idx2id[sorted_idx[j]] == cur_id and flag10:
                ImageRank10 += 1
                flag10 = 0
    img_num = len(img_set)


    for i in tqdm(range(len(idx2id))):
        # if i % 1000 == 0: print(i)
        cur_id = idx2id[i]
        sorted_score, sorted_idx = torch.sort(text_sim[i, ].mul(torch.tensor(img_mask)), descending=True)
        sorted_idx = sorted_idx.numpy().tolist()
        flag1, flag5, flag10 = 1, 1, 1
        for j in range(10):
            if j < 1 and idx2id[sorted_idx[j]] == cur_id and flag1:
                TextRank1 += 1
                flag1 = 0
            if j < 5 and idx2id[sorted_idx[j]] == cur_id and flag5:
                TextRank5 += 1
                flag5 = 0
            if idx2id[sorted_idx[j]] == cur_id and flag10:
                TextRank10 += 1
                flag10 = 0

    # print("text completed")

    end = time.time()
    print("Consuming {:.2f} seconds".format(end - start))
    print("TextRank1:{:.4f}, TextRank5:{:.4f}, TextRank10:{:.4f}".format(TextRank1/num, TextRank5/num, TextRank10/num))
    print("ImageRank1:{:.4f}, ImageRank5:{:.4f}, ImageRank10:{:.4f}".format(ImageRank1/img_num, ImageRank5/img_num, ImageRank10/img_num))
    return [TextRank1/num, TextRank5/num, TextRank10/num, ImageRank1/img_num, ImageRank5/img_num, ImageRank10/img_num]

@torch.no_grad()
def get_retrieval_scores_batched(clip_model, myTransformer, joint_loader, relation, args):
    clip_model.eval()
    myTransformer.eval()

    scores = []
    for batch in tqdm(joint_loader):
        image_options = []
        for i_option in batch["image_options"]:
            image_embeddings = clip_model.encode_image(i_option.cuda()).cpu().numpy() # B x D
            image_embeddings = image_embeddings / np.linalg.norm(image_embeddings, axis=1, keepdims=True) # B x D
            image_options.append(np.expand_dims(image_embeddings, axis=1))
        
        caption_options = []
        for index, c_option in enumerate(batch["caption_options"]):
            caption_tokenized = torch.cat([clip.tokenize(c) for c in c_option])
            caption_embeddings = clip_model.encode_text(caption_tokenized.cuda()).cpu().numpy() # B x D
            caption_embeddings = caption_embeddings / np.linalg.norm(caption_embeddings, axis=1, keepdims=True) # B x D
            
            # knowledge
            if index == 1:
                head_inputs = batch["head_inputs"]
                relation_inputs = batch["relation_inputs"]
                tail_inputs = batch["tail_inputs"]
                attention_mask = batch["attention_mask"].cuda()
            elif index == 0:
                head_inputs = batch["reversed_head_inputs"]
                relation_inputs = batch["reversed_relation_inputs"]
                tail_inputs = batch["reversed_tail_inputs"]
                attention_mask = batch["reversed_attention_mask"].cuda()
                
            knowledge_emb = myTransformer(head_inputs, relation_inputs, tail_inputs, 0, attention_mask)
            knowledge_emb = knowledge_emb / knowledge_emb.norm(dim=1, keepdim=True)    
            knowledge_emb = knowledge_emb.cpu().numpy()
                   
            caption_embeddings = caption_embeddings + knowledge_emb * args.knowledge_weight

            caption_options.append(np.expand_dims(caption_embeddings, axis=1))
            
        image_options = np.concatenate(image_options, axis=1) # B x K x D
        caption_options = np.concatenate(caption_options, axis=1) # B x L x D
        batch_scores = np.einsum("nkd,nld->nkl", image_options, caption_options) # B x K x L
        scores.append(batch_scores)
    
    all_scores = np.concatenate(scores, axis=0) # N x K x L
    return all_scores

drop_relations = ['adjusting',
 'attached to',
 'between',
 'bigger than',
 'biting',
 'boarding',
 'brushing',
 'chewing',
 'cleaning',
 'climbing',
 'close to',
 'coming from',
 'coming out of',
 'contain',
 'crossing',
 'dragging',
 'draped over',
 'drinking',
 'drinking from',
 'driving',
 'driving down',
 'driving on',
 'eating from',
 'eating in',
 'enclosing',
 'exiting',
 'facing',
 'filled with',
 'floating in',
 'floating on',
 'flying',
 'flying above',
 'flying in',
 'flying over',
 'flying through',
 'full of',
 'going down',
 'going into',
 'going through',
 'grazing in',
 'growing in',
 'growing on',
 'guiding',
 'hanging from',
 'hanging in',
 'hanging off',
 'hanging over',
 'higher than',
 'holding onto',
 'hugging',
 'in between',
 'jumping off',
 'jumping on',
 'jumping over',
 'kept in',
 'larger than',
 'leading',
 'leaning over',
 'leaving',
 'licking',
 'longer than',
 'looking in',
 'looking into',
 'looking out',
 'looking over',
 'looking through',
 'lying next to',
 'lying on top of',
 'making',
 'mixed with',
 'mounted on',
 'moving',
 'on the back of',
 'on the edge of',
 'on the front of',
 'on the other side of',
 'opening',
 'painted on',
 'parked at',
 'parked beside',
 'parked by',
 'parked in',
 'parked in front of',
 'parked near',
 'parked next to',
 'perched on',
 'petting',
 'piled on',
 'playing',
 'playing in',
 'playing on',
 'playing with',
 'pouring',
 'reaching for',
 'reading',
 'reflected on',
 'riding on',
 'running in',
 'running on',
 'running through',
 'seen through',
 'sitting behind',
 'sitting beside',
 'sitting by',
 'sitting in front of',
 'sitting near',
 'sitting next to',
 'sitting under',
 'skiing down',
 'skiing on',
 'sleeping in',
 'sleeping on',
 'smiling at',
 'sniffing',
 'splashing',
 'sprinkled on',
 'stacked on',
 'standing against',
 'standing around',
 'standing behind',
 'standing beside',
 'standing in front of',
 'standing near',
 'standing next to',
 'staring at',
 'stuck in',
 'surrounding',
 'swimming in',
 'swinging',
 'talking to',
 'topped with',
 'touching',
 'traveling down',
 'traveling on',
 'tying',
 'typing on',
 'underneath',
 'wading in',
 'waiting for',
 'walking across',
 'walking by',
 'walking down',
 'walking next to',
 'walking through',
 'working in',
 'working on',
 'worn on',
 'wrapped around',
 'wrapped in',    
 "by", 
 "of", 
 "near", "next to", 
 "with",
 "beside",
 "on the side of",
 "around"]

def macroacc_evaluation(scores, dataset, drop_relations=drop_relations):
    
    metrics = {"Accuracy": None}
    preds = np.argmax(np.squeeze(scores, axis=1), axis=-1)
    correct_mask = (preds == 1)
    metrics["Accuracy"] = np.mean(correct_mask)
    
    all_relations = np.array(dataset.all_relations)
    # Log the accuracy of all relations
    for relation in np.unique(all_relations):
        if relation in drop_relations:
            continue
        relation_mask = (all_relations == relation)
        if relation_mask.sum() == 0:
            continue
        metrics[f"{relation}-Acc"] = correct_mask[relation_mask].mean()
    
    return metrics

def macroacc_evaluation_attribute(scores, dataset):
    
    metrics = {"Accuracy": None}
    preds = np.argmax(np.squeeze(scores, axis=1), axis=-1)
    correct_mask = (preds == 1)
    metrics["Accuracy"] = np.mean(correct_mask)
    
    all_relations = np.array(dataset.all_attributes)
    # Log the accuracy of all relations
    for relation in np.unique(all_relations):
        relation_mask = (all_relations == relation)
        if relation_mask.sum() == 0:
            continue
        metrics[f"{relation}-Acc"] = correct_mask[relation_mask].mean()
    
    return metrics

def test_vg_relation(clip_model, myTransformer, vg_relation_dataloader, vg_relation_dataset, args):
    scores = get_retrieval_scores_batched(clip_model, myTransformer, vg_relation_dataloader, "rel", args)
    # np.save('/root/code/clip_order/checkpoints/case_study/relation/our_score.npy',scores)
    metrics = macroacc_evaluation(scores, vg_relation_dataset)
    all_accs = []
    for k,v in metrics.items():
        if "-Acc" in k:
            all_accs.append(v)
    acc_test_relation = np.mean(all_accs)
    print("acc_test_relation", acc_test_relation)
    return acc_test_relation


def test_vg_attribution(clip_model, myTransformer, vg_attribution_dataloader, vg_attribution_dataset, args):
    scores = get_retrieval_scores_batched(clip_model, myTransformer, vg_attribution_dataloader, "attribute", args)
    
    metrics = macroacc_evaluation_attribute(scores, vg_attribution_dataset)
    all_accs = []
    for k,v in metrics.items():
        if "-Acc" in k:
            all_accs.append(v)
    acc_test_attribution = np.mean(all_accs)
    print("acc_test_attribution", acc_test_attribution)
    return acc_test_attribution
