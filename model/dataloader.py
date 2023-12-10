from torch.utils.data import Dataset, DataLoader
from clip import tokenize
from PIL import Image
import json
from transformers import BertTokenizer
import torch
import os
from easydict import EasyDict as edict

class Mydataset(Dataset):
    def __init__(self, data_path, transform = None):
        self.transform = transform
        self.img_text = []
        f = open(data_path, 'r')
        for line in f:
            row = line.strip().split('\t')
            self.img_text.append((row[1], row[2], row[3]))
        f.close()
    def __getitem__(self, item):
        img_path, true_text, false_text  = self.img_text[item]
        # img = Image.open('img_path').convert('RGB')
        img = Image.open('/root/data/coco_data/' + img_path)
        if self.transform is not None:
            img = self.transform(img)
        true_text = tokenize(true_text)
        false_text = tokenize(false_text)

        return img, true_text, false_text
    def __len__(self):
        return len(self.img_text)

class CoCoDataset(Dataset):
    def __init__(self, data_path, transform = None):
        self.transform = transform
        self.img_text = []
        f = open(data_path, 'r')
        for line in f:
            row = line.strip().split('\t')
            self.img_text.append((row[0], row[1], row[2]))
        f.close()
    def __getitem__(self, item):
        id, img_path, text = self.img_text[item]
        img = Image.open('/root/data/coco_data/' + img_path)
        if self.transform is not None:
            img = self.transform(img)
        text = tokenize(text)

        return int(id), img, text
    def __len__(self):
        return len(self.img_text)



class CoCoDataset_aug_object_attribute(Dataset):
    def __init__(self, data_path, transform = None):
        self.transform = transform
        self.img_text = []
        self.length = 150
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        with open(data_path, "r") as f:
            infomation = json.load(f)
        for item in infomation:
            self.img_text.append((item['image'], item['text'], item['triples'], item['pairs']))

    def __getitem__(self, item):
        img_path, text, triples, pairs = self.img_text[item]
        img = Image.open('/root/data/coco_data/' + img_path)
        if self.transform is not None:
            img = self.transform(img)
        
        text_clip = tokenize(text,  truncate=True)
        
        extend_text = " "
        if len(triples) > 0:
            for triple in triples:
                extend_text = extend_text + triple[0] + "," + triple[1] + "," + triple[2] + "."
        
        if len(pairs) > 0:
            for pair in pairs:
                extend_text = extend_text + pair[0] + "," + pair[1] + "."

        
        text_inputs = self.tokenizer.encode_plus(
            extend_text,
            max_length=self.length,
            add_special_tokens=True,
            padding = "max_length",
            return_tensors="pt",
            truncation=True
        )

        return img, text_clip, text_inputs

    def __len__(self):
        return len(self.img_text)


class CoCoDataset_aug(Dataset):
    def __init__(self, data_path, transform = None):
        self.transform = transform
        self.img_text = []
        self.length = 20
        self.padding_num = 6

        with open(data_path, "r") as f:
            infomation = json.load(f)

        for item in infomation:
            self.img_text.append((item['image'], item['text'], item['triples'], item['pairs']))


    def __getitem__(self, item):
        img_path, text, triples, pairs = self.img_text[item]
        img = Image.open('/root/data/coco_data/' + img_path)
        if self.transform is not None:
            img = self.transform(img)
        

        text = tokenize(text,  truncate=True)


        head_word_list = list()
        relation_word_list = list()
        tail_word_list = list()
        if len(triples) > 0:
            for triple in triples:
                head_word_list.append(triple[0])
                relation_word_list.append(triple[1])
                tail_word_list.append(triple[2])
        if len(pairs) > 0:
            for pair in pairs:
                head_word_list.append(pair[0])
                relation_word_list.append("is")
                tail_word_list.append(pair[1])

        head_word_list = tokenize(head_word_list, context_length=self.length) 
        relation_word_list = tokenize(relation_word_list, context_length=self.length) 
        tail_word_list = tokenize(tail_word_list, context_length=self.length) 

        if head_word_list.shape[0] > self.padding_num:
            head_word_list = head_word_list[:self.padding_num,]
            relation_word_list = relation_word_list[:self.padding_num,]
            tail_word_list = tail_word_list[:self.padding_num,]
        else:
            extend_list = torch.zeros([self.padding_num - head_word_list.shape[0],self.length],dtype=torch.int32)
            head_word_list = torch.cat((head_word_list, extend_list), dim=0)
            relation_word_list = torch.cat((relation_word_list, extend_list), dim=0)
            tail_word_list = torch.cat((tail_word_list, extend_list), dim=0)
        return img, text, head_word_list, relation_word_list, tail_word_list


    def __len__(self):
        return len(self.img_text)

class CoCoDataset_aug_update(Dataset):
    def __init__(self, data_path, transform = None):
        self.transform = transform
        self.length = 5
        self.padding_num = 6
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        with open(data_path, "r") as f:
            infomation = json.load(f)

        self.img_text = []
        for item in infomation:
            self.img_text.append((item['image'], item['text'], item['triples'], item['pairs']))
            

    def __getitem__(self, item):
        img_path, text, triples, pairs = self.img_text[item]

        img = Image.open('/root/data/coco_data/' + img_path)
        if self.transform is not None:
            img = self.transform(img)

        text = tokenize(text,  truncate=True)

        head_word_list = list()
        relation_word_list = list()
        tail_word_list = list()

        if len(triples) > 0:
            for triple in triples:
                if len(head_word_list) < self.padding_num:
                    head_word_list.append(triple[0])
                    relation_word_list.append(triple[1])
                    tail_word_list.append(triple[2])
        if len(pairs) > 0:
            for pair in pairs:
                if len(head_word_list) < self.padding_num:
                    head_word_list.append(pair[1])
                    relation_word_list.append("is")
                    tail_word_list.append(pair[0])
                

        token_type_ids = torch.zeros([1,self.padding_num], dtype=int)

        if len(head_word_list) > 0:
            attention_mask = torch.cat((torch.ones([1, len(head_word_list)], dtype=int), torch.zeros([1, self.padding_num-len(head_word_list)], dtype=int)), dim=1)
        else:
            attention_mask = torch.zeros([1, self.padding_num], dtype=int)
        

        for i in range(self.padding_num - len(head_word_list)):
            head_word_list.append('')
            relation_word_list.append('')
            tail_word_list.append('')
        
 
        head_inputs = self.tokenizer.batch_encode_plus(
            head_word_list,
            max_length=self.length,
            add_special_tokens=True,
            padding = "max_length",
            return_tensors="pt",
            truncation=True
        )
        relation_inputs = self.tokenizer.batch_encode_plus(
            relation_word_list,
            max_length=self.length,
            add_special_tokens=True,
            padding = "max_length",
            return_tensors="pt",
            truncation=True
        )
        tail_inputs = self.tokenizer.batch_encode_plus(
            tail_word_list,
            max_length=self.length,
            add_special_tokens=True,
            padding = "max_length",
            return_tensors="pt",
            truncation=True
        )

        return img, text, head_inputs, relation_inputs, tail_inputs, token_type_ids, attention_mask

    def __len__(self):
        return len(self.img_text)
        

class CoCoDataset_aug_update_withneg(Dataset):
    def __init__(self, data_path, transform = None):
        self.transform = transform
        self.length = 5
        self.padding_num = 6
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        with open(data_path, "r") as f:
            infomation = json.load(f)

        self.img_text = []
        for item in infomation:
            self.img_text.append((item['image'], item['text'], item['triples'], item['pairs'], item['neg_text']))
            

    def __getitem__(self, item):
        img_path, text, triples, pairs, neg_text = self.img_text[item]
 
        img = Image.open('/root/data/coco_data/' + img_path)
        if self.transform is not None:
            img = self.transform(img)

        text = tokenize(text, truncate=True)

        neg_text = tokenize(neg_text, truncate=True)


        head_word_list = list()
        relation_word_list = list()
        tail_word_list = list()

        if len(triples) > 0:
            for triple in triples:
                if len(head_word_list) < self.padding_num:
                    head_word_list.append(triple[0])
                    relation_word_list.append(triple[1])
                    tail_word_list.append(triple[2])
        if len(pairs) > 0:
            for pair in pairs:
                if len(head_word_list) < self.padding_num:
                    head_word_list.append(pair[1])
                    relation_word_list.append("is")
                    tail_word_list.append(pair[0])
                

        token_type_ids = torch.zeros([1,self.padding_num], dtype=int)

        if len(head_word_list) > 0:
            attention_mask = torch.cat((torch.ones([1, len(head_word_list)], dtype=int), torch.zeros([1, self.padding_num-len(head_word_list)], dtype=int)), dim=1)
        else:
            attention_mask = torch.zeros([1, self.padding_num], dtype=int)
        

        for i in range(self.padding_num - len(head_word_list)):
            head_word_list.append('')
            relation_word_list.append('')
            tail_word_list.append('')
        

        head_inputs = self.tokenizer.batch_encode_plus(
            head_word_list,
            max_length=self.length,
            add_special_tokens=True,
            padding = "max_length",
            return_tensors="pt",
            truncation=True
        )
        relation_inputs = self.tokenizer.batch_encode_plus(
            relation_word_list,
            max_length=self.length,
            add_special_tokens=True,
            padding = "max_length",
            return_tensors="pt",
            truncation=True
        )
        tail_inputs = self.tokenizer.batch_encode_plus(
            tail_word_list,
            max_length=self.length,
            add_special_tokens=True,
            padding = "max_length",
            return_tensors="pt",
            truncation=True
        )

        return img, text, head_inputs, relation_inputs, tail_inputs, token_type_ids, attention_mask, neg_text

    def __len__(self):
        return len(self.img_text)


class VG_Relation_Test(Dataset):
    def __init__(self, image_preprocess):
        self.length = 20

        self.padding_num = 1 
        self.root_dir = "/root/data/visual_genome_data/vg_image/"
        with open("data/visual_genome_relation_aug.json", "r") as f:
            self.dataset = json.load(f)
        self.all_relations = list()
        for item in self.dataset:
            item["image_path"] = os.path.join(self.root_dir, item["image_path"])
            self.all_relations.append(item["relation_name"])
        self.image_preprocess = image_preprocess
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):    
        rel = self.dataset[index]
        image = Image.open(rel["image_path"]).convert('RGB')    
        # Get the bounding box that contains the relation
        image = image.crop((rel["bbox_x"], rel["bbox_y"], rel["bbox_x"]+rel["bbox_w"], rel["bbox_y"]+rel["bbox_h"]))
        
        if self.image_preprocess is not None:
            image = self.image_preprocess(image)  
            
        caption = rel["true_caption"]
        reversed_caption = rel["false_caption"]   

        # 处理三元组部分
        head_word_list = list()
        relation_word_list = list()
        tail_word_list = list()
        for triple in rel['false_triples']:
            head_word_list.append(triple[0])
            relation_word_list.append(triple[1])
            tail_word_list.append(triple[2])
        for triple in rel['true_triples']:
            head_word_list.append(triple[0])
            relation_word_list.append(triple[1])
            tail_word_list.append(triple[2])

        head_word_list = tokenize(head_word_list, context_length=self.length) 
        relation_word_list = tokenize(relation_word_list, context_length=self.length) 
        tail_word_list = tokenize(tail_word_list, context_length=self.length) 

        item = edict({"image_options": [image], 
                      "caption_options": [reversed_caption, caption], 
                      "relation": rel["relation_name"],
                      "head_word_list": head_word_list,
                      "relation_word_list": relation_word_list,
                      "tail_word_list": tail_word_list})
        return item


class VG_Attribution_Test(Dataset):
    def __init__(self, image_preprocess):
        self.length = 20

        self.padding_num = 12
        self.root_dir = "/root/data/visual_genome_data/vg_image/"
        with open("data/visual_genome_attribution_aug.json", "r") as f:
            self.dataset = json.load(f)
        for item in self.dataset:
            item["image_path"] = os.path.join(self.root_dir, item["image_path"])
        self.all_attributes = [f"{item['attributes'][0]}_{item['attributes'][1]}" for item in self.dataset]
        self.image_preprocess = image_preprocess


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        scene = self.dataset[index]
        image = Image.open(scene["image_path"]).convert('RGB')
        # Get the bounding box that contains the relation
        image = image.crop((scene["bbox_x"], scene["bbox_y"], scene["bbox_x"] + scene["bbox_w"], scene["bbox_y"] + scene["bbox_h"]))

        if self.image_preprocess is not None:
            image = self.image_preprocess(image)

        true_caption = scene["true_caption"]
        false_caption = scene["false_caption"]


        head_word_list = list()
        relation_word_list = list()
        tail_word_list = list()
        for triple in scene['false_triples']:
            head_word_list.append(triple[0])
            relation_word_list.append(triple[1])
            tail_word_list.append(triple[2])
        for triple in scene['true_triples']:
            head_word_list.append(triple[0])
            relation_word_list.append(triple[1])
            tail_word_list.append(triple[2])

        head_word_list = tokenize(head_word_list, context_length=self.length) 
        relation_word_list = tokenize(relation_word_list, context_length=self.length) 
        tail_word_list = tokenize(tail_word_list, context_length=self.length) 

        item = edict({"image_options": [image], 
                "caption_options": [false_caption, true_caption], 
                "relation": "attribution",
                "head_word_list": head_word_list,
                "relation_word_list": relation_word_list,
                "tail_word_list": tail_word_list})
        return item