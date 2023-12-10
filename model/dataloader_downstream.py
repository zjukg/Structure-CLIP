from torch.utils.data import Dataset, DataLoader
from clip import tokenize
from PIL import Image
import json
from transformers import BertTokenizer
import torch
import os
from easydict import EasyDict as edict


class VG_Relation(Dataset):
    def __init__(self, transform = None):
        self.transform = transform
        self.length = 5
        self.padding_num = 6
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.root_dir = "/root/data/visual_genome_data/vg_image/"
        
        with open("data/visual_genome_relation_aug.json", "r") as f:
            self.dataset = json.load(f)
        self.all_relations = list()
        for item in self.dataset:
            item["image_path"] = os.path.join(self.root_dir, item["image_path"])
            self.all_relations.append(item["relation_name"])
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):    
        rel = self.dataset[index]
        # step1 
        image = Image.open(rel["image_path"]).convert('RGB')    
        # Get the bounding box that contains the relation
        image = image.crop((rel["bbox_x"], rel["bbox_y"], rel["bbox_x"]+rel["bbox_w"], rel["bbox_y"]+rel["bbox_h"]))
        
        if self.transform is not None:
            image = self.transform(image)
        
        # step2 
        caption = rel["true_caption"]
        reversed_caption = rel["false_caption"]   

        # step3 
        # 3.1 
        triples = rel['true_triples']
        head_inputs, relation_inputs, tail_inputs, attention_mask = self.get_triple_token_infor(triples)

        # 3.2 
        reversed_triples = rel['false_triples']
        reversed_head_inputs, reversed_relation_inputs, reversed_tail_inputs, reversed_attention_mask = self.get_triple_token_infor(reversed_triples)

        item = edict({"image_options": [image], 
                      "caption_options": [reversed_caption, caption], 
                      "relation": rel["relation_name"],
                      "head_inputs": head_inputs,
                      "relation_inputs": relation_inputs,
                      "tail_inputs": tail_inputs,
                      "attention_mask": attention_mask,
                      "reversed_head_inputs": reversed_head_inputs,
                      "reversed_relation_inputs": reversed_relation_inputs,
                      "reversed_tail_inputs": reversed_tail_inputs,
                      "reversed_attention_mask": reversed_attention_mask})
        return item

    def get_triple_token_infor(self, triples):
        head_word_list = list()
        relation_word_list = list()
        tail_word_list = list()

        if len(triples) > 0:
            for triple in triples:
                if len(head_word_list) < self.padding_num:
                    head_word_list.append(triple[0])
                    relation_word_list.append(triple[1])
                    tail_word_list.append(triple[2])
                

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
        return head_inputs, relation_inputs, tail_inputs, attention_mask


class VG_Attribution(Dataset):
    def __init__(self, data_path=None, transform=None):

        self.transform = transform
        self.length = 5
        self.padding_num = 6
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        if 'adjchange' in data_path:
            self.root_dir = '/root/data/coco_data/'
        else:
            self.root_dir = "/root/data/visual_genome_data/vg_image/"
        self.data_path = data_path
        
        with open(self.data_path, "r") as f:
            self.dataset = json.load(f)
        for item in self.dataset:
            item["image_path"] = os.path.join(self.root_dir, item["image_path"])
        if 'adjchange' not in data_path:
            self.all_attributes = [f"{item['attributes'][0]}_{item['attributes'][1]}" for item in self.dataset]

    def __len__(self):
        return len(self.dataset)
    
    def get_triple_token_infor(self, triples):

        head_word_list = list()
        relation_word_list = list()
        tail_word_list = list()

        if len(triples) > 0:
            for triple in triples:
                if len(head_word_list) < self.padding_num:
                    head_word_list.append(triple[0])
                    relation_word_list.append(triple[1])
                    tail_word_list.append(triple[2])
                
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
        return head_inputs, relation_inputs, tail_inputs, attention_mask

    def __getitem__(self, index):
        scene = self.dataset[index]
        # step1 
        image = Image.open(scene["image_path"]).convert('RGB')
        # Get the bounding box that contains the relation
        if self.root_dir != '/root/data/coco_data/':
            image = image.crop((scene["bbox_x"], scene["bbox_y"], scene["bbox_x"] + scene["bbox_w"], scene["bbox_y"] + scene["bbox_h"]))

        if self.transform is not None:
            image = self.transform(image)

        # step2 
        true_caption = scene["true_caption"]
        false_caption = scene["false_caption"]
        if "train" in self.data_path:
            true_caption = tokenize(true_caption,  truncate=True)
            false_caption = tokenize(false_caption,  truncate=True)

        # step3 
        triples = scene['true_triples']
        head_inputs, relation_inputs, tail_inputs, attention_mask = self.get_triple_token_infor(triples)

        reversed_triples = scene['false_triples']
        reversed_head_inputs, reversed_relation_inputs, reversed_tail_inputs, reversed_attention_mask = self.get_triple_token_infor(reversed_triples)

        item = edict({"image_options": [image], 
                      "caption_options": [false_caption, true_caption], 
                      "relation": "attribution",
                      "head_inputs": head_inputs,
                      "relation_inputs": relation_inputs,
                      "tail_inputs": tail_inputs,
                      "attention_mask": attention_mask,
                      "reversed_head_inputs": reversed_head_inputs,
                      "reversed_relation_inputs": reversed_relation_inputs,
                      "reversed_tail_inputs": reversed_tail_inputs,
                      "reversed_attention_mask": reversed_attention_mask})

        return item
    
    