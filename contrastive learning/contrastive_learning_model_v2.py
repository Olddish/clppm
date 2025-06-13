import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, AutoConfig
import pandas as pd
import numpy as np
import datetime
import ast
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os

seed = 2023
np.random.seed(2023)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def data_load(path,sheet):
    data = pd.read_excel(path,sheet_name=sheet)
    return data

class EncoderDataset(Dataset):
    def __init__(self, process_texts, micro_texts, fea_texts, alloy_poss):
        self.process_texts = process_texts
        self.micro_texts = micro_texts
        self.fea_texts = fea_texts
        self.alloy_poss = alloy_poss

    def __getitem__(self, index):
        process_text = self.process_texts[index]
        micro_text = self.micro_texts[index]
        fea_text = self.fea_texts[index]
        alloy_pos = ast.literal_eval(self.alloy_poss[index])
        return process_text, micro_text, fea_text, alloy_pos

    def __len__(self):
        return len(self.process_texts)
    
def pos_complete(pos_list):
    max_len = max([len(i) for i in pos_list])
    for i in pos_list:
        if len(i) < max_len:
            i.extend([-1]*(max_len-len(i)))
    return pos_list

def tokenize_input(texts):
    model_name = 'task1-v3/MatSciBERT'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    inputs = tokenizer.batch_encode_plus(texts,padding='max_length', max_length=512,
                                         truncation=True,
                                         return_tensors='pt',
                                         is_split_into_words=False)
    return inputs
    

def collate_fn(datasetx):

    process_texts = [i[0] for i in datasetx]
    micro_texts = [i[1] for i in datasetx]
    fea_texts = [i[2] for i in datasetx]
    alloy_poss = [i[3] for i in datasetx]

    complt_alloy_poss = pos_complete(alloy_poss)
    process_inputs = tokenize_input(process_texts)
    micro_inputs = tokenize_input(micro_texts)
    fea_inputs = tokenize_input(fea_texts)

    return process_inputs, micro_inputs, fea_inputs, torch.tensor(complt_alloy_poss)


def data_loader(dataset,batch_size,shuffle:bool):

    loader = DataLoader(dataset=dataset,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        collate_fn=collate_fn,
                        drop_last=False)
    return loader

def vec_pool(word_vec:torch.Tensor,pos:torch.Tensor):

    device = torch.device("cuda")

    pool = torch.nn.AdaptiveAvgPool1d(1).to(device)
    hidden_size = word_vec.size(-1)
    batch_vec = torch.zeros(pos.size(0),hidden_size)

    for batch_num in range(pos.size(0)):

        pos_maskt = pos[batch_num,:][pos[batch_num,:] != -1] 
        seq_vec = torch.zeros((pos_maskt.size(0),hidden_size))

        for seq_num in range(pos_maskt.size(0)):
        
            seq_vec[seq_num] = word_vec[batch_num,pos_maskt[seq_num]]
        # print(seq_vec.size())
        seq_vec_reshaped = seq_vec.transpose(0, 1).unsqueeze(0)
        pooled_vec = pool(seq_vec_reshaped)
        pooled_vec = pooled_vec.squeeze(0).transpose(0, 1)

        batch_vec[batch_num] = pooled_vec  

    return batch_vec.to(device)


class ProcessEncoder(PreTrainedModel):
    def __init__(self, config, model_name):
        super(ProcessEncoder, self).__init__(config)
        self.model = AutoModel.from_pretrained(model_name,config=config)

        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.encoder.layer[11].parameters():
            param.requires_grad = True

    def forward(self, process_inputs):
        process_vec = self.model(**process_inputs).last_hidden_state
        return process_vec

class PhysicalEncoder(PreTrainedModel):
    def __init__(self, config, model_name):
        super(PhysicalEncoder, self).__init__(config)
        self.PhyEncoder = AutoModel.from_pretrained(model_name,config=config)

        for param in self.PhyEncoder.parameters():
            param.requires_grad = False
        for param in self.PhyEncoder.encoder.layer[11].parameters():
            param.requires_grad = True

    def forward(self, physical_features):
        physical_features_vec = self.PhyEncoder(**physical_features).last_hidden_state
        return physical_features_vec
    
class MicrostructureEncoder(PreTrainedModel):
    def __init__(self, config, model_name):
        super(MicrostructureEncoder, self).__init__(config)
        self.MircoEncoder = AutoModel.from_pretrained(model_name,config=config)

        for param in self.MircoEncoder.parameters():
            param.requires_grad = False
        for param in self.MircoEncoder.encoder.layer[11].parameters():
            param.requires_grad = True

    def forward(self, mirco_inputs):
        mirco_vec = self.MircoEncoder(**mirco_inputs).last_hidden_state
        return mirco_vec
    
class EmbedingLayer(nn.Module):
    def __init__(self):
        super(EmbedingLayer, self).__init__()
        self.embed_phy_procss = nn.Linear(768*3,768)
        self.embed_mirco = nn.Linear(768*2,768)

    def forward(self, physical_features_vec, process_vec, mirco_vec, alloy_pos):
        physical_features_sentence_vec = physical_features_vec[ :, 0, :]
        poolt_alloy_vec_process = vec_pool(process_vec,alloy_pos)
        process_sentence_vec = process_vec[ :, 0, :]
        comnied_vec = torch.cat((poolt_alloy_vec_process, process_sentence_vec, physical_features_sentence_vec),dim=1)
        comnied_vec_embed = self.embed_phy_procss(comnied_vec)
        normalized_combied_vec = F.normalize(comnied_vec_embed, p=2, dim=1)

        poolt_alloy_vec_micro = vec_pool(mirco_vec,alloy_pos)
        micro_sentence_vec = mirco_vec[ :,0, :]
        micro_with_alloy_vec = torch.cat((poolt_alloy_vec_micro,micro_sentence_vec),dim=1)
        micro_vec_embed = self.embed_mirco(micro_with_alloy_vec)
        normalized_miroc_vec = F.normalize(micro_vec_embed, p=2, dim=1)
        return normalized_combied_vec, normalized_miroc_vec


class InfoNCELoss(nn.Module):
    def __init__(self, temperature):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, combined_vec, mirco_vec, labels):
        # 相似度计算
        logits = torch.matmul(combined_vec, mirco_vec.T) / self.temperature
        
        # 计算损失
        loss_i = F.cross_entropy(logits, labels, reduction='mean')
        loss_t = F.cross_entropy(logits.T, labels, reduction='mean')
        loss = (loss_i + loss_t) / 2
        return loss, loss_i, loss_t

def train_loop(phyEncoder, proEncoder, mircoEncoder, embedinglayer, cost_fuction, train_loader, val_loader, epoch_num, lr):
        
    device = torch.device("cuda")
    phyEncoder.to(device)
    proEncoder.to(device)
    mircoEncoder.to(device)
    embedinglayer.to(device)

    optimizer_phyEncoer = torch.optim.AdamW(phyEncoder.parameters(),lr=lr,weight_decay=1e-4)
    optimizer_proEncoder = torch.optim.AdamW(proEncoder.parameters(),lr=lr,weight_decay=1e-4)
    optimizer_mirco = torch.optim.AdamW(mircoEncoder.parameters(),lr=lr,weight_decay=1e-4)
    optimizer_embedinglayer = torch.optim.Adam(embedinglayer.parameters(),lr=lr,weight_decay=1e-4)
    scaler = GradScaler()

    batch_train_all_loss = []
    batch_train_loss_i = []
    batch_train_loss_t = []
    batch_val_loss = []

    time = datetime.datetime.now()
    writer1 = SummaryWriter("task1-v3/contras_learning/logs/encoder/"+str(time)+'/train')
    writer2 = SummaryWriter("task1-v3/contras_learning/logs/encoder/"+str(time)+'/val')

    for epoch in tqdm(range(epoch_num),ncols=80):
        phyEncoder.train()
        proEncoder.train()
        mircoEncoder.train()
        embedinglayer.train()
        for step, (train_process_inputs,train_micro_inputs,train_physical_feartures,train_alloy_poss) in enumerate(train_loader):
            
            train_physical_feartures = train_physical_feartures.to(device)
            train_process_inputs = train_process_inputs.to(device)
            train_micro_inputs = train_micro_inputs.to(device)

            train_alloy_poss = train_alloy_poss.to(device)
            train_bathsize = train_physical_feartures["input_ids"].shape[0]
            train_labels = torch.arange(train_bathsize).to(device)

            with autocast():
                train_phy_vec = phyEncoder(train_physical_feartures)
                train_process_vec = proEncoder(train_process_inputs)
                train_mirco_vec = mircoEncoder(train_micro_inputs)    
                train_embeded_combied_vec, train_embeded_mirco_vec = embedinglayer(train_phy_vec, train_process_vec, train_mirco_vec, train_alloy_poss)
                train_loss = cost_fuction(train_embeded_combied_vec, train_embeded_mirco_vec, train_labels)
            scaler.scale(train_loss[0]).backward()
            scaler.step(optimizer_phyEncoer)
            scaler.step(optimizer_proEncoder)
            scaler.step(optimizer_mirco)
            scaler.step(optimizer_embedinglayer)
            scaler.update()
            optimizer_phyEncoer.zero_grad()
            optimizer_proEncoder.zero_grad()
            optimizer_mirco.zero_grad()
            optimizer_embedinglayer.zero_grad()

            batch_train_all_loss.append(train_loss[0].item())
            batch_train_loss_i.append(train_loss[1].item())
            batch_train_loss_t.append(train_loss[2].item())

        writer1.add_scalars('epoch_train', {"all_loss":np.mean(batch_train_all_loss),
                            "loss_i":np.mean(batch_train_loss_i),"loss_t":np.mean(batch_train_loss_t)},epoch)
        batch_train_all_loss.clear()
        batch_train_loss_i.clear()
        batch_train_loss_t.clear()

        phyEncoder.float()
        proEncoder.float()
        mircoEncoder.float()
        embedinglayer.float()
        phyEncoder.eval()
        proEncoder.eval()
        mircoEncoder.eval()
        embedinglayer.eval()
        for step, (val_process_inputs, val_micro_inputs, val_physical_feartures, val_alloy_poss) in enumerate(val_loader):
            
            val_physical_feartures = val_physical_feartures.to(device)
            val_process_inputs = val_process_inputs.to(device)
            val_micro_inputs = val_micro_inputs.to(device)
            val_alloy_poss = val_alloy_poss.to(device)
            val_bathsize = val_physical_feartures["input_ids"].shape[0]
            val_labels = torch.arange(val_bathsize).to(device)

            with torch.no_grad():
                val_phy_vec = phyEncoder(val_physical_feartures)
                val_process_vec = proEncoder(val_process_inputs)
                val_mirco_vec = mircoEncoder(val_micro_inputs)
                val_embeded_combied_vec, val_embeded_mirco_vec = embedinglayer(val_phy_vec, val_process_vec, val_mirco_vec, val_alloy_poss)
            val_loss = cost_fuction(val_embeded_combied_vec, val_embeded_mirco_vec, val_labels)
            batch_val_loss.append(val_loss[0].item())
        writer2.add_scalars('epoch_val', {"loss":np.mean(batch_val_loss)}, epoch)
        batch_val_loss.clear()
    writer1.close()
    writer2.close()
    return phyEncoder, proEncoder, mircoEncoder, embedinglayer

if __name__ == "__main__":
    #数据加载
    path = 'task1-v3/datasets/Ti_alloy_datasets_v2.2.1.xlsx'
    sheet = 'Sheet1'
    data = data_load(path,sheet)
    #数据划分
    
    train_data,val_data = np.split(data.sample(frac=1), [int(.8*len(data))])
    train_data = train_data.reset_index(drop=True)
    val_data = val_data.reset_index(drop=True)
    #训练集数据加载
    train_alloy_poss = train_data['Alloy_pos']
    train_process_texts = train_data['process_descriptor']
    train_micro_texts = train_data['micro_descriptor']
    train_fea_texts = train_data['fea_descriptor']
    train_dataset = EncoderDataset(train_process_texts,train_micro_texts,train_fea_texts,train_alloy_poss)
    train_loader = data_loader(train_dataset, 200, True)
    #验证集数据加载
    val_alloy_poss = val_data['Alloy_pos']
    val_process_texts = val_data['process_descriptor']
    val_micro_texts = val_data['micro_descriptor']
    val_fea_texts = val_data['fea_descriptor']
    val_dataset = EncoderDataset(val_process_texts,val_micro_texts,val_fea_texts,val_alloy_poss)
    val_loader = data_loader(val_dataset, 128, False)
    #模型训练
    model_name = "task1-v3/MatSciBERT"
    config = AutoConfig.from_pretrained(model_name)
    phyEncoder = PhysicalEncoder(config,model_name)
    proEncoder = ProcessEncoder(config,model_name)
    mircoEncoder = MicrostructureEncoder(config,model_name)
    embedinglayer = EmbedingLayer()
    cost_fuction = InfoNCELoss(0.08)
    epoch = 115
    lr = 1e-3
    phyEn, proEn, mircoEn, _ = train_loop(phyEncoder, proEncoder, mircoEncoder, embedinglayer, cost_fuction, train_loader, val_loader, epoch, lr)
    #模型保存
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    phyEn = phyEn.float()
    proEncoder = proEn.float()
    mircoEncoder = mircoEn.float()
    # embedinglayer = embedinglayer.float()
    save_path = "task1-v3/contras_learning/encoder_v2/"
    PhyEncoder_save_path = save_path + "phy_encoder/"
    ProEncoder_save_path = save_path + "pro_encoder/"
    MircoEncoder_save_path = save_path + "mirco_encoder/"
    for path in [PhyEncoder_save_path, ProEncoder_save_path, MircoEncoder_save_path]:
        if not os.path.exists(path):
            os.makedirs(path)
    phyEn.PhyEncoder.save_pretrained(PhyEncoder_save_path)
    proEn.model.save_pretrained(ProEncoder_save_path)
    mircoEn.MircoEncoder.save_pretrained(MircoEncoder_save_path)
    # torch.save(embedinglayer.state_dict(),save_path+"embed/embedinglayer.pth")
    tokenizer.save_pretrained(save_path+"tokenizer/")