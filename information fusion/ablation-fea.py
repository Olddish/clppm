import pandas as pd
import ast
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import torch
import datetime
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
seed = 2023
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def data_load(path, sheet):
    data = pd.read_excel(path, sheet_name=sheet)
    return data

class PEDataset(Dataset):
    def __init__(self, process_texts, micro_texts, fea_texts, labels, alloy_poss):
        self.labels = labels
        self.process_texts = process_texts
        self.micro_texts = micro_texts
        self.fea_texts = fea_texts
        self.alloy_poss = alloy_poss

    def __getitem__(self, index):
        process_text = self.process_texts[index]
        micro_text = self.micro_texts[index]
        fea_text = self.fea_texts[index]
        label = self.labels[index]
        alloy_pos = ast.literal_eval(self.alloy_poss[index])
        return process_text, micro_text, fea_text, label, alloy_pos

    def __len__(self):
        return len(self.labels)
    
def pos_complete(pos_list):
    max_len = max([len(i) for i in pos_list])
    for i in pos_list:
        if len(i) < max_len:
            i.extend([-1]*(max_len-len(i)))
    return pos_list


def tokenize_input(texts):
    tokenizer = AutoTokenizer.from_pretrained('task1-v3/MatSciBERT')
    inputs = tokenizer.batch_encode_plus(texts,padding='max_length', max_length=512,
                                         truncation=True,
                                         return_tensors='pt',
                                         is_split_into_words=False)
    return inputs

def to_floats(lst):
    return [float(x) for x in lst]

def collate_fn(datasetx):

    process_texts = [i[0] for i in datasetx]
    micro_texts = [i[1] for i in datasetx]
    # fea_texts = [i[2] for i in datasetx]
    labels = [i[3] for i in datasetx]
    alloy_poss = [i[4] for i in datasetx]

    complt_alloy_poss = pos_complete(alloy_poss)
    process_inputs = tokenize_input(process_texts)
    micro_inputs = tokenize_input(micro_texts)
    # fea_inputs = tokenize_input(fea_texts)
    labels = to_floats(labels)

    return process_inputs, micro_inputs, torch.Tensor(labels), torch.tensor(complt_alloy_poss)


def data_loader(dataset,batch_size,shuffle:bool):

    loader = DataLoader(dataset=dataset,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        collate_fn=collate_fn,
                        drop_last=False)
    return loader

def r2_score(y_true, y_pred):
    ss_res = torch.sum((y_true - y_pred)**2)
    ss_tot = torch.sum((y_true - torch.mean(y_true))**2)
    r2 = 1 - ss_res / ss_tot
    return r2.item()

def mse_score(y_true, y_pred):
    mse = torch.mean((y_true - y_pred)**2)
    return mse.item()

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


class Model(torch.nn.Module):
    def __init__(self,model_name,tuning:bool):
        super(Model,self).__init__()
        
        self.fc1 = torch.nn.Linear(768*4, 768)
        self.dr1 = torch.nn.Dropout(0.5)
        self.fc2 = torch.nn.Linear(768,1)
        self.bn = torch.nn.BatchNorm1d(768)


        if tuning:
            self.pretrained = AutoModel.from_pretrained(model_name)
        else:
            with torch.no_grad():
                self.pretrained = AutoModel.from_pretrained(model_name)
            for param in self.pretrained.parameters():
                param.requires_grad = False


    def forward(self,process_inputs,mirco_inputs,alloy_pos):

        process_vec = self.pretrained(**process_inputs).last_hidden_state
        mirco_vec = self.pretrained(**mirco_inputs).last_hidden_state
        # fea_vec = self.pretrained(**fea_inputs).last_hidden_state

        process_sentence_vec = process_vec[ :,0, :]
        micro_sentence_vec = mirco_vec[ :,0, :]
        # fea_sentence_vec = fea_vec[ :,0, :]

        poolt_alloy_vec_prpcess = vec_pool(process_vec,alloy_pos)
        poolt_alloy_vec_micro = vec_pool(mirco_vec,alloy_pos)

        process_with_alloy = torch.cat((poolt_alloy_vec_prpcess,process_sentence_vec),dim=1)
        micro_with_alloy = torch.cat((poolt_alloy_vec_micro,micro_sentence_vec),dim=1)
        combined = torch.cat((process_with_alloy, micro_with_alloy), dim=1)
        normalized_features = F.normalize(combined, p=2, dim=1)
   
        out = self.fc1(normalized_features)
        out = self.bn(out)
        out = F.relu(out)
        out = self.dr1(out)
        out = self.fc2(out)
        return out
    
def train_loop(model,train_loader,val_loader,epoch_num,lr):
        
    device = torch.device("cuda")
    model.to(device)
    cost_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=lr,weight_decay=1e-1)
    scaler = GradScaler()

    batch_train_outputs = []
    batch_train_labels = []
    
    batch_val_outputs = []
    batch_val_labels = []

    time = datetime.datetime.now()
    writer1 = SummaryWriter("task1-v3/information_fusion/text-text/logs/2f-nofea/"+str(time)+'/train')
    writer2 = SummaryWriter("task1-v3/information_fusion/text-text/logs/2f-nofea/"+str(time)+'/val')

    for epoch in tqdm(range(epoch_num),ncols=80):
        model.train()
        for step, (train_process_inputs,train_micro_inputs,train_labels,train_alloy_poss) in enumerate(train_loader):
            train_process_inputs = train_process_inputs.to(device)
            train_micro_inputs = train_micro_inputs.to(device)
            # train_fea_inputs = train_fea_inputs.to(device)
            train_labels = train_labels.to(device).reshape(-1,1)
            train_alloy_poss = train_alloy_poss.to(device)
            with autocast():
                train_outputs = model(train_process_inputs,train_micro_inputs,train_alloy_poss)
                train_loss = cost_function(train_outputs,train_labels)
            
            scaler.scale(train_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            batch_train_outputs.append(train_outputs)
            batch_train_labels.append(train_labels)

        epoch_train_outputs = torch.cat(batch_train_outputs).squeeze(1)
        epoch_train_labels = torch.cat(batch_train_labels).squeeze(1)
        train_r2 = r2_score(epoch_train_labels,epoch_train_outputs)
        train_mse = mse_score(epoch_train_labels,epoch_train_outputs)
        if epoch == 119:
            last_train_outputs = epoch_train_outputs
            last_train_labels = epoch_train_labels
        writer1.add_scalars('epoch_train', {"loss":train_mse, "r2":train_r2}, epoch)
        batch_train_outputs.clear()
        batch_train_labels.clear()

        model.float()
        model.eval()
        for step, (val_process_inputs,val_micro_inputs,val_labels,val_alloy_poss) in enumerate(val_loader):
            val_process_inputs = val_process_inputs.to(device)
            val_micro_inputs = val_micro_inputs.to(device)
            # val_fea_inputs = val_fea_inputs.to(device)
            val_labels = val_labels.to(device).reshape(-1,1)
            val_alloy_poss = val_alloy_poss.to(device)
            with torch.no_grad():
                val_outputs = model(val_process_inputs,val_micro_inputs,val_alloy_poss)
            batch_val_outputs.append(val_outputs)
            batch_val_labels.append(val_labels)
        
        epoch_val_outputs = torch.cat(batch_val_outputs).squeeze(1)
        epoch_val_labels = torch.cat(batch_val_labels).squeeze(1)
        val_r2 = r2_score(epoch_val_labels,epoch_val_outputs)
        val_mse = mse_score(epoch_val_labels,epoch_val_outputs)

        if epoch == 119:
            last_val_outputs = epoch_val_outputs
            last_val_labels = epoch_val_labels
        writer2.add_scalars('epoch_val', {"loss":val_mse, "r2":val_r2}, epoch)
        batch_val_outputs.clear()
        batch_val_labels.clear()
    writer1.close()
    writer2.close()
    return last_val_outputs, last_val_labels, last_train_outputs, last_train_labels, model

def test_loop(model,test_loader):
    device = torch.device("cuda")
    model.to(device)
    model.eval()
    for step, (test_process_inputs,test_micro_inputs,test_labels,test_alloy_poss) in enumerate(test_loader):
        test_process_inputs = test_process_inputs.to(device)
        test_micro_inputs = test_micro_inputs.to(device)
        # test_fea_inputs = test_fea_inputs.to(device)
        test_labels = test_labels.to(device).reshape(-1,1)
        test_alloy_poss = test_alloy_poss.to(device)
        with torch.no_grad():
            test_outputs = model(test_process_inputs,test_micro_inputs,test_alloy_poss)
        r2 = r2_score(test_labels,test_outputs)
        print(r2)
    return test_outputs, test_labels

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
    train_labels = train_data["DAR"]
    train_dataset = PEDataset(train_process_texts,train_micro_texts,train_fea_texts,train_labels,train_alloy_poss)
    train_loader = data_loader(train_dataset, 32, True)
    #验证集数据加载
    val_alloy_poss = val_data['Alloy_pos']
    val_process_texts = val_data['process_descriptor']
    val_micro_texts = val_data['micro_descriptor']
    val_fea_texts = val_data['fea_descriptor']
    val_labels = val_data["DAR"]
    val_dataset = PEDataset(val_process_texts,val_micro_texts,val_fea_texts,val_labels,val_alloy_poss)
    val_loader = data_loader(val_dataset, 128, False)
    #模型训练
    model_name = "task1-v3/MatSciBERT"
    model = Model(model_name,tuning=False)
    epoch = 120
    lr = 1e-4
    val_outputs,val_labels,train_outputs,train_labels,trained_model = train_loop(model,train_loader,val_loader,epoch,lr=lr)
    #验证集输出保存
    val_outputs = val_outputs.cpu().detach().numpy()
    val_labels = val_labels.cpu().detach().numpy()
    #训练集输出保存
    train_outputs = train_outputs.cpu().detach().numpy()
    train_labels = train_labels.cpu().detach().numpy()
    #存为excel
    time = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
    df_val = pd.DataFrame({'Val Outputs': val_outputs, 'Val Labels':val_labels})
    df_val.to_excel('task1-v3/information_fusion/text-text/results/val/'+str(time)+"-2f-nofea.xlsx",  index=False)
    df_train = pd.DataFrame({'Train Outputs': train_outputs, 'Train Labels': train_labels})
    df_train.to_excel('task1-v3/information_fusion/text-text/results/train/'+str(time)+"-2f-nofea.xlsx", index=False)
    trained_model.float()
    #测试集数据加载
    test_path = "task1-v3/datasets/MPEA_V2.2.1.xlsx"
    sheet = 'Sheet1'
    test_data = data_load(test_path, sheet)
    test_alloy_poss = test_data['Alloy_pos']
    test_process_texts = test_data['process_descriptor']
    test_micro_texts = test_data['micro_descriptor']
    test_fea_texts = test_data['fea_descriptor']
    test_labels = test_data["DAR"]
    test_dataset = PEDataset(test_process_texts, test_micro_texts, test_fea_texts, test_labels, test_alloy_poss)
    test_loader = data_loader(test_dataset, 64, False)
    # 模型测试
    test_outputs, test_labels = test_loop(trained_model, test_loader)
    test_outputs = test_outputs.cpu().detach().numpy().flatten()
    test_labels = test_labels.cpu().detach().numpy().flatten()
    df = pd.DataFrame({'Test Outputs': test_outputs, 'Test Labels': test_labels})
    df.to_excel('task1-v3/information_fusion/text-text/results/test/'+str(time)+"-2f-nofea.xlsx", index=False)


    