import pandas as pd
import numpy as np
import os
import psutil
import gc
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
import transformers
from transformers import BertTokenizer, BertPreTrainedModel, BertModel
from torch.optim import AdamW
import warnings
warnings.filterwarnings("ignore")

my_path = "" #add the data path here
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#-------Hyperparameters---------
MODEL_TYPE = 'bert-base-multilingual-uncased'
NUM_FOLDS = 5
NUM_FOLDS_TO_TRAIN = 3 
L_RATE = 1e-6
MAX_LEN = 300
NUM_EPOCHS = 6
BATCH_SIZE = 32
NUM_CORES = 0

# -------Loading the data and split---------
df = pd.read_csv(my_path + "file_name.csv")
train, test, y_train, y_test = train_test_split(df[['text','valence']], df[['valence']], test_size=0.2, random_state=42)

# initialize kfold
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=103)

# for stratification
y = train['valence']

# Put the folds into a list. This is a list of tuples.
fold_list = list(kf.split(train, y))

ds_train_list = []
ds_val_list = []
for i, fold in enumerate(fold_list):
    ds_train = train[train.index.isin(fold[0])]
    ds_val = train[train.index.isin(fold[1])]
    ds_train_list.append(ds_train)
    ds_val_list.append(ds_val
    )

#----------Dataset class for train data----------
#Data Loader using torch Dataset and DataLoader
class TrainDataset(Dataset):

    def __init__(self,dataframe):
        self.df = dataframe

    def __getitem__(self,index):

        text = self.df.loc[index,'text']
        #tokenizing the data
        encoded_dict = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length = MAX_LEN,
            return_attention_mask=True,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        #Extracting the model inputs, already tensors
        padded_tokens_list = encoded_dict['input_ids'].squeeze()
        att_mask = encoded_dict['attention_mask'].squeeze()
        token_type_ids = encoded_dict['token_type_ids'].squeeze()
        #transforming target label to tensors
        # target = self.df.loc[index,'valence']
        target = torch.tensor(self.df.loc[index,'valence'],dtype=torch.float32)
        sample = (padded_tokens_list,att_mask,token_type_ids,target)
        return sample
        
    def __len__(self):
        return len(self.df)
      
#--------Dataset class for test data------------
class TestDataset(Dataset):

    def __init__(self,df):
        self.df = df

    def __getitem__(self,index):
        text = self.df.loc[index,'text']
        #tokenizing the data
        encoded_dict = tokenizer.encpde_plus(
            text,
            add_special_tokens=True,
            max_length = MAX_LEN,
            return_attention_mask=True,
            return_tensors = 'pt',
            truncation=True,
            padding='max_length'
        )
        #Extracting the model inputs, already tensors
        padded_tokens_list = encoded_dict['input_ids'].squeeze()
        att_mask = encoded_dict['attention_mask'].squeeze()
        token_type_ids = encoded_dict['token_type_ids'].squeeze()
        sample = (padded_tokens_list,att_mask,token_type_ids)
        return sample

    def __len__(self):
        return len(self.df)

#------Regression model with BERT--------
# we want predict numerical values of sentiments, we build a regression head over the BertModel
class RegressionLayer(BertPreTrainedModel):

    def __init__(self,confg):
        super().__init__(confg)
        self.bert=BertModel(confg)
        self.regressor = nn.Linear(confg.hidden_size,1)
        self.init_weights() #initializing weights for the whole model

    #defining the forward pass, returns the MSE loss and logits which only one value per sample.
    def forward(self,input_ids=None,attention_mask=None,labels=None, token_type_ids=None):
        outputs = self.bert(input_ids=input_ids,attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
        pooled_output = outputs.pooler_output
        logits = self.regressor(pooled_output)
        last_hidden_state = outputs.last_hidden_state
        
        loss= None
        if labels is not None:
            loss=nn.MSELoss()(logits.view(-1),labels.view(-1))
        return {'loss':loss,'logits':logits, 'last_hidden_state':last_hidden_state}

#------Memory Usage--------
#You can added this function to check memory usage
def memory_usage():
    mem = psutil.Process(os.getpid()).memory_info().rss / 1024**3
    print(f"Memory usage: {mem:.2f} GB")

# Model and tokenizer instances
tokenizer = BertTokenizer.from_pretrained(MODEL_TYPE,do_lower_case=True)
model = RegressionLayer.from_pretrained(MODEL_TYPE)

#------Training loop------
seed_val = 1024
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

#traing on 1 fold for experiments
fold_train = ds_train_list[0].reset_index(drop=True)
fold_val = ds_val_list[0].reset_index(drop=True)

#calling our dataset class
train_dataset = TrainDataset(fold_train)
val_dataset = TrainDataset(fold_val)

#Saving the MSE for each epoch
epoch_val_mse=[]
#Tracking epochs without improvement on validation dataset
epochs_without_improvement = 0
epoch=0

#If in 3 consecutive epochs, no improvement on validation data, the algorithm will stop
while epochs_without_improvement<3 and epoch<NUM_EPOCHS:
    print('======== Epoch {:} / {:} ========'.format(epoch+1, NUM_EPOCHS))

    if epoch==0:
        model.to(device)
        optimizer = AdamW(model.parameters(),
                         lr=L_RATE,
                         eps=1e-8)
    else:
        #For every epoch>0 we will use the weights of the last epoch to initialize the model's weights of the current epoch.
        model_path = 'model_epoch'+'.bin'
        model.load_state_dict(torch.load(model_path))
        model.to(device)
    #Calling dataloader
    train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=BATCH_SIZE,
                                               shuffle=True,num_workers=NUM_CORES)
    val_dataloader = torch.utils.data.DataLoader(val_dataset,batch_size=BATCH_SIZE,
                                               shuffle=True,num_workers=NUM_CORES)
    print('Training...')
    model.train()
    torch.set_grad_enabled(True)
    total_train_loss = 0
    
    for i, batch in enumerate(train_dataloader):
        
        print('Batch '+str(i+1)+'/'+str(len(train_dataloader)),end='\r')
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        token_type_ids = batch[2].to(device)
        labels = batch[3].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels)
        
        loss = outputs['loss']
        total_train_loss +=loss.item()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        optimizer.step()
        print('Train loss epoch '+str(epoch+1) +' :',total_train_loss)
        print(torch.isnan(outputs['last_hidden_state']).any())
        print(torch.isnan(outputs['logits']).any())
    print('Validation...')
    gc.collect()
    torch.set_grad_enabled(False)
    model.eval()
    total_val_loss = 0
    labels_list=[]
    preds_list=[]
    for j,val_batch in enumerate(val_dataloader):

        input_ids = val_batch[0].to(device)
        attention_mask = val_batch[1].to(device)
        token_type_ids = val_batch[2].to(device)
        labels = val_batch[3].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels)
        loss = outputs['loss']
        total_val_loss +=loss.item()
        val_preds = outputs['logits'].detach().cpu().numpy()
        val_labels = labels.detach().cpu().numpy()
        labels_list.extend(val_labels)
        preds_list.extend(val_preds)
        
    MSE = mean_squared_error(labels_list, preds_list) #squared=False
    
    print('MSE: ',MSE)
    print('val_loss',total_val_loss)

    if epoch==0:
        model_name = 'model_epoch'+'.bin'
        torch.save(model.state_dict(), model_name)
        print('Saved model as ', model_name)
    else:
        if MSE<min(epoch_val_mse):
            model_name = 'model_epoch'+'.bin'
            torch.save(model.state_dict(), model_name)
            print('Val accuracy improved,saved model as ', model_name)
            epochs_without_improvement=0
        else:
            epochs_without_improvement+=1
            print('Val accuracy not improved, total epochs without improvement ', 
                  epochs_without_improvement)
    epoch_val_mse.append(MSE)
    epoch+=1
