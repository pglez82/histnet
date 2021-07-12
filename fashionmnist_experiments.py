from datetime import datetime
from tensorflow.keras.datasets import fashion_mnist as mnist
from ann.histnet.featureextraction.cnn import CNNFeatureExtractionModule
from ann.histnet.histnet import HistNet
from ann.dqn import DQNNet
from ann.utils import BagGenerator
import torch
from torch.utils.data import TensorDataset, DataLoader
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
import os

import sys
sys.path.append(r'/media/nas/pgonzalez/QuaPy')
import quapy as qp
from quantification.utils import absolute_error,relative_absolute_error,binary_kld
from quapy.method.meta import QuaNet

def preprocess_data(x_train,y_train,x_test,y_test):
    positiveclass = 0
    negativeclass = 6

    x_train = x_train[(y_train==positiveclass) | (y_train==negativeclass),:]
    y_train = y_train[(y_train==positiveclass) | (y_train==negativeclass)]
    x_test = x_test[(y_test==positiveclass) | (y_test==negativeclass),:]
    y_test = y_test[(y_test==positiveclass) | (y_test==negativeclass)]


    positives = y_train==positiveclass
    negatives = y_train==negativeclass
    y_train[positives]=1
    y_train[negatives]=0

    positives = y_test==positiveclass
    negatives = y_test==negativeclass
    y_test[positives]=1
    y_test[negatives]=0

    x_train = x_train.astype(float)
    x_test = x_test.astype(float)
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

    x_train=np.expand_dims(x_train, axis=1) #add the channel dimension
    x_test=np.expand_dims(x_test, axis=1)

    return x_train,y_train,x_test,y_test

def save_results(results,results_mae,results_mrae,results_kld,histnet,dataset):
    print("Done.")
    print(results)
    print(results_mae)
    now = datetime.now()

    results_mae.loc['mean'] = results_mae.mean()
    results_mrae.loc['mean'] = results_mrae.mean()
    results_kld.loc['mean'] = results_kld.mean()

    if not os.path.exists("resultsmnist"):
        os.makedirs("resultsmnist")

    filename1 = 'resultsmnist/results_{}_{}.csv'.format(dataset,now.strftime('%Y-%m-%d_%H-%M-%S'))
    filename2 = 'resultsmnist/results_mae_{}_{}.csv'.format(dataset,now.strftime('%Y-%m-%d_%H-%M-%S'))
    filename3 = 'resultsmnist/results_mrae_{}_{}.csv'.format(dataset,now.strftime('%Y-%m-%d_%H-%M-%S'))
    filename4 = 'resultsmnist/results_kld_{}_{}.csv'.format(dataset,now.strftime('%Y-%m-%d_%H-%M-%S'))


    with open(filename1, 'a') as the_file:
        the_file.write(str(histnet.__dict__))

    results.to_csv(filename1,float_format='%.5f',mode='a')
    results_mae.to_csv(filename2,float_format='%.5f',mode='a')
    results_mrae.to_csv(filename3,float_format='%.5f',mode='a')
    results_kld.to_csv(filename4,float_format='%.5f',mode='a')

class CNNMNISTClassifier(pl.LightningModule):
    def __init__(self,repr_size=32):
        super(CNNMNISTClassifier, self).__init__()        
        self.cnnmodule = CNNFeatureExtractionModule(output_size=repr_size)
        self.output = torch.nn.Linear(repr_size, 2)
        self.loss = torch.nn.CrossEntropyLoss()
    
    def transform(self,x):
        return self.cnnmodule(x)

    def forward(self, x):
        features = self.cnnmodule(x)
        return self.output(features)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        train_loss = self.loss(y_hat, y)
        self.log('train_loss', train_loss)
        return train_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log('val_loss', loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

class CNNMNISTTrainer():
    def __init__(self,repr_size=32,random_seed=2032) -> None:
        self.random_seed = random_seed
        self.classes_ = np.asarray([0, 1])
        self.repr_size = repr_size

    def load_data(self,x_train,y_train):
        #separate data for validation
        x_train, x_val, y_train, y_val= train_test_split(x_train, y_train, test_size=0.2, stratify=y_train, random_state=self.random_seed)
        x_train = torch.Tensor(x_train)
        y_train = torch.LongTensor(y_train)
        x_val = torch.Tensor(x_val)
        y_val = torch.LongTensor(y_val)
        train_dataset = TensorDataset(x_train,y_train)
        train_dataloader = DataLoader(train_dataset,batch_size=100,shuffle=True)
        val_dataset = TensorDataset(x_val,y_val)
        val_dataloader = DataLoader(val_dataset,batch_size=100,shuffle=True)
        return train_dataloader,val_dataloader

    def fit(self,x,y):
        train_dataloader,val_dataloader = self.load_data(x,y)
        self.model = CNNMNISTClassifier(repr_size=self.repr_size)
        self.trainer = pl.Trainer(check_val_every_n_epoch=1,callbacks=[EarlyStopping(monitor='val_loss')])
        self.trainer.fit(self.model, train_dataloader, val_dataloader)

    def predict_proba(self,x):
        x = torch.Tensor(x)
        logits = self.model(x)
        return F.softmax(logits,dim=1).detach().numpy()

    def predict(self,x):
        probs = self.predict_proba(x)
        return np.argmax(probs,axis=1)

    def transform(self,x):
        x = torch.Tensor(x)
        return self.model.transform(x).detach().numpy()

    def get_params(self):
        return {}


device = torch.device('cuda:0')
sample_size = 100
seed_value=2032
dataset_name = 'mnist_fashion_0_6'

np.random.seed(seed=seed_value)
torch.manual_seed(seed_value)
pl.seed_everything(seed_value)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

(x_train, y_train), (x_test, y_test) = mnist.load_data()

#Preprocess and filter the data
x_train, y_train, x_test, y_test = preprocess_data(x_train,y_train,x_test,y_test)

datatrain = qp.data.LabelledCollection(x_train, y_train)

#HistNet Labels

np.random.seed(seed=seed_value)
torch.manual_seed(seed_value)
pl.seed_everything(seed_value)

fe = CNNFeatureExtractionModule(output_size=16)
histnetc = HistNet(train_epochs=5000,test_epochs=1,start_lr=0.001,end_lr=0.0001,n_bags=500,bag_size=sample_size,n_bins=8,random_seed = seed_value,linear_sizes=[128],
                    feature_extraction_module=fe,bag_generator=BagGenerator(device=device),batch_size=16,loss_function=torch.nn.L1Loss(),patience=20,
                    dropout=0.3,epsilon=0.005,weight_decay=0,histogram='softrbf',use_labels=True,use_labels_epochs=10,val_split=0.2,device=device,verbose=1,dataset_name=dataset_name)
histnetc.fit(X=x_train, y=y_train)

#HistNet No Labels

np.random.seed(seed=seed_value)
torch.manual_seed(seed_value)
pl.seed_everything(seed_value)

fe = CNNFeatureExtractionModule(output_size=16)
histnet = HistNet(train_epochs=5000,test_epochs=1,start_lr=0.0001,end_lr=0.0001,n_bags=500,bag_size=sample_size,n_bins=8,random_seed = seed_value,linear_sizes=[128],
                    feature_extraction_module=fe,bag_generator=BagGenerator(device=device),batch_size=16,loss_function=torch.nn.L1Loss(),patience=20,
                    dropout=0.3,epsilon=0.005,weight_decay=0,histogram='softrbf',use_labels=False,val_split=0.2,device=device,verbose=1,dataset_name=dataset_name)
histnet.fit(X=x_train, y=y_train)

np.random.seed(seed=seed_value)
torch.manual_seed(seed_value)
pl.seed_everything(seed_value)

#DQN
fe = CNNFeatureExtractionModule(output_size=16)
dqn = DQNNet(train_epochs=5000,test_epochs=1,start_lr=0.0001,end_lr=0.0001,n_bags=500,bag_size=sample_size,random_seed = seed_value,linear_sizes=[128],
                    feature_extraction_module=fe,bag_generator=BagGenerator(device=device),batch_size=16,loss_function=torch.nn.L1Loss(),
                    dropout=0.3,epsilon=0.005,weight_decay=0.00005,val_split=0.2,device=device,verbose=1,dataset_name=dataset_name)
dqn.fit(X=x_train, y=y_train)


np.random.seed(seed=seed_value)
torch.manual_seed(seed_value)
pl.seed_everything(seed_value)

#Quanet
classifier_quanet = CNNMNISTTrainer(repr_size=16)
quanet = QuaNet(classifier_quanet,checkpointname='QuaNet-MNIST.dat',sample_size=sample_size,device='cuda')
quanet.fit(datatrain)

#Traditional quantifiers

np.random.seed(seed=seed_value)
torch.manual_seed(seed_value)
pl.seed_everything(seed_value)

classifier = CNNMNISTTrainer()
classifier.fit(x_train,y_train)

cc_quapy = qp.method.aggregative.CC(classifier).fit(datatrain,fit_learner=False)
pcc_quapy = qp.method.aggregative.PCC(classifier).fit(datatrain,fit_learner=False)
ac_quapy = qp.method.aggregative.ACC(classifier).fit(datatrain,fit_learner=False)
pac_quapy = qp.method.aggregative.PACC(classifier).fit(datatrain,fit_learner=False)
hdy = qp.method.aggregative.HDy(classifier).fit(datatrain,fit_learner=False)

methods = [cc_quapy,pcc_quapy,ac_quapy,pac_quapy,hdy,quanet,dqn,histnet,histnetc]
method_names = ['cc_quapy','pcc_quapy','ac_quapy','pac_quapy','hdy','quanet','dqn','histnet','histnetc']

print("Making predictions...")

np.random.seed(seed=seed_value)
torch.manual_seed(seed_value)
pl.seed_everything(seed_value)

n_test_bags = 2000
results=pd.DataFrame(columns=(['true']+method_names),dtype='float')
results_mae = pd.DataFrame(columns=(method_names),dtype='float')
results_mrae = pd.DataFrame(columns=(method_names),dtype='float')
results_kld = pd.DataFrame(columns=(method_names),dtype='float')
datatest = qp.data.LabelledCollection(x_test, y_test)

for r in tqdm(range(n_test_bags)):
    p=np.random.random()*(0.99-0.01) + 0.01
    index = datatest.sampling_index(sample_size, p,1-p)
    sample = datatest.sampling_from_index(index)
    results.loc[r,'true']=p
    predictions_test = classifier.predict_proba(x_test).astype('float')
    for method,method_name in zip(methods,method_names):
        if isinstance(method,qp.method.base.BaseQuantifier):
            predictions = method.quantify(sample.instances)
        else:
            predictions = method.predict(sample.instances)
        mae = absolute_error(p, predictions[0])
        mrae = relative_absolute_error(p,predictions[0])
        kld = binary_kld(p,predictions[0])
        results.loc[r,method_name]=predictions[0]
        results_mae.loc[r,method_name]=mae
        results_mrae.loc[r,method_name]=mrae
        results_kld.loc[r,method_name]=kld

save_results(results,results_mae,results_mrae,results_kld,histnet,dataset_name)