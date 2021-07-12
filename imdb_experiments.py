from ann.dqn import DQNNet
import sys,os
import pandas as pd
import numpy as np
import torch
from tensorflow.keras.preprocessing import sequence
from ann.histnet.histnet import HistNet
from ann.utils import BagGenerator
from ann.histnet.featureextraction.lstm import LSTMFeatureExtractionModule 
from quantification.utils import absolute_error,relative_absolute_error,binary_kld
from datetime import datetime
from tqdm import tqdm

#Quapy imports. Until the library is stable enough, lets depend from our local clone
sys.path.append(r'/media/nas/pgonzalez/QuaPy')
import quapy as qp
from quapy.method.meta import QuaNet
from quapy.classification.neural import NeuralClassifierTrainer, LSTMnet



vocabulary_size = 5000
embedding_size = 150
max_len = 200
dropout = 0.5
sample_size=500
device = torch.device('cuda:0')
seed_value=2032

np.random.seed(seed=seed_value)
torch.manual_seed(seed_value)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

dataset_name='imdb'

dataset = qp.datasets.fetch_reviews(dataset_name,pickle=True)
qp.data.preprocessing.index(dataset, inplace=True,max_features=5000)

vocabulary_size = dataset.vocabulary_size

x_train, y_train = dataset.training.instances, dataset.training.labels
x_test, y_test = dataset.test.instances, dataset.test.labels

x_train = sequence.pad_sequences(x_train, maxlen=max_len)
x_test = sequence.pad_sequences(x_test, maxlen=max_len)

datatrain = qp.data.LabelledCollection(x_train, y_train)

#HistNet
fe = LSTMFeatureExtractionModule(vocabulary_size=vocabulary_size,embedding_size=embedding_size,
                                hidden_size=128,
                                output_size=32,dropout_lstm=0,dropout_linear=0.2)
histnetc = HistNet(train_epochs=5000,test_epochs=1,start_lr=0.001,end_lr=0.001,n_bags=500,bag_size=sample_size,n_bins=8,random_seed = seed_value,linear_sizes=[128],
                    feature_extraction_module=fe,bag_generator=BagGenerator(device=device),batch_size=16,loss_function=torch.nn.L1Loss(),
                    dropout=0.3,epsilon=0.005,weight_decay=0.00005,histogram='softrbf',use_labels=True,use_labels_epochs=10,val_split=0.2,device=device,verbose=1,dataset_name=dataset_name)
histnetc.fit(X=x_train, y=y_train)

np.random.seed(seed=seed_value)
torch.manual_seed(seed_value)

#HistNet No Labels
fe = LSTMFeatureExtractionModule(vocabulary_size=vocabulary_size,embedding_size=embedding_size,
                                hidden_size=128,
                                output_size=32,dropout_lstm=0,dropout_linear=0.2)
histnet = HistNet(train_epochs=5000,test_epochs=1,start_lr=0.001,end_lr=0.001,n_bags=500,bag_size=sample_size,n_bins=8,random_seed = seed_value,linear_sizes=[128],
                    feature_extraction_module=fe,bag_generator=BagGenerator(device=device),batch_size=16,loss_function=torch.nn.L1Loss(),
                    dropout=0.3,epsilon=0.005,weight_decay=0.00005,histogram='softrbf',use_labels=False,val_split=0.2,device=device,verbose=1,dataset_name=dataset_name)
histnet.fit(X=x_train, y=y_train)

np.random.seed(seed=seed_value)
torch.manual_seed(seed_value)


#Basic classifiers
np.random.seed(seed=seed_value)
torch.manual_seed(seed_value)

classifier_basic = LSTMnet(vocabulary_size=vocabulary_size,n_classes=2,embedding_size=embedding_size,hidden_size=128,repr_size=100,drop_p=dropout)
learner_basic = NeuralClassifierTrainer(classifier_basic,padding_length=max_len,device='cuda:0')
learner_basic.fit(datatrain.instances,datatrain.labels)

cc_quapy = qp.method.aggregative.CC(learner_basic).fit(datatrain,fit_learner=False)
pcc_quapy = qp.method.aggregative.PCC(learner_basic).fit(datatrain,fit_learner=False)
ac_quapy = qp.method.aggregative.ACC(learner_basic).fit(datatrain,fit_learner=False)
pac_quapy = qp.method.aggregative.PACC(learner_basic).fit(datatrain,fit_learner=False)
hdy_quapy = qp.method.aggregative.HDy(learner_basic).fit(datatrain,fit_learner=False)

#train a classifier for the data
classifier_quanet = LSTMnet(vocabulary_size=vocabulary_size,n_classes=2,embedding_size=embedding_size,hidden_size=128,repr_size=100,drop_p=dropout)
learner_quanet = NeuralClassifierTrainer(classifier_quanet,padding_length=max_len,device='cuda:0')

#Fit quanet
np.random.seed(seed=seed_value)
torch.manual_seed(seed_value)

quanet = QuaNet(learner_quanet,checkpointname='QuaNet-IMDB.dat',sample_size=sample_size,device='cuda')
quanet.fit(datatrain)

np.random.seed(seed=seed_value)
torch.manual_seed(seed_value)

#DQNUsing
fe = LSTMFeatureExtractionModule(vocabulary_size=vocabulary_size,embedding_size=embedding_size,
                                hidden_size=128,
                                output_size=32,dropout_lstm=0,dropout_linear=0.2)
dqn = DQNNet(train_epochs=5000,test_epochs=1,start_lr=0.001,end_lr=0.001,n_bags=500,bag_size=sample_size,random_seed = seed_value,linear_sizes=[128],
                    feature_extraction_module=fe,bag_generator=BagGenerator(device=device),batch_size=16,loss_function=torch.nn.L1Loss(),
                    dropout=0.3,epsilon=0.005,weight_decay=0.00005,val_split=0.2,device=device,verbose=1,dataset_name=dataset_name)
dqn.fit(X=x_train, y=y_train)

methods = [cc_quapy,pcc_quapy,ac_quapy,pac_quapy,hdy_quapy,quanet,dqn,histnet,histnetc]
method_names = ['cc_quapy','pcc_quapy','ac_quapy','pac_quapy','hdy_quapy','quanet','dqn','histnet','histnetc']

np.random.seed(seed=seed_value)
torch.manual_seed(seed_value)

results=pd.DataFrame(columns=(['true']+method_names),dtype='float')
results_mae = pd.DataFrame(columns=(method_names),dtype='float')
results_mrae = pd.DataFrame(columns=(method_names),dtype='float')
results_kld = pd.DataFrame(columns=(method_names),dtype='float')
datatest = qp.data.LabelledCollection(x_test, y_test)

print("Making predictions...")
for r in tqdm(range(2000)):
    p=np.random.random()*(0.99-0.01) + 0.01
    index = datatest.sampling_index(sample_size, p,1-p)
    sample = datatest.sampling_from_index(index)
    results.loc[r,'true']=p
    predictions_test = learner_basic.predict_proba(x_test).astype('float')
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

    now = datetime.now()

results_mae.loc['mean'] = results_mae.mean()
results_mrae.loc['mean'] = results_mrae.mean()
results_kld.loc['mean'] = results_kld.mean()

if not os.path.exists("resultsimdb"):
    os.makedirs("resultsimdb")

filename1 = 'resultsimdb/results_imdb_{}.csv'.format(now.strftime('%Y-%m-%d_%H-%M-%S'))
filename2 = 'resultsimdb/results_mae_imdb_{}.csv'.format(now.strftime('%Y-%m-%d_%H-%M-%S'))
filename3 = 'resultsimdb/results_mrae_imdb_{}.csv'.format(now.strftime('%Y-%m-%d_%H-%M-%S'))
filename4 = 'resultsimdb/results_kld_imdb_{}.csv'.format(now.strftime('%Y-%m-%d_%H-%M-%S'))


with open(filename1, 'a') as the_file:
    the_file.write(str(histnet.__dict__))

results.to_csv(filename1,float_format='%.5f',mode='a')
results_mae.to_csv(filename2,float_format='%.5f',mode='a')
results_mrae.to_csv(filename3,float_format='%.5f',mode='a')
results_kld.to_csv(filename4,float_format='%.5f',mode='a')