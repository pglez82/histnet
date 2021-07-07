import torch
import torch.nn.functional as F
import numpy as np
import copy
import time
from sklearn.model_selection import train_test_split

class DQNModule(torch.nn.Module):
    def __init__(self,n_classes,dropout,feature_extraction_module,linear_sizes):
        super(DQNModule, self).__init__()


        self.feature_extraction_module = feature_extraction_module

        layers = []

        prev_size=feature_extraction_module.output_size
        for linear_size in linear_sizes:
            layers.append(torch.nn.Linear(prev_size,linear_size))
            layers.append(torch.nn.LeakyReLU())
            layers.append(torch.nn.Dropout(dropout))
            prev_size=linear_size

        layers.append(torch.nn.Linear(prev_size,n_classes))
        layers.append(torch.nn.Softmax(dim=0))

        self.layers = torch.nn.Sequential(*layers)

    def forward(self, input):
        return self.layers(torch.max(self.feature_extraction_module(input),dim=0)[0])

class DQNNet:
    """Note that this network does NOT use a classifier.
    HistNet builds creates artificial samples with fixed size and learns from them. Every example in each sample goes through
    the network and we build a histogram with all the examples in a sample. This is used in the second part of the network where we use
    this vector to quantify.

    Args:
        train_epochs (int): How many times to repeat the process of going over training data.
        test_epochs (int): How many times to repeat the process over the testing data (returned prevalences are averaged)
        start_lr (float): Learning rate for the network (initial value)
        end_lr (float): Learning rate for the network. The value will be decreasing after a few epochs without improving.
        n_bags (int): How many artificial samples to build per epoch.
        bag_size (int): Number of examples per sample.
        random_seed (int): Seed to make results reproducible. This net need to generate the bags so the seed is important
        dropout (float): Dropout to use in the network (avoid overfitting)
        weight_decay (float): L2 regularization for the model
        val_split (float): by default we validate using the train data. If a split is given, we partition the data and use this percentage for
                   validation and early stopping
        loss_function: loss function to optimize. The progress and early stopping will be based in L1 always.
        epsilon (float): if the error is less than this number, do not update the weights.
        device (torch.device): Device to use for training/testing
        callback_epoch: function to call after each epoch. Useful to optimize with Optuna
        verbose (int): verbose
        dataset_name (str): only for loggin purposes
    """
    def __init__(self,train_epochs,test_epochs,start_lr,end_lr,n_bags,bag_size,random_seed,linear_sizes,
                feature_extraction_module,bag_generator,batch_size,dropout=0,weight_decay=0,lr_factor=0.1,val_split=0,loss_function=torch.nn.L1Loss(),
                epsilon=0,device=torch.device('cpu'),patience=20,callback_epoch=None,verbose=0,dataset_name=""):
        self.train_epochs = train_epochs
        self.test_epochs = test_epochs
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.n_bags = n_bags
        self.bag_size = bag_size
        self.random_seed = random_seed
        self.linear_sizes = linear_sizes
        self.bag_generator=bag_generator
        self.dropout = dropout
        self.weight_decay=weight_decay
        self.lr_factor = lr_factor
        self.batch_size=batch_size
        self.val_split=val_split
        self.patience = patience
        self.loss_function = loss_function
        self.device = device
        self.verbose = verbose
        self.epsilon = epsilon
        self.callback_epoch = callback_epoch
        self.dataset_name = dataset_name
        self.feature_extraction_module = feature_extraction_module
        #make results reproducible
        torch.manual_seed(random_seed)

    def move_data_device(self,data):
        if torch.is_tensor(data):
             return data.to(self.device)
        else:
            if data.dtype=='float64':
                return torch.tensor(data).float().to(self.device)
            elif data.dtype=='int32' or data.dtype=='int64':
                return torch.tensor(data).long().to(self.device)
            else:
                return torch.tensor(data).to(self.device)

    def compute_validation_loss(self,X_val,y_val,loss):
        samples_indexes,p = self.bag_generator.compute_train_bags(n_bags=self.n_bags,bag_size=self.bag_size,y=y_val)
        val_loss = 0
        l1_loss = 0
        with torch.no_grad():
            self.model.eval()
            for i,sample_indexes in enumerate(samples_indexes):
                X_bag = X_val.index_select(0,sample_indexes)
                y_bag = y_val.index_select(0,sample_indexes)
                p_hat = self.model.forward(X_bag)
                total_loss = loss(p_hat,p[i,:])
                l1_loss += F.l1_loss(p_hat,p[i,:]).item()
                val_loss += total_loss.item()

            val_loss /= self.n_bags
            l1_loss /= self.n_bags
        return val_loss,l1_loss #We want to monitor always l1_loss

    def fit(self, X, y):
        #split training into train and validation
        if self.val_split>0:
            X_train, X_val, y_train, y_val= train_test_split(X, y, test_size=self.val_split, stratify=y, random_state=self.random_seed)
            X_val = self.move_data_device(X_val)
            y_val = self.move_data_device(y_val)
            if self.verbose>0:
                print("Spliting {} examples in training set [training: {}, validation: {}]".format(X.shape,X_train.shape,X_val.shape))
        else:
            X_train = X
            y_train = y
        
        #compute some data from the training dataset: n_features, n_examples, classes and n_classes
        self.n_features = X.shape[1]
        self.classes=np.unique(y)
        self.n_classes = len(self.classes)
        self.model = DQNModule(n_classes=self.n_classes,dropout=self.dropout,feature_extraction_module=self.feature_extraction_module,
                                linear_sizes=self.linear_sizes)
        self.model.to(self.device)
        self.best_error = 1 #Highest value. We want to store the best error during the epochs
        #if os.path.isfile('model.pyt'):
        #    self.model.load_state_dict(torch.load('model.pyt'))
        #    return
        
        #Move data to device
        X_train = self.move_data_device(X_train)
        y_train = self.move_data_device(y_train)

        if self.verbose>0:
            print("Using device {}".format(self.device))
        
        loss = self.loss_function
        self.optimizer = torch.optim.AdamW(self.model.parameters(),lr=self.start_lr,weight_decay=self.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,patience=self.patience,factor=self.lr_factor,cooldown=0,verbose=True)
        for epoch in range(self.train_epochs):
            start_epoch = time.time()
            if self.verbose>0:
                print("[{}] Starting epoch {}...".format(self.dataset_name,epoch),end='')
            #compute the training bags
            samples_indexes,p = self.bag_generator.compute_train_bags(n_bags=self.n_bags,bag_size=self.bag_size,y=y_train)
            self.model.train()
            train_loss = 0
            l1_loss_tr = 0
            for i,sample_indexes in enumerate(samples_indexes):
                X_bag = X_train.index_select(0,sample_indexes)
                p_hat = self.model.forward(X_bag)
                quant_loss = loss(p_hat,p[i,:])
                total_loss = quant_loss
                l1_loss_tr += F.l1_loss(p_hat,p[i,:]).item()
                train_loss += total_loss.item()

                if (abs(quant_loss.item())>=self.epsilon): #Juanjo Idea. Taken from SVR.
                    total_loss.backward()

                if i%self.batch_size==0 or i==self.n_bags-1:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            train_loss /= self.n_bags
            l1_loss_tr /= self.n_bags
            end_epoch = time.time()
            elapsed = end_epoch - start_epoch
            print("[Time:{:.2f}s]".format(elapsed),end='')
            if self.val_split>0:
                val_loss,l1_loss_val = self.compute_validation_loss(X_val,y_val,loss)
            else:
                val_loss=train_loss
                l1_loss_val = l1_loss_tr

            if self.callback_epoch is not None:
                self.callback_epoch(val_loss,epoch)

            if self.verbose>0:
                print("finished. Traing Loss=[{:.5f},L1={:.5f}]. Val loss = [{:.5f},L1={:.5f}]".format(train_loss,l1_loss_tr,val_loss,l1_loss_val),end='')
            #Save the best model up to this moment
            if l1_loss_val<self.best_error:
                self.best_error = l1_loss_val
                self.best_model = copy.deepcopy(self.model.state_dict())
                if self.verbose>0:
                    print("[saved best model in this epoch]",end='')

            print("")
            
            self.scheduler.step(val_loss)
            if self.optimizer.param_groups[0]['lr']<self.end_lr:
                if self.verbose>0:
                    print("Early stopping!")
                break

        if self.verbose>0:
            print("Restoring best model...")
        self.model.load_state_dict(self.best_model)
        #torch.save(self.model.state_dict(), 'model.pyt')
        return self.best_error
        

    def predict(self, X):
        """Makes the prediction over each sample repeated for n epochs. Final result will be the average."""
        X = self.move_data_device(X)
        #Special case to compare with Sebastiani
        if X.shape[0]==self.bag_size:
            with torch.no_grad():
                self.model.eval()
                return self.model.forward(X).cpu().detach().numpy()
        else:
            predictions=torch.zeros((self.n_bags*self.test_epochs,self.n_classes),device=self.device)
            for epoch in range(self.test_epochs):
                start_epoch = time.time()
                if self.verbose>10:
                    print("[{}] Starting testing epoch {}... ".format(self.dataset_name,epoch),end='')
                samples_indexes= self.bag_generator.compute_prediction_bags(dataset_size=X.shape[0],n_bags=self.n_bags,bag_size=self.bag_size)
                with torch.no_grad():
                    self.model.eval()
                    for i,sample_indexes in enumerate(samples_indexes):
                        predictions[(epoch*self.n_bags)+i,:] = self.model.forward(X[sample_indexes,:])
                end_epoch = time.time()
                elapsed = end_epoch - start_epoch
                print("[Time:{:.2f}s]".format(elapsed),end='')
                print("done.")

            return torch.mean(predictions,axis=0).cpu().detach().numpy()
