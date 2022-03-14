import numpy as np
import os
import random
from skorch import NeuralNetRegressor
from skorch.callbacks import ProgressBar, Checkpoint
from skorch.callbacks import EarlyStopping, LRScheduler, WarmRestartLR
import torch.nn.functional as F
import torch
from torch import nn
from torch import erf, erfinv
from skorch import NeuralNet, NeuralNetBinaryClassifier, NeuralNetClassifier
from ipdb import set_trace as st

__all__ = [
    'seed_torch',
    'init_weights',
    'shuffle_n'
]

def my_custom_loss_func(y_true, y_pred):

    y_pred = torch.from_numpy(y_pred).cuda()
    y_true = torch.from_numpy(y_true)
    # import ipdb;ipdb.set_trace()
    loss = my_weight_rmse_CB(y_pred, y_true)
    return loss

def shuffle_n(input, n, verbose=False, seed=2333):
    
    seed_torch(seed)
    # import ipdb; ipdb.set_trace()
    idx = np.arange(len(input)//n)
    np.random.shuffle(idx)

    out = input[n*idx[0]:np.minimum(n*(idx[0]+1), len(input))]
    for i in np.arange(1, len(idx)):
        out = np.hstack((out,
                         input[n*idx[i]:np.minimum(n*(idx[i]+1),
                                                   len(input))]))
    # import ipdb; ipdb.set_trace()
    if verbose:
        print('Shape after shuffling:', out.shape)
    return out


def seed_torch(seed=2333):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.01)

def norm_cdf(cdf, thres_cdf):
    cdf_05 = np.zeros(cdf.shape)
    cdf_05[np.where(cdf == thres_cdf)[0]] = 0.5
    # import ipdb;ipdb.set_trace()
    idx_pos = np.where(cdf >= thres_cdf)[0]
    idx_neg = np.where(cdf < thres_cdf)[0]

    cdf_05[idx_pos] = (cdf[idx_pos] - thres_cdf)/(cdf.max() - thres_cdf)*0.5+0.5
    cdf_05[idx_neg] = 0.5 - (cdf[idx_neg] - thres_cdf)/(cdf.min() - thres_cdf)*0.5

    return cdf_05

def cdf_AH(X):
    X = X.astype(np.int)
    cdf = np.zeros(X.max() - X.min()+1)
    ccdf = np.zeros(X.max() - X.min()+1)
    CDF = np.zeros(X.shape[0])
    CCDF = np.zeros(X.shape[0])
    x = np.arange(X.min(), X.max()+1)

    for i in x:
        idx = np.where(X <= i)[0]
        cdf[i-X.min()] = len(idx)/len(X)
        ccdf[i-X.min()] = 1 - len(idx)/len(X)

    for i, dst_clu in enumerate(X):
        try:
            idx = np.where(x == dst_clu)[0]
            CDF[i] = cdf[idx]
            CCDF[i] = ccdf[idx]
        except:
            import ipdb;ipdb.set_trace()

        CDF[i] = cdf[idx]
        CCDF[i] = ccdf[idx]

    # import ipdb;ipdb.set_trace()
    return CCDF, CDF, ccdf[np.where(x == -100)[0]]

def maxmin_scale(X, X_t):

    X_max = X.max(axis=0)
    X_min = X.min(axis=0)

    X = (X - X_min)/(X_max-X_min)
    X_t = (X_t - X_min)/(X_max-X_min)

    return X, X_t

def std_scale(X, X_t):

    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)

    X = (X - X_mean)/X_std
    X_t = (X_t - X_mean)/X_std

    return X, X_t


my_callbacks = [
    Checkpoint(),
    EarlyStopping(patience=5),
    LRScheduler(WarmRestartLR),
    ProgressBar(),
]

class MLP(torch.nn.Module):
    def __init__(self, out=12, in_dims=5):
        super(MLP, self).__init__()

        self.drop = torch.nn.Dropout(p=0.6)
        # dropout
        self.fc1 = torch.nn.Linear(in_dims, 12)
        # Fully-connected classifier layer
        self.fc2 = torch.nn.Linear(12, 12)
        # Fully-connected classifier layer
        self.fc3 = torch.nn.Linear(12, out)
        # Fully-connected classifier layer

    def forward(self, x):
        # point B

        # out = np.zeros(X.shape)
        # for i in np.arange(out):
        # x = X[:, i]
        x = self.fc1(x)
        x = F.relu(x)
        # x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        x = F.relu(x)
        out = self.fc3(x)
        # return F.log_softmax(x, dim=1)
        return out


class MLP_EN(torch.nn.Module):
    def __init__(self, out=12, in_dims=5):
        super(MLP_EN, self).__init__()

        self.drop = torch.nn.Dropout(p=0.6)
        # dropout
        self.fc1 = torch.nn.Linear(in_dims, 12)
        # Fully-connected classifier layer
        self.fc2 = torch.nn.Linear(12, 12)
        # Fully-connected classifier layer
        self.fc3 = torch.nn.Linear(12, out)
        # Fully-connected classifier layer

    def forward(self, x):
        # point B

        # out = np.zeros(X.shape)
        # for i in np.arange(out):
        # x = X[:, i]
        x = self.fc1(x)
        x = F.relu(x)
        # x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        x = F.relu(x)
        out = self.fc3(x)
        # return F.log_softmax(x, dim=1)
        return out

class CNN_1D(torch.nn.Module):
    def __init__(self, num_channel=9, out=1):
        super(CNN_1D, self).__init__()
        self.conv1 = torch.nn.Conv1d(num_channel, 18, kernel_size=2)
        # 9 input channels, 18 output channels
        self.conv2 = torch.nn.Conv1d(18, 36, kernel_size=2)
        # 18 input channels from previous Conv. layer, 36 out
        self.conv2_drop = torch.nn.Dropout2d(p=0.2)
        # dropout
        self.fc1 = torch.nn.Linear(36*1, 36)
        # Fully-connected classifier layer
        self.fc2 = torch.nn.Linear(36, 16)
        # Fully-connected classifier layer
        self.fc3 = torch.nn.Linear(16, out)
        # Fully-connected classifier layer

    def forward(self, x):
        x = F.relu(F.max_pool1d(self.conv1(x), 2))
        # print(x.shape)
        # x = F.relu(F.max_pool1d(self.conv2_drop(self.conv2(x)), 2))
        # print(x.shape)


        # point A
        x = x.view(x.shape[0], -1)

        # point B
        x = self.fc1(x)
        x = F.leaky_relu(x)
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        x = self.fc3(x)
        # return F.log_softmax(x, dim=1)
        return x.float()


class lstm_gp(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, num_layers=2):
        super(lstm_gp, self).__init__()
        # ipdb.set_trace()
        self.rnn = torch.nn.GRU(input_size, hidden_size,
                                num_layers, batch_first=True)  # , bidirectional=True
        self.gru1 = torch.nn.GRU(input_size,hidden_size,hidden_size, bidirectional=True) #, bidirectional=True
        self.gru2 = torch.nn.GRU(hidden_size*2,hidden_size,num_layers, bidirectional=True) #, bidirectional=True
        self.rnn1 = torch.nn.LSTM(input_size,hidden_size,hidden_size, bidirectional=True) #, bidirectional=True
        self.rnn2 = torch.nn.LSTM(hidden_size,hidden_size,num_layers, bidirectional=False) #, bidirectional=True
        
        self.reg = torch.nn.Linear(hidden_size, output_size)

    def forward(self,x):

        # import ipdb;ipdb.set_trace()
        x, _ = self.rnn(x)
        
        s,b,h = x.shape
        x = x.view(s*b, h)
        #ipdb.set_trace()
        x = self.reg(x)
        x = x.view(s,b,-1)
        # import ipdb;ipdb.set_trace()
        return x[:, -1]
    
class lstm_reg(torch.nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.6,
                 output_size=1, num_layers=2):
        super(lstm_reg, self).__init__()
        # ipdb.set_trace()
        self.gru1 = torch.nn.GRU(input_size, hidden_size,
                                num_layers,
                                batch_first=True
        )  # , bidirectional=True
        self.gru2 = torch.nn.GRU(input_size,hidden_size,
                                 num_layers,
                                 bidirectional=True,
                                 batch_first=True) #, bidirectional=True
        
        self.gru3 = torch.nn.GRU(hidden_size*2,hidden_size,num_layers, bidirectional=True) #, bidirectional=True
        self.lstm1 = torch.nn.LSTM(input_size,
                                  hidden_size,
                                  num_layers,
                                  # bidirectional=True,
                                  batch_first=True) #, bidirectional=True
        
        self.lstm2 = torch.nn.LSTM(hidden_size,hidden_size,num_layers, bidirectional=False) #, bidirectional=True
        
        self.reg = torch.nn.Linear(hidden_size, output_size)
        self.out = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, 20),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            # torch.nn.Linear(256, 64),
            # torch.nn.ReLU(),
            # torch.nn.Dropout(dropout),
            torch.nn.Linear(20, output_size),
            # nn.LogSoftmax(dim=1),
        )

    def forward(self,x):

        # import ipdb;ipdb.set_trace()
        x, _ = self.gru1(x)
        # x, _ = self.lstm1(x)
        
        s,b,h = x.shape
        x = x.reshape([s*b, h])
        #ipdb.set_trace()
        # x = self.out(x)
        x = self.reg(x)
        x = x.view(s,b,-1)
        return x
    
class PhysinformedNet(NeuralNetRegressor):
    def __init__(self,  *args, thres=0.5,
                 **kwargs):
        super().__init__(*args, **kwargs)
        # self.ReLU = torch.nn.ReLU()
        # self.ReLU6 = torch.nn.ReLU6()
        self.thres = thres

    def get_loss(self, y_pred, y_true, X, training=False):
        # print('y_pred:', y_pred.shape)
        # print('y_true:', y_true.shape)
        # print('X:', X.shape)
        # loss_ori = super().get_loss(y_pred[:, -1, :].squeeze(),
                                    # y_true.squeeze(), X=X, training=training)
        # loss += self.lambda1 * sum([w.abs().sum() for w in self.module_.parameters()])
        # X = X.cuda()

        '''

        loss_RMS = torch.mean((y_pred.sum(axis=1) -
                          y_true.cuda())**2)

        loss_RMS = torch.mean((y_pred.std(axis=1) -
                          y_true.cuda())**2)
        '''
        # import ipdb;ipdb.set_trace()

        loss_CB = my_weight_rmse_CB(y_pred, y_true,
                                    thres=self.thres)

        
        # import ipdb;ipdb.set_trace()
        loss_RMS = torch.mean((y_pred -
                          y_true.cuda())**2)
        # loss_RE = torch.mean((y_pred -
                          # y_true.cuda())**2)
        

        # ipdb.set_trace()

        # print('loss_term1:', loss_term1)
        # print('loss:', loss)
        # ipdb.set_trace()
        loss = loss_CB
        # loss = loss_RMS

        # loss += loss_term0.mean()
        # print('loss+0:', loss)
        # loss += loss_term1.squeeze()
        # print('loss+1:', loss)
        # loss += loss_term2.squeeze()
        # print('loss+2:', loss)
        # loss += loss_term3.squeeze()
        # print('loss+3:', loss)

        return loss

class PhysinformedNet_EN(NeuralNet):
    def __init__(self, *args, pre_weight='False',
                 num_output, weight, alpha, l1_ratio,
                 # P,
                 loss,
                 **kwargs):
        super().__init__(*args, **kwargs)
        # self.ReLU = nn.ReLU()
        # self.ReLU6 = nn.ReLU6()
        # self.criterion = criterion
        # self.X = 0
        # self.loss = my_BCEloss
        # self.loss = my_weight_BCEloss
        # self.loss = my_L1_loss
        # self.P = P
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.loss = loss
        self.pre_weight = pre_weight
        self.weight = weight
        self.num_output = num_output

    def get_loss(self, y_pred, y_true, X, training=False):
        # print('y_pred:', y_pred.shape)
        # print('y_true:', y_true.shape)
        # print('X:', X.shape)
        # loss_ori = torch.zeros(1).cuda()

        if np.isnan(y_pred.cpu().detach().numpy().sum()):

            pass

            # import ipdb;ipdb.set_trace()
        '''
        print('BCE value:', self.loss(y_pred.type(torch.FloatTensor).cuda(),
                                      y_true.type(torch.FloatTensor).cuda()))
        '''        
       
             
        loss_ori = self.loss(y_pred=y_pred.type(torch.FloatTensor).cuda(),
                             y=y_true.type(torch.FloatTensor).cuda(),
                             pre_weight=self.pre_weight,
                             # alpha=self.alpha,
                             # l1_ratio=self.l1_ratio,
                             # P=self.P,
                             weight=self.weight)
        # import ipdb; ipdb.set_trace()
        
        l1_lambda = self.alpha*self.l1_ratio
        l1_reg = torch.tensor(0.)
        l1_reg = l1_reg.cuda()
        for param in self.module.parameters():
            l1_reg += torch.sum(torch.abs(param))
        loss1 = l1_lambda * l1_reg
        loss_ori += loss1
        
        l2_lambda = self.alpha*(1-self.l1_ratio) 
        l2_reg = torch.tensor(0.)
        l2_reg = l2_reg.cuda()
        for param in self.module.parameters():
            l2_reg += torch.norm(param).sum()
        loss_ori += l2_lambda * l2_reg
        
        # loss_ori += loss_ori_t

        # loss_ori = loss_ori
        # import ipdb; ipdb.set_trace()
        return loss_ori



class PhysinformedNet_1D(NeuralNetRegressor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ReLU = torch.nn.ReLU()
        self.ReLU6 = torch.nn.ReLU6()

    def get_loss(self, y_pred, y_true, X, training=False):
        # print('y_pred:', y_pred.shape)
        # print('y_true:', y_true.shape)
        # print('X:', X.shape)
        # import ipdb;ipdb.set_trace()
        
        loss_ori = super().get_loss(y_pred.squeeze(),
                                    y_true.squeeze(), 
                                    X=X, 
                                    training=training)
        # loss += self.lambda1 * sum([w.abs().sum() for w in self.module_.parameters()])
        # X = X.cuda()

        '''

        loss_RMS = torch.mean((y_pred.sum(axis=1) -
                          y_true.cuda())**2)

        loss_RMS = torch.mean((y_pred.std(axis=1) -
                          y_true.cuda())**2)

        loss_RMS = torch.mean((y_pred[:,:, 0] -
                          y_true[:,:].cuda())**2)
        loss_RE = torch.mean((y_pred[:,:, 0] -
                          y_true[:,:].cuda())**2)
        '''

        # ipdb.set_trace()

        # print('loss_term1:', loss_term1)
        # print('loss:', loss)
        # ipdb.set_trace()
        loss = loss_ori

        # loss += loss_term0.mean()
        # print('loss+0:', loss)
        # loss += loss_term1.squeeze()
        # print('loss+1:', loss)
        # loss += loss_term2.squeeze()
        # print('loss+2:', loss)
        # loss += loss_term3.squeeze()
        # print('loss+3:', loss)

        return loss

class PhysinformedNet_single(NeuralNet):
    def __init__(self, *args, pre_weight='False',
                 num_output, weights, **kwargs):
        super().__init__(*args, **kwargs)
        self.ReLU = nn.ReLU()
        self.ReLU6 = nn.ReLU6()
        # self.criterion = criterion
        self.X = 0
        # self.loss = my_BCEloss
        # self.loss = my_weight_BCEloss
        # self.loss = my_L1_loss
        # self.P = P
        self.loss = my_weight_rmse_CB
        self.pre_weight = pre_weight
        self.weight = weights

    def get_loss(self, y_pred, y_true, X, training=False):
        # print('y_pred:', y_pred.shape)
        # print('y_true:', y_true.shape)
        # print('X:', X.shape)
        # loss_ori = torch.zeros(1).cuda()

        if np.isnan(y_pred.cpu().detach().numpy().sum()):

            pass

            # import ipdb;ipdb.set_trace()
        '''
        print('BCE value:', self.loss(y_pred.type(torch.FloatTensor).cuda(),
                                      y_true.type(torch.FloatTensor).cuda()))
        '''        
        # import ipdb; ipdb.set_trace()
             
        loss_ori = self.loss(y_pred=y_pred.type(torch.FloatTensor).cuda(),
                             y=y_true.type(torch.FloatTensor).cuda(),
                             pre_weight=self.pre_weight,
                             # P = self.P,
                             weight=self.weight)
        # loss_ori += loss_ori_t

        # loss_ori = loss_ori

        return loss_ori


def my_weight_rmse_CB(y_pred, y, thres=0.5,
                      pre_weight='False'):

    loss = 0
    eps = 1e-7
    # weight = np.ones(y.shape[1])/y.shape[1]

    # import ipdb;ipdb.set_trace()
    resi = y_pred.cpu() - y
    resi = resi**2

    # a = l1_ratio*alpha
    # b = (1-l1_ratio)*alpha

    for i in range(y.shape[1]):
        idx_pos = torch.where(y[:, i] >= thres)[0]
        idx_neg = torch.where(y[:, i] < thres)[0]
        beta = 0.9999
        # import ipdb;ipdb.set_trace()
        if len(idx_pos) != 0:
            E_pos = (1-beta**len(idx_pos))/(1-beta) 
            MSE_pos = torch.sum(resi[idx_pos, i])/E_pos
            loss += MSE_pos
        if len(idx_neg) != 0:
            E_neg = (1-beta**len(idx_neg))/(1-beta) 
            # CE_neg = -1*torch.mean(torch.log(1 - y_pred[idx_neg, i] - eps))
            MSE_neg = torch.sum(resi[idx_neg, i])/E_neg
            loss += MSE_neg
    # loss += a*np.sum(np.abs(weight))

    return loss

def my_weight_rmse2(weight, y_pred, y, pre_weight='False'):

    loss = 0
    eps = 1e-7
    # weight = np.ones(y.shape[1])/y.shape[1]

    if pre_weight:
        weight = np.logspace(2, 0, num=y.shape[1])
        # conv_point = weight[-10]
        weight[-1:] = 0/y.shape[1]
        weight = norm_1d(weight)

    # weights = torch.from_numpy(weight).type(torch.FloatTensor).cuda()
    resi = y_pred - y
    resi = resi**2

    # import ipdb;ipdb.set_trace()
    # P = torch.cos(0.9*np.pi*(y - 0.5))
    
    # resi = resi*P
    for i in range(y.shape[1]):
        loss += torch.mean(resi[:, i])*weight[i]

    '''
    for i in range(y.shape[1]):
        idx_pos = torch.where(y[:, i] >= 0.5)[0]
        idx_neg = torch.where(y[:, i] < 0.5)[0]
        beta = 0.9999
        # import ipdb;ipdb.set_trace()
        if len(idx_pos) != 0:
            E_pos = (1-beta**len(idx_pos))/(1-beta) 
            MSE_pos = torch.sum(resi[idx_pos, i])/E_pos
            loss += MSE_pos*weight[i]/2
        if len(idx_neg) != 0:
            E_neg = (1-beta**len(idx_neg))/(1-beta) 
            # CE_neg = -1*torch.mean(torch.log(1 - y_pred[idx_neg, i] - eps))
            MSE_neg = torch.sum(resi[idx_neg, i])/E_neg
            loss += MSE_neg*weight[i]/2
    '''
    
    return loss


def norm_1d(data):
    return data/data.sum()

class MLP(nn.Module):
    def __init__(self, length, dropout):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(length, 64),
            nn.ReLU(),
            # nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 16),
            nn.ReLU(),
            # nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        # convert tensor (128, 1, 28, 28) --> (128, 1*28*28)
        x = x.view(x.size(0), -1)
        x = self.layers(x)
        return x
    
        
class PhysinformedNet_AR_cpu(NeuralNetRegressor):
    def __init__(self,  *args, beta, 
                 **kwargs):
        super().__init__(*args, **kwargs)
        # self.ReLU = torch.nn.ReLU()
        # self.ReLU6 = torch.nn.ReLU6()
        self.beta = beta
        # self.d = d

    def get_loss(self, y_pred, y_true, X, training=False):
        
        # import ipdb;ipdb.set_trace()
        d = y_true[:, 1] - y_true[:, 0]
        # d = (d - d.mean())/d.std()

        # d = d.cuda()
        N = d.shape[0]
        sigma = torch.exp(y_pred).squeeze()
        
        x = torch.zeros(sigma.shape[0])
        CRPS = torch.zeros(sigma.shape[0])
        RS = torch.zeros(sigma.shape[0])
        
        for i in range(N):
            x[i] = d[i]/np.sqrt(2)/sigma[i]
            
        # import ipdb;ipdb.set_trace()
        
        ind = torch.argsort(x)
        ind_orig = torch.argsort(ind)+1
        
        # x = x.cuda()
        # import ipdb;ipdb.set_trace()


        def AR(i):
            
            CRPS = sigma[i]*(np.sqrt(2)*x[i]*erf(x)[i]
                + np.sqrt(2/np.pi)*torch.exp(-x[i]**2) 
                - 1/np.sqrt(np.pi))

            # import ipdb;ipdb.set_trace()
            RS = N*(x[i]/N*(erf(x)[i]+1) - 
                x[i]*(2*ind_orig[i]-1)/N**2 + 
                torch.exp(-x[i]**2)/np.sqrt(np.pi)/N)

        
        for i in range(N):
                    
            CRPS[i] = sigma[i]*(np.sqrt(2)*x[i]*erf(x)[i]
                + np.sqrt(2/np.pi)*torch.exp(-x[i]**2) 
                - 1/np.sqrt(np.pi))

            # import ipdb;ipdb.set_trace()
            RS[i] = N*(x[i]/N*(erf(x)[i]+1) - 
                x[i]*(2*ind_orig[i]-1)/N**2 + 
                torch.exp(-x[i]**2)/np.sqrt(np.pi)/N)
        # loss_CB = 
        
    
        loss = self.beta*CRPS+(1-self.beta)*RS
        loss = torch.nanmean(loss)
                
        return loss

class PhysinformedNet_AR(NeuralNetRegressor):
    
    def __init__(self,  *args, beta, 
                 mean, std, CRPS_min, RS_min,
                 **kwargs):
        super().__init__(*args, **kwargs)
        # self.ReLU = torch.nn.ReLU()
        # self.ReLU6 = torch.nn.ReLU6()
        self.beta = beta
        self.CRPS_min = CRPS_min
        self.RS_min = RS_min
        self.mean = mean
        self.std = std
        # self.d = d

    def get_loss(self, y_pred, y_true, X, training=False):
        
        # import ipdb;ipdb.set_trace()
        d = y_true[:, 1] - y_true[:, 0]
        # d = (d - self.mean)/self.std
        # y_pred = (y_pred - self.mean)/self.std

        # d = d.cuda()
        N = d.shape[0]
        sigma = torch.exp(y_pred).squeeze().to(self.device)
        
        x = torch.zeros(sigma.shape[0])
        CRPS = torch.zeros(sigma.shape[0])
        RS = torch.zeros(sigma.shape[0])
        
        for i in range(N):
            x[i] = d[i]/np.sqrt(2)/sigma[i]
            
        # import ipdb;ipdb.set_trace()
        
        ind = torch.argsort(x)
        ind_orig = torch.argsort(ind)+1
        ind_orig = ind_orig.to(self.device)
        x = x.to(self.device)
        CRPS_1 = np.sqrt(2)*x*erf(x)
        CRPS_2 = np.sqrt(2/np.pi)*torch.exp(-x**2) 
        CRPS_3 = 1/torch.sqrt(torch.tensor(np.pi))
        # CRPS_1 = CRPS_1.to(self.device)
        # CRPS_2 = CRPS_2.to(self.device)
        CRPS_3 = CRPS_3.to(self.device)
                  
        CRPS = sigma*(CRPS_1 + CRPS_2 - CRPS_3)

        # import ipdb;ipdb.set_trace()
        RS = N*(x/N*(erf(x)+1) - 
            x*(2*ind_orig-1)/N**2 + 
            torch.exp(-x**2)/np.sqrt(np.pi)/N)
        
        RS = RS.to(self.device)
        # import ipdb;ipdb.set_trace()

        # for i in range(N):
                    
        #     CRPS[i] = sigma[i]*(np.sqrt(2)*x[i]*erf(x)[i]
        #         + np.sqrt(2/np.pi)*torch.exp(-x[i]**2) 
        #         - 1/np.sqrt(np.pi))

        #     # import ipdb;ipdb.set_trace()
        #     RS[i] = N*(x[i]/N*(erf(x)[i]+1) - 
        #         x[i]*(2*ind_orig[i]-1)/N**2 + 
        #         torch.exp(-x[i]**2)/np.sqrt(np.pi)/N)
        # # loss_CB = 
        
        # beta = RS_min/(RS_min+CRPS_min)
        # 1-beta = CRPS_min/(RS_min+CRPS_min)
        # loss = self.beta*CRPS+(1-self.beta)*RS
        loss = CRPS/self.CRPS_min+RS/self.RS_min
        loss = torch.mean(loss)
                
        return loss


class PhysinformedNet_AR_2D(NeuralNetRegressor):
    
    def __init__(self,  *args, beta, 
                 mean, std, CRPS_min, RS_min,
                 **kwargs):
        super().__init__(*args, **kwargs)
        # self.ReLU = torch.nn.ReLU()
        # self.ReLU6 = torch.nn.ReLU6()
        self.beta = beta
        self.CRPS_min = CRPS_min
        self.RS_min = RS_min
        self.mean = mean
        self.std = std
        # self.d = d

    def get_loss(self, y_pred, y_true, X, training=False):
        
        # import ipdb;ipdb.set_trace()
        d = y_true[:, :, 1] - y_true[:, :, 0]
        d = d.to(self.device)

        N = d.shape[0]
        sigma = torch.exp(y_pred).squeeze().to(self.device)
        
        x = torch.zeros(sigma.shape)
        CRPS = torch.zeros(sigma.shape)
        RS = torch.zeros(sigma.shape)
        loss = 0
        
        x = d/sigma
        x = x/np.sqrt(2)
        x = x.to(self.device)
        
        for idx in range(x.shape[1]):
            x_t = x[:, idx]
            ind = torch.argsort(x_t)
            ind_orig = torch.argsort(ind)+1
            ind_orig = ind_orig.to(self.device)
            CRPS_1 = np.sqrt(2)*x_t*erf(x_t)
            CRPS_2 = np.sqrt(2/np.pi)*torch.exp(-x_t**2) 
            CRPS_3 = 1/torch.sqrt(torch.tensor(np.pi))
            CRPS_3 = CRPS_3.to(self.device)
            # st()
            CRPS = sigma[:, idx]*(CRPS_1 + CRPS_2 - CRPS_3)

            # import ipdb;ipdb.set_trace()
            RS = N*(x_t/N*(erf(x_t)+1) - 
                x_t*(2*ind_orig-1)/N**2 + 
                torch.exp(-x_t**2)/np.sqrt(np.pi)/N)
        
            RS = RS.to(self.device)
            # st()
            loss += torch.mean(CRPS/self.CRPS_min+RS/self.RS_min)
              
        return loss/x.shape[1]