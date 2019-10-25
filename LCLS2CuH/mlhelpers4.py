import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler,MinMaxScaler
import keras
from keras.models import Sequential
from keras.layers import Dense, GaussianNoise
from scipy.constants import physical_constants
#from rsbeams.rsdata.SDDS import readSDDS
from SDDS import readSDDS
from IPython.display import clear_output
m_e = physical_constants["electron mass energy equivalent in MeV"][0]

# define class for showing training plot - found online
class PlotLosses(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        
        self.fig = plt.figure()
        
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1
        
        clear_output(wait=True)
        plt.plot(self.x, self.losses, label="loss")
        plt.plot(self.x, self.val_losses, label="val_loss")
        plt.legend()
        plt.ylabel('error')
        plt.xlabel('epoch')
        plt.show();
        
plot_losses = PlotLosses()

def plot_a_bunch_of_beams(h3,bunches,nbins):
    plt.figure(figsize=(15,15))
    plt.subplot(1,6,1)
    idx=np.random.randint(0,bunches.shape[0])
    plt.imshow(h3[idx,:].reshape(nbins,nbins))
    plt.subplot(1,6,2)
    idx=np.random.randint(0,bunches.shape[0])
    plt.imshow(h3[idx,:].reshape(nbins,nbins))
    plt.subplot(1,6,3)
    idx=np.random.randint(0,bunches.shape[0])
    plt.imshow(h3[idx,:].reshape(nbins,nbins))
    plt.subplot(1,6,4)
    idx=np.random.randint(0,bunches.shape[0])
    plt.imshow(h3[idx,:].reshape(nbins,nbins))
    plt.subplot(1,6,5)
    idx=np.random.randint(0,bunches.shape[0])
    plt.imshow(h3[idx,:].reshape(nbins,nbins))
    plt.subplot(1,6,6)
    idx=np.random.randint(0,bunches.shape[0])
    plt.imshow(h3[idx,:].reshape(nbins,nbins))
    plt.show()
    plt.figure(figsize=(15,15))
    plt.subplot(1,6,1)
    idx=np.random.randint(0,bunches.shape[0])
    plt.imshow(h3[idx,:].reshape(nbins,nbins))
    plt.subplot(1,6,2)
    idx=np.random.randint(0,bunches.shape[0])
    plt.imshow(h3[idx,:].reshape(nbins,nbins))
    plt.subplot(1,6,3)
    idx=np.random.randint(0,bunches.shape[0])
    plt.imshow(h3[idx,:].reshape(nbins,nbins))
    plt.subplot(1,6,4)
    idx=np.random.randint(0,bunches.shape[0])
    plt.imshow(h3[idx,:].reshape(nbins,nbins))
    plt.subplot(1,6,5)
    idx=np.random.randint(0,bunches.shape[0])
    plt.imshow(h3[idx,:].reshape(nbins,nbins))
    plt.subplot(1,6,6)
    idx=np.random.randint(0,bunches.shape[0])
    plt.imshow(h3[idx,:].reshape(nbins,nbins))
    plt.show()
    plt.figure(figsize=(15,15))
    plt.subplot(1,6,1)
    idx=np.random.randint(0,bunches.shape[0])
    plt.imshow(h3[idx,:].reshape(nbins,nbins))
    plt.subplot(1,6,2)
    idx=np.random.randint(0,bunches.shape[0])
    plt.imshow(h3[idx,:].reshape(nbins,nbins))
    plt.subplot(1,6,3)
    idx=np.random.randint(0,bunches.shape[0])
    plt.imshow(h3[idx,:].reshape(nbins,nbins))
    plt.subplot(1,6,4)
    idx=np.random.randint(0,bunches.shape[0])
    plt.imshow(h3[idx,:].reshape(nbins,nbins))
    plt.subplot(1,6,5)
    idx=np.random.randint(0,bunches.shape[0])
    plt.imshow(h3[idx,:].reshape(nbins,nbins))
    plt.subplot(1,6,6)
    idx=np.random.randint(0,bunches.shape[0])
    plt.imshow(h3[idx,:].reshape(nbins,nbins))
    plt.show()
    
# make dataset
def make_dataset(bunches,nbins=150):

    #prefill matrices
    xedges=np.empty((bunches.shape[0],nbins))
    yedges=np.empty((bunches.shape[0],nbins))
    h3=np.empty((bunches.shape[0],nbins*nbins))
    xprof=np.empty((bunches.shape[0],nbins))
    yprof=np.empty((bunches.shape[0],nbins))
    xedges=np.empty((bunches.shape[0],nbins+1))
    yedges=np.empty((bunches.shape[0],nbins+1))
    xedgesb=np.empty((bunches.shape[0],nbins+1))
    yedgesb=np.empty((bunches.shape[0],nbins+1))

    #get set up to decide ROI
    for i in range(0,bunches.shape[0]):
        h,xedges[i,:],yedges[i,:]=np.histogram2d((bunches[i,:,4] - np.average(bunches[i,:,4]))*1e12,bunches[i,:,5]*m_e, bins=nbins);
        h3[i,:]=np.flipud(h.transpose()).flatten()
        xprof[i,:]=np.sum(np.flipud(h.transpose()),axis=0)
        yprof[i,:]=np.sum(np.flipud(h.transpose()),axis=1)

    #make dataset with fixed ROI
    for i in range(0,bunches.shape[0]):
        #h,xedgesb[i,:],yedgesb[i,:]=np.histogram2d((bunches[i,:,4] - np.average(bunches[i,:,4]))*1e12,bunches[i,:,5]*m_e,range=([[np.min(xedges), np.max(xedges)],[np.min(yedges), np.max(yedges)]]),bins=nbins);
        h,xedgesb[i,:],yedgesb[i,:]=np.histogram2d((bunches[i,:,4] - np.average(bunches[i,:,4]))*1e12,bunches[i,:,5]*m_e,range=([[np.mean(xedges)-4*np.std(xedges), np.mean(xedges)+4*np.std(xedges)],[np.mean(yedges)-4*np.std(yedges), np.mean(yedges)+4*np.std(yedges)]]),bins=nbins);

        h3[i,:]=np.flipud(h.transpose()).flatten()
        xprof[i,:]=np.sum(np.flipud(h.transpose()),axis=0)
        yprof[i,:]=np.sum(np.flipud(h.transpose()),axis=1)

    return h3,xprof,yprof,xedgesb[0,:],yedgesb[0,:]