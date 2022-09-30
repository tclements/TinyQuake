import numpy as np 
import os
import matplotlib.pyplot as plt 
import pickle 

# plot number of parameters in DNN, CNN, RNN
srcdir = os.path.dirname(os.path.abspath(__file__))
maindir = os.path.abspath(os.path.join(srcdir, ".."))
figdir = os.path.join(maindir, "FIGURES")


with open(os.path.join(maindir, "DNN/DNN_models.pkl"), 'rb') as f:
    DNNmodels = pickle.load(f)
Nmodels = len(DNNmodels)
DNNacc = np.zeros(Nmodels)
DNNsize = np.zeros(Nmodels)
DNNparams = np.zeros(Nmodels)

for ii in range(Nmodels): 
    DNNacc[ii] = DNNmodels[ii]["report"]["accuracy"]
    DNNsize[ii] = DNNmodels[ii]["model_header_size"]
    DNNparams[ii] = DNNmodels[ii]["params"]

with open(os.path.join(maindir, "CNN/CNN_models.pkl"), 'rb') as f:
    CNNmodels = pickle.load(f)
Nmodels = len(CNNmodels)
CNNacc = np.zeros(Nmodels)
CNNsize = np.zeros(Nmodels)
CNNparams = np.zeros(Nmodels)

for ii in range(Nmodels): 
    CNNacc[ii] = CNNmodels[ii]["report"]["accuracy"]
    CNNsize[ii] = CNNmodels[ii]["model_header_size"]
    CNNparams[ii] = CNNmodels[ii]["params"]

with open(os.path.join(maindir, "RNN/RNN_models.pkl"), 'rb') as f:
    RNNmodels = pickle.load(f)
Nmodels = len(RNNmodels)
RNNacc = np.zeros(Nmodels)
RNNsize = np.zeros(Nmodels)
RNNparams = np.zeros(Nmodels)
for ii in range(Nmodels): 
    RNNacc[ii] = RNNmodels[ii]["report"]["accuracy"]
    RNNsize[ii] = RNNmodels[ii]["model_header_size"]
    RNNparams[ii] = RNNmodels[ii]["params"]

# plot accuracy vs trainable parameters 
fig, ax = plt.subplots(figsize=(6,6))
ax.scatter(DNNparams,DNNacc,100,alpha=0.85,label="DNN",c="dodgerblue",edgecolor="k")
ax.scatter(RNNparams,RNNacc,80,alpha=0.85,label="RNN",c="chartreuse",edgecolor="k",marker="D")
ax.scatter(CNNparams,CNNacc,100,alpha=0.85,label="CNN",c="crimson",edgecolor="k",marker="^")
ax.set_xscale("log")
ax.set_xlabel("Trainable Parameters",fontsize=18)
ax.set_ylabel("Accuracy",fontsize=18)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlim([1e2, 1e5])
ax.set_ylim([0.595, 0.93])
ax.spines['left'].set_bounds(0.60, 0.93)
ax.tick_params(direction='in')
ax.spines['bottom'].set_bounds(1e2,1e5)
ax.legend(loc="lower right",fontsize=14)
ax.tick_params(labelsize=12)
plt.tight_layout()
plt.savefig(os.path.join(figdir, "accuracy-vs-params.pdf"))
plt.close()


