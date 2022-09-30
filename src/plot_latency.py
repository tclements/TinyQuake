import numpy as np 
import os
import scipy 
import matplotlib.pyplot as plt 
import pickle 

# plot accuracy vs. latency 
srcdir = os.path.dirname(os.path.abspath(__file__))
maindir = os.path.abspath(os.path.join(srcdir, ".."))
figdir = os.path.join(maindir, "FIGURES")

# read accuracies for DNN models 
DNNlatency = np.array([1,1,1,2,4,9,23,1,1,5])
with open(os.path.join(maindir, "DNN/DNN_models.pkl"), 'rb') as f:
    DNNmodels = pickle.load(f)
Nmodels = len(DNNmodels)
DNNacc = np.zeros(Nmodels)
DNNparams = np.zeros(Nmodels)

for ii in range(Nmodels): 
    DNNacc[ii] = DNNmodels[ii]["report"]["accuracy"]
    DNNparams[ii] = DNNmodels[ii]["params"]

CNNlatency = np.array([5,6,9,12,19,26,45,67,120,194,5,6,9,13,20,31,49,84,136,258])
with open(os.path.join(maindir, "CNN/CNN_models.pkl"), 'rb') as f:
    CNNmodels = pickle.load(f)
Nmodels = len(CNNmodels)
CNNacc = np.zeros(Nmodels)
CNNparams = np.zeros(Nmodels)

for ii in range(Nmodels): 
    CNNacc[ii] = CNNmodels[ii]["report"]["accuracy"]
    CNNparams[ii] = CNNmodels[ii]["params"]

RNNlatency = np.array([10,17,39,56,115,252,20,32,58,113,237])
with open(os.path.join(maindir, "RNN/RNN_models.pkl"), 'rb') as f:
    RNNmodels = pickle.load(f)
Nmodels = len(RNNmodels)
RNNacc = np.zeros(Nmodels)
RNNparams = np.zeros(Nmodels)
for ii in range(Nmodels): 
    RNNacc[ii] = RNNmodels[ii]["report"]["accuracy"]
    RNNparams[ii] = RNNmodels[ii]["params"]

# plot latency vs accuracy 
fig, ax = plt.subplots(figsize=(6,6))
ax.axvline([0.1],c="grey",linestyle="--",linewidth=2,alpha=0.75)
ax.scatter(DNNlatency / 1000,DNNacc,100,alpha=0.85,label="DNN",c="dodgerblue",edgecolor="k")
ax.scatter(RNNlatency / 1000,RNNacc,80,alpha=0.85,label="RNN",c="chartreuse",edgecolor="k",marker="D")
ax.scatter(CNNlatency / 1000,CNNacc,100,alpha=0.85,label="CNN",c="crimson",edgecolor="k",marker="^")
ax.text(0.15, 0.67, "100 milliseconds",c="grey", fontsize=18,rotation='vertical')
ax.set_xscale("log")
ax.set_xlabel("On-device Latency [s]",fontsize=18)
ax.set_ylabel("Accuracy",fontsize=18)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_ylim([0.595, 0.93])
ax.set_xlim([1e-4, 1])
ax.spines['left'].set_bounds(0.60, 0.93)
ax.tick_params(direction='in')
ax.spines['bottom'].set_bounds(1e-4,1)
ax.legend(loc="best",fontsize=14)
ax.tick_params(labelsize=12)
plt.tight_layout()
plt.savefig(os.path.join(figdir, "accuracy-vs-latency.pdf"))
plt.close()



