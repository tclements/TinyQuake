import numpy as np 
import os
import scipy 
import matplotlib.pyplot as plt 
import pickle 

srcdir = os.path.dirname(os.path.abspath(__file__))
maindir = os.path.abspath(os.path.join(srcdir, ".."))
figdir = os.path.join(maindir, "FIGURES")


# CNNinvoke = [4648,4648,4904,6824,6824,7992,11816,11816,14168,21816,21816,26520,41816,41816,51224]
with open(os.path.join(maindir, "CNN/CNN_models.pkl"), 'rb') as f:
    CNNmodels = pickle.load(f)
Nmodels = len(CNNmodels)
CNNacc = np.zeros(Nmodels)
CNNlat = np.zeros(Nmodels)
CNNparams = np.zeros(Nmodels)
CNNinvoke = np.zeros(Nmodels)

for ii in range(Nmodels): 
    CNNacc[ii] = CNNmodels[ii]["report"]["accuracy"]
    CNNlat[ii] = CNNmodels[ii]["latency"]
    CNNparams[ii] = CNNmodels[ii]["params"]
    CNNinvoke[ii] = CNNmodels[ii]["invoke"]

# RNNinvoke = [4660,4648,4904,7992,6836,6824,7992,14168,11828,11816,14168,26520,21828,21816,26520]
with open(os.path.join(maindir, "RNN/RNN_models.pkl"), 'rb') as f:
    RNNmodels = pickle.load(f)
Nmodels = len(RNNmodels)
RNNacc = np.zeros(Nmodels)
RNNlat = np.zeros(Nmodels)
RNNparams = np.zeros(Nmodels)
RNNinvoke = np.zeros(Nmodels)
for ii in range(Nmodels): 
    RNNacc[ii] = RNNmodels[ii]["report"]["accuracy"]
    RNNlat[ii] = RNNmodels[ii]["latency"]
    RNNparams[ii] = RNNmodels[ii]["params"]
    RNNinvoke[ii] = RNNmodels[ii]["invoke"]

# plot latency vs accuracy 
fig, ax = plt.subplots(figsize=(6,6))
ax.scatter(RNNinvoke ,RNNacc,80,alpha=0.85,label="RNN",c="chartreuse",edgecolor="k",marker="D")
ax.scatter(CNNinvoke,CNNacc,100,alpha=0.85,label="CNN",c="crimson",edgecolor="k",marker="^")
ax.set_xscale('log', basex=2)
ax.set_xlabel("Invoke Memory [bytes]",fontsize=18)
ax.set_ylabel("Accuracy",fontsize=18)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_ylim([0.845, 0.92])
ax.spines['left'].set_bounds(0.85, 0.92)
ax.tick_params(direction='in')
ax.spines['bottom'].set_bounds(2**12.1,2**16)
ax.legend(loc="lower right",fontsize=14)
ax.tick_params(labelsize=12)
plt.tight_layout()
plt.savefig(os.path.join(figdir,"accuracy-vs-invoke.pdf"))
plt.close()



