import numpy as np 
import os
import matplotlib.pyplot as plt 
import h5py 
import dataprep 
import pandas as pd 

# plot P-wave, S-wave, Noise examples 
srcdir = os.path.dirname(os.path.abspath(__file__))
datadir = os.path.dirname(os.path.join(srcdir, "../data/"))
maindir = os.path.abspath(os.path.join(srcdir, ".."))
figdir = os.path.join(maindir, "FIGURES")

csvfile = os.path.join(datadir, "merge.csv")
h5path = os.path.join(datadir, "merge.hdf5")
df = pd.read_csv(csvfile)
EQdf = df[df['trace_category'] == "earthquake_local"]
EQdf["snr_db"] = np.nanmedian(dataprep.extract_snr(EQdf["snr_db"].values),axis=0)
EQdf = EQdf.sort_values("snr_db",ascending=True)
fl = h5py.File(h5path, 'r')

# get location of highest snr 
ID = EQdf.iloc[900900]["trace_name"]
dataset = fl.get('data/'+str(ID))
data = np.array(dataset).transpose()              
p_start = int(dataset.attrs['p_arrival_sample'])
s_start = int(dataset.attrs['s_arrival_sample'])
snr = dataset.attrs['snr_db']
data = dataprep.highpass(data,0.5,100.)
fl.close()

# plot three classes together 
startat = 500
numsamples = 2000
fs = 100.
freq = 2. 
yoffset = 7500 
ytext = 1000
t = np.linspace(0,numsamples/fs,numsamples)
fig, ax = plt.subplots(figsize=(6,3.75))
ax.plot(t,data[0,startat:startat+numsamples],c="k",alpha=0.85)
ax.plot(t,data[1,startat:startat+numsamples] - yoffset,c="k",alpha=0.85)
ax.plot(t,data[2,startat:startat+numsamples] - 2 * yoffset,c="k",alpha=0.85)
ax.axvline([(p_start - startat) / fs],c="red",linewidth=1)
ax.axvline([(s_start - startat) / fs],c="blue",linewidth=1)
ax.text(0.2,ytext,"E",fontsize=18)
ax.text(0.2,ytext - yoffset,"N",fontsize=18)
ax.text(0.2,ytext - 2 * yoffset,"Z",fontsize=18)
# ax.axvspan((startat+50) / fs,(startat+250) / fs ,alpha=0.5, color="grey",label="Noise") 
ax.axvspan((p_start-startat-100) / fs,(p_start - startat+100) / fs ,alpha=0.5, color="red",label="P-wave") 
ax.axvspan((s_start-startat-100) / fs,(s_start - startat+100) / fs ,alpha=0.5, color="blue",label="S-wave")
ax.set_xlabel("Time [s]",fontsize=18)
ax.set_yticks([], [])
ax.tick_params(axis='both', which='major', labelsize=12)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.set_xlim([0.,20.])
ax.tick_params(direction='in')
plt.tight_layout()
fig.savefig(os.path.join(figdir, "all3.pdf"))
plt.close()

pwave = data[:,p_start-100:p_start+100].copy()
swave = data[:,s_start-100:s_start+100].copy()
noise = data[:,50:250].copy()
pwave = pwave / np.max(np.abs(pwave))
swave = swave / np.max(np.abs(swave))
noise = noise / np.max(np.abs(noise))

# plot all three classes normalized to same amplitude 
zoom_t = np.linspace(0,1.99,200)
yticks = [-1,0,1]
fig, (ax1, ax2, ax3) = plt.subplots(nrows=3)
ax1.plot(zoom_t,noise[0,:],c="grey",alpha=0.5,linewidth = 2)
ax1.set_title("Noise               ",fontsize=20,color="Grey")
ax1.set_xlim(xmin=0)
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
# Only show ticks on the left and bottom spines
ax1.yaxis.set_ticks_position('left')
ax1.xaxis.set_ticks_position('bottom')
ax1.set_xticks([], [])
ax1.tick_params(labelsize=12)
ax1.yaxis.set_ticks(yticks)
ax1.tick_params(direction='in')
ax1.spines['bottom'].set_bounds(min(zoom_t), max(zoom_t))
ax1.set_xlim([-0.05, 2.])
ax2.plot(zoom_t,pwave[2,:],c="red",alpha=0.5,linewidth = 2)
ax2.set_title("P-wave               ",fontsize=20,color="red")
ax2.set_ylabel("Normalized Amplitude",fontsize=14)
ax2.set_xlim(xmin=0)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.spines['bottom'].set_visible(False)
# Only show ticks on the left and bottom spines
ax2.yaxis.set_ticks_position('left')
ax2.xaxis.set_ticks_position('bottom')
ax2.set_xticks([], [])
ax2.yaxis.set_ticks(yticks)
ax2.tick_params(labelsize=12)
ax2.tick_params(direction='in')
ax2.spines['bottom'].set_bounds(min(zoom_t), max(zoom_t))
ax2.set_xlim([-0.05, 2.])
ax3.plot(zoom_t,swave[0,:],c="blue",alpha=0.5,linewidth = 2)
ax3.set_title("S-wave               ",fontsize=20,color="blue")
ax3.set_xlabel("Time [s]",fontsize=18)
ax3.set_xlim(xmin=0)
ax3.spines['right'].set_visible(False)
ax3.spines['top'].set_visible(False)
# Only show ticks on the left and bottom spines
ax3.yaxis.set_ticks_position('left')
ax3.xaxis.set_ticks_position('bottom')
ax3.tick_params(labelsize=12)
ax3.yaxis.set_ticks(yticks)
ax3.set_ylim([-1.25, 1])
ax3.spines['left'].set_bounds(-1, 1)
ax3.tick_params(direction='in')
ax3.spines['bottom'].set_bounds(0,2.1)
ax3.set_xlim([-0.05, 2.])
plt.tight_layout()
plt.savefig(os.path.join("classes.pdf"))
plt.close()
