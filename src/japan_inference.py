import dataprep
import glob
import os 
import obspy 
import matplotlib.pyplot as plt 
import numpy as np 
import seaborn as sns
import sklearn.metrics
import tensorflow as tf
import tensorflow.keras as keras 

"""
This script performs inference on data from the K-NET & KiK-net networks. 

Steps to reproduce: 

1) Download an event from https://www.kyoshin.bosai.go.jp/kyoshin/quake/index_en.html
2) Untar downloaded .tar file (tar -xvf file.tar)
3) gunzip .knt.tar.gz and .kik.tar.gz files
4) untar .knt.tar and .kik.tar files 
5) Move all .EW, .NS, .UD files to ../data/JAPAN-TEST/WAVES/
6) Run with python src/japan_inference.py

This script uses event 20220502222100 as an example. 
"""

def station_to_tensor(st): 
    'Convert obspy.Stream to tf.Tensor'

    # slide through in moving windows 
    starttime = []
    endtime = []
    Xs = [] 
    for win in st.slide(window_length=1.975, step=0.5, offset=2.0):
        starttime.append(win[0].stats.starttime - st[0].stats.knet["evot"])
        endtime.append(win[0].stats.endtime - st[0].stats.knet["evot"])
        Xs.append(np.vstack([win[0].data,win[1].data,win[2].data]).T.astype(np.float32))
    Xs = tf.stack(Xs)[:,:,:,tf.newaxis]
    starttime = np.array(starttime)
    endtime = np.array(endtime)
    return Xs, starttime, endtime

def model_inference(model, X): 
    'Evalute TinyQuake Model on input data'
    Nsteps = X.shape[0]
    preds = np.zeros((Nsteps, 3))
    model.allocate_tensors()
    model_input_index = model.get_input_details()[0]["index"]
    model_output_index = model.get_output_details()[0]["index"]
    for ii in range(Nsteps): 
        model.set_tensor(model_input_index, X[ii:ii+1])
        model.invoke()
        preds[ii,:] = model.get_tensor(model_output_index)
        
    return preds

if __name__ == "__main__": 

    # load test data 
    srcdir = os.path.dirname(os.path.abspath(__file__))
    datadir = os.path.dirname(os.path.join(srcdir, "../data/"))
    maindir = os.path.abspath(os.path.join(srcdir, ".."))
    figdir = os.path.join(maindir, "FIGURES")

    # check that models work on test data 
    maindir = os.path.abspath(os.path.join(srcdir, ".."))
    CNNDIR = os.path.join(maindir, "CNN")
    CNNMODELDIR = os.path.join(CNNDIR,"models")
    model_name = os.path.join(CNNMODELDIR, "model_17_16_32_32.tflite")
    model = tf.lite.Interpreter(model_name)

    # load Japanese test data 
    japandir = os.path.join(datadir, "JAPAN-TEST/WAVES/")

    # apply minimal processing 
    st = obspy.read(japandir + "*")
    st.detrend(type="linear")
    st.taper(max_percentage=0.05, max_length=2.0)
    st.filter(type="highpass", freq=0.5)
    st.resample(40.0, window="hann")
    st.sort(["starttime"])

    # run through each set of stations 
    stations = np.array([st[ii].stats.station for ii in range(len(st))])
    stations_unique = np.unique(stations)
    event_distances = np.zeros(len(stations_unique))
    network_preds = {}
    for (ii, station) in enumerate(stations_unique): 
        ind = np.where(stations == station)[0]
        if len(ind) != 3: 
            continue
        stat = st[ind[0]:ind[-1]+1]
        knet = stat[0].stats.knet
        event_distance, az, baz = obspy.geodetics.base.calc_vincenty_inverse(knet["evla"], knet["evlo"], knet["stla"], knet["stlo"])
        event_distance /= 1000
        event_distances[ii] = event_distance
        X, starttime, endtime = station_to_tensor(stat)
        preds = model_inference(model, X)
        station_preds = {"knet":knet, "event_distance":event_distance, "starttime":starttime, "endtime":endtime, "preds":preds}
        network_preds[station] = station_preds

    # plot predictions with moveout plot 
    distance_ind = np.argsort(event_distances)
    stations_unique = stations_unique[distance_ind]
    event_distances = event_distances[distance_ind]

    fig, ax = plt.subplots(figsize=(9,6))
    for (ii, station) in enumerate(stations_unique):
        # if ii % 2 != 0:
        #     continue  
        ind = np.where(stations == station)[0][-1]
        tr = st[ind]
        t = tr.times(type="utcdatetime")[80:] - tr.stats.knet["evot"]
        data = tr.data[80:] * tr.stats.calib * 981 
        data /= np.abs(data).max()
        data *= 5
        ax.plot(t, data + event_distances[ii], c="black", alpha=0.25, linewidth=0.5)
        
        # check for P-waves & S-waves
        starttime = network_preds[station]["starttime"]
        endtime = network_preds[station]["endtime"]
        preds = network_preds[station]["preds"]
        indP = np.where(preds[:,0] > 0.75)[0]
        indS = np.where(preds[:,1] > 0.75)[0]

        # plot S-waves
        if len(indS > 0):
            for jj in range(len(indS)):
                t_s_ind = np.where((t >= starttime[indS[jj]]) & (t <= endtime[indS[jj]]))[0]
                ax.plot(t[t_s_ind], data[t_s_ind] + event_distances[ii], c="blue", alpha=0.5, linewidth=0.5)

        # plot P-waves 
        if len(indP > 0):
            for jj in range(len(indP)):
                t_p_ind = np.where((t >= starttime[indP[jj]]) & (t <= endtime[indP[jj]]))[0]
                ax.plot(t[t_p_ind], data[t_p_ind] + event_distances[ii], c="red", alpha=0.5, linewidth=0.5)

    # plot non-appearing lines for P/S 
    ax.plot([1000,1001], [1000,1001], color="red", label="P-wave detection", linewidth=2)
    ax.plot([1000,1001], [1000,1001], color="blue", label="S-wave detection", linewidth=2)
    ax.set_xlim([-11,70])
    ax.set_ylim([-5,255])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_bounds(0, 250)
    ax.tick_params(direction='in')
    ax.spines['bottom'].set_bounds(-10,70)
    ax.legend(loc="best",fontsize=14)
    ax.set_xlabel("Time relative to origin [s]",fontsize=18)
    ax.set_ylabel("Epicentral Distance [km]",fontsize=18)
    ax.tick_params(labelsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(figdir, "japan-inference.pdf"))
    plt.close()