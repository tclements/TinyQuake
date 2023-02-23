import dataprep
import glob
import os 
import obspy 
from obspy import UTCDateTime
import matplotlib.pyplot as plt 
import numpy as np 
import seaborn as sns
import sklearn.metrics
import tensorflow as tf
import tensorflow.keras as keras 

"""
This script performs inference on data from the K-NET, KiK-net and Community seismic networks. 

Steps to reproduce: 

Japanese data 
1) Download an event from https://www.kyoshin.bosai.go.jp/kyoshin/quake/index_en.html
2) Untar downloaded .tar file (tar -xvf file.tar)
3) gunzip .knt.tar.gz and .kik.tar.gz files
4) untar .knt.tar and .kik.tar files 
5) Move all .EW, .NS, .UD files to ../data/japan/

6) Download an event from http://csn.caltech.edu/data/
7) Unzip data and move .sac files into ../data/ci..
6) Run with python src/japan_inference.py

This script uses event 20080614084300 from Japan and ci38695658 from the CSN as examples. 
"""

def station_to_tensor(st, origin_time): 
    'Convert obspy.Stream to tf.Tensor'

    # slide through in moving windows 
    starttime = []
    endtime = []
    Xs = [] 
    for win in st.slide(window_length=1.975, step=0.5, offset=2.0):
        starttime.append(win[0].stats.starttime - origin_time)
        endtime.append(win[0].stats.endtime - origin_time)
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

def preprocess_event(filepath): 
    st = obspy.read(filepath)
    st.detrend(type="linear")
    st.taper(max_percentage=0.05, max_length=2.0)
    st.filter(type="highpass", freq=0.5)
    st.resample(40.0, window="hann")
    st.sort(["starttime"])
    return st 

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

    ##### load Japanese test data #####
    japandir = os.path.join(datadir, "japan/")

    # apply minimal processing 
    japan_origin = UTCDateTime(2008, 6, 13, 23, 43 ,45.360)
    japan = preprocess_event(japandir + "*")

    # run through each set of stations 
    japan_stations = np.array([tr.stats.station for tr in japan])
    japan_unique_stations = np.unique(japan_stations)
    japan_distances = np.zeros(len(japan_unique_stations))
    japan_preds = {}
    for (ii, station) in enumerate(japan_unique_stations): 
        ind = np.where(japan_stations == station)[0]
        if len(ind) != 3: 
            print(station)
            continue
        stat = japan[ind[0]:ind[-1]+1]
        knet = stat[0].stats.knet
        event_distance, az, baz = obspy.geodetics.base.calc_vincenty_inverse(knet["evla"], knet["evlo"], knet["stla"], knet["stlo"])
        event_distance /= 1000
        japan_distances[ii] = event_distance
        X, starttime, endtime = station_to_tensor(stat, japan_origin)
        preds = model_inference(model, X)
        station_preds = {"knet":knet, "event_distance":event_distance, "starttime":starttime, "endtime":endtime, "preds":preds}
        japan_preds[station] = station_preds

    ##### load CSN data #### 
    csndir = os.path.join(datadir, "ci38695658")
    origin_time = UTCDateTime(2020, 9, 19, 6,38, 46.93)
    event_lat = 34.03800
    event_lon = -118.08000

    csn = preprocess_event(os.path.join(csndir, "*.sac"))
    csn = csn.slice(origin_time - 3, origin_time + 20)

    # convert data from g to cm/s^2 
    for tr in csn: 
        tr.data *= 981

    # run through each set of stations  
    csn_stations = np.array([tr.stats.station for tr in csn])
    csn_unique_stations = np.unique(csn_stations)
    csn_distances = np.zeros(len(csn_unique_stations))
    csn_SNR = np.zeros(len(csn_unique_stations))
    csn_preds = {}
    for (ii, station) in enumerate(csn_unique_stations): 
        ind = np.where(csn_stations == station)[0]
        if len(ind) != 3: 
            continue
        stat = csn[ind[0]:ind[-1]+1]
        for tr in stat:
            t = tr.times(type="utcdatetime") - origin_time
            before_ind = np.where(t <= 0)[0]
            after_ind = np.where(t > 0)[0]
            csn_SNR[ii] += np.sqrt(np.mean(tr.data[after_ind] ** 2)) / np.sqrt(np.mean(tr.data[before_ind]**2))
        csn_SNR[ii] /= 3
        sac = stat[0].stats.sac
        event_distance, az, baz = obspy.geodetics.base.calc_vincenty_inverse(event_lat, event_lon, sac["stla"], sac["stlo"])
        event_distance /= 1000
        csn_distances[ii] = event_distance
        X, starttime, endtime = station_to_tensor(stat, origin_time)
        preds = model_inference(model, X)
        station_preds = {"sac":sac, "event_distance":event_distance, "starttime":starttime, "endtime":endtime, "preds":preds}
        csn_preds[station] = station_preds
    csn_SNR[np.isnan(csn_SNR)] = 0.0

    # plot predictions with moveout plot 
    japan_ind = np.argsort(japan_distances)
    japan_unique_stations = japan_unique_stations[japan_ind]
    japan_distances = japan_distances[japan_ind]

    csn_ind = np.argsort(csn_distances)
    csn_unique_stations = csn_unique_stations[csn_ind]
    csn_distances = csn_distances[csn_ind]
    csn_SNR = csn_SNR[csn_ind]

    # japan travel time 
    TPmodel = TauPyModel(model="ak135")
    source_station_dist = np.arange(0.0,300.0)
    japan_p_travel = np.zeros_like(source_station_dist)
    japan_s_travel = np.zeros_like(source_station_dist)
    for (ii, d) in enumerate(source_station_dist): 
        tt_pred = TPmodel.get_travel_times(
            source_depth_in_km=japan[0].stats.knet["evdp"], 
            distance_in_degree=source_station_dist[ii] / 110.9, 
            phase_list=["p", "s"],
        )
        japan_p_travel[ii] = tt_pred[0].time
        japan_s_travel[ii] = tt_pred[1].time

    # csn travel time 
    csn_p_travel = np.zeros_like(source_station_dist)
    csn_s_travel = np.zeros_like(source_station_dist)
    for (ii, d) in enumerate(source_station_dist): 
        tt_pred = TPmodel.get_travel_times(
            source_depth_in_km=16.9, 
            distance_in_degree=source_station_dist[ii] / 110.9, 
            phase_list=["p", "s"],
        )
        csn_p_travel[ii] = tt_pred[0].time
        csn_s_travel[ii] = tt_pred[1].time

    fig, ax = plt.subplots(figsize=(9.5,6))
    dx = 2.0
    for ii in np.arange(2,200,dx): 
        range_ind = np.where((ii < japan_distances) & (japan_distances < ii + dx))
        if range_ind[0].shape == (0,): 
            continue
        else:
            range_ind = range_ind[0][0]
        station = japan_unique_stations[range_ind]
        ind = np.where(japan_stations == station)[0]
        ENZ = japan[slice(ind[0], ind[-1] + 1)]
        Z = ENZ.select(channel="UD*")[0]
        t = Z.times(type="utcdatetime") - japan_origin 
        data = Z.data
        data /= data.std()
        data *= dx / 5
        ax.plot(t, data + japan_distances[range_ind], c="black", alpha=0.25, linewidth=0.5)

        # check for P-waves & S-waves
        starttime = japan_preds[station]["starttime"]
        endtime = japan_preds[station]["endtime"]
        preds = japan_preds[station]["preds"]
        indP = np.where(preds[:,0] > 0.75)[0]
        indS = np.where(preds[:,1] > 0.75)[0]

        # plot S-waves
        if len(indS > 0):
            for jj in range(len(indS)):
                t_s_ind = np.where((t >= starttime[indS[jj]]) & (t <= endtime[indS[jj]]))[0]
                ax.plot(t[t_s_ind], data[t_s_ind] + japan_distances[range_ind], c="blue", alpha=0.5, linewidth=0.5)

        # plot P-waves 
        if len(indP > 0):
            for jj in range(len(indP)):
                t_p_ind = np.where((t >= starttime[indP[jj]]) & (t <= endtime[indP[jj]]))[0]
                ax.plot(t[t_p_ind], data[t_p_ind] + japan_distances[range_ind], c="red", alpha=0.5, linewidth=0.5)

    ax.plot(japan_p_travel, source_station_dist, color="red", label="P-wave arrival", linewidth=1, linestyle="dashed", alpha=0.75)
    ax.plot(japan_s_travel, source_station_dist, color="blue", label="S-wave arrival", linewidth=1, linestyle="dashdot", alpha=0.75)

    # plot non-appearing lines for P/S 
    ax.plot([1000,1001], [1000,1001], color="red", label="P-wave detection", linewidth=2)
    ax.plot([1000,1001], [1000,1001], color="blue", label="S-wave detection", linewidth=2)
    ax.set_xlim([-1,41])
    ax.set_ylim([-5,121])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_bounds(0, 120)
    ax.tick_params(direction='in')
    ax.spines['bottom'].set_bounds(0,40)
    ax.legend(loc="lower right",fontsize=10, bbox_to_anchor=(0.95, 0.05))
    ax.set_xlabel("Time relative to origin [s]",fontsize=18)
    ax.set_ylabel("Epicentral Distance [km]",fontsize=18)
    ax.tick_params(labelsize=12)
    ax.set_title("2008 M6.9 Iwate-Miyagi Earthquake",fontsize=18)
    plt.savefig(os.path.join(figdir, "japan-inference-revised.pdf"))
    plt.close()

    fig, ax = plt.subplots(figsize=(9.5,6))
    dx = 0.5
    for ii in np.arange(2,40,dx): 
        range_ind = np.where((ii < csn_distances) & (csn_distances < ii + dx))
        if range_ind[0].shape == (0,): 
            continue
        else:
            range_ind = range_ind[0][np.argmax(csn_SNR[range_ind[0]])] 
        station = csn_unique_stations[range_ind]
        ind = np.where(csn_stations == station)[0]
        ENZ = csn[slice(ind[0], ind[-1] + 1)]
        Z = ENZ.select(channel="HNZ")[0]
        t = Z.times(type="utcdatetime") - origin_time 
        data = Z.data
        data /= np.abs(data).max()
        data *= dx
        ax.plot(t, data + csn_distances[range_ind], c="black", alpha=0.25, linewidth=0.5)
        
        # check for P-waves & S-waves
        starttime = csn_preds[station]["starttime"]
        endtime = csn_preds[station]["endtime"]
        preds = csn_preds[station]["preds"]
        indP = np.where(preds[:,0] > 0.9)[0]
        indS = np.where(preds[:,1] > 0.9)[0]

        # plot S-waves
        if len(indS > 0):
            for jj in range(len(indS)):
                t_s_ind = np.where((t >= starttime[indS[jj]]) & (t <= endtime[indS[jj]]))[0]
                ax.plot(t[t_s_ind], data[t_s_ind] + csn_distances[range_ind], c="blue", alpha=0.5, linewidth=0.5)

        # plot P-waves 
        if len(indP > 0):
            for jj in range(len(indP)):
                t_p_ind = np.where((t >= starttime[indP[jj]]) & (t <= endtime[indP[jj]]))[0]
                ax.plot(t[t_p_ind], data[t_p_ind] + csn_distances[range_ind], c="red", alpha=0.5, linewidth=0.5)

    ax.plot(csn_p_travel, source_station_dist, color="red", label="P-wave arrival", linewidth=1, linestyle="dashed", alpha=0.75)
    ax.plot(csn_s_travel, source_station_dist, color="blue", label="S-wave arrival", linewidth=1, linestyle="dashdot", alpha=0.75)
    # plot non-appearing lines for P/S 
    ax.plot([1000,1001], [1000,1001], color="red", label="P-wave detection", linewidth=2)
    ax.plot([1000,1001], [1000,1001], color="blue", label="S-wave detection", linewidth=2)
    ax.set_xlim([1,15.0])
    ax.set_ylim([3,41])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_bounds(5, 40)
    ax.tick_params(direction='in')
    ax.spines['bottom'].set_bounds(2,14)
    ax.legend(loc="lower right",fontsize=10, bbox_to_anchor=(0.95, 0.1))
    ax.set_xlabel("Time relative to origin [s]",fontsize=18)
    ax.set_ylabel("Epicentral Distance [km]",fontsize=18)
    ax.tick_params(labelsize=12)
    ax.set_title("2020 M4.5 El Monte, CA Earthquake", fontsize=18)
    plt.savefig(os.path.join(figdir, "csn-inference-revised.pdf"))
    plt.close()