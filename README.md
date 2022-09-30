# Earthquake Detection at the Edge 

Tim Clements
tclements at usgs dot gov

## Abstract 
Earthquake  detection  is  the  critical  first  step  inEarthquake  Early  Warning  (EEW)  systems.  For  robust  EEWsystems, detection accuracy, detection latency and sensor densityare critical to providing real-time earthquake alerts. TraditionalEEW  systems  use  fixed  sensor  networks  or,  more  recently,  net-works  of  mobile  phones  equipped  with  micro-electromechanicalsystems  (MEMS)  accelerometers.  Internet  of  things  (IoT)  edgedevices,  with  built-in  machine  learning  (ML)  capable  micro-controllers, and always-on, always internet-connected, stationaryMEMS  accelerometers  provide  the  opportunity  to  deploy  ML-based  earthquake  detection  and  warning  using  a  single-stationapproach at a global scale. Here, we test and evaluate deep learn-ing ML algorithms for earthquake detection on Arduino CortexM4  microcontrollers.  We  show  the  trade-offs  between  detectionaccuracy  and  latency  on  resource-constrained  microcontrollers.

## How to Use 

Data used in this study is available on [GitHub](https://github.com/smousavi05/STEAD). To train the models, it is expected that you place the merged HDF5 and csv file in the `data` directory. Only the testing data is available in this repo. 

1. The conda environment used to create this work is `environment.yml`
2. Data preparation is done through the `src/dataprep.py` script. 
3. Training is done with the `src/train.py` script. 
4. Model testing is done with the `src/test.py` script. 
5. The Arduino file for loading models on device is `src/FC_detection.ino`. 
6. Tflite and header files for each model are in the `DNN`, `CNN` and `RNN`. 
7. Information about model training is provided as in pickle files `DNN/DNN_models.pkl`, `CNN/CNN_models.pkl` and `RNN/RNN_models.pkl`. 
8. Scripts to make the figures in the `FIGURES` directory are in `src/plot_*.py`. 