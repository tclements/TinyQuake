# TinyQuake

Earthquake detection with tinyML on Arduino! 

Tim Clements
tclements at usgs dot gov

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