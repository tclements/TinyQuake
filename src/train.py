import os, re, subprocess, pickle
import numpy as np 
import scipy 
import scipy.signal as signal
import matplotlib.pyplot as plt 
import tensorflow as tf 
import tensorflow.keras as keras 
from tensorflow.keras import layers
import pandas as pd 
import h5py 
from tqdm import tqdm
import shutil
import sklearn.metrics
import dataprep

def DNN_model(dnn_sizes,input_shape,dropout=0.1):
    model = keras.Sequential()
    model.add(keras.Input(shape=input_shape))  
    model.add(layers.Flatten())
    for ii in range(len(dnn_sizes)-1):
        model.add(layers.Dense(dnn_sizes[ii], activation='relu'))
        model.add(layers.Dropout(dropout))
    model.add(layers.Dense(dnn_sizes[-1], activation='softmax'))
    return model

def CNN_model(filters,kernel_sizes,input_shape,pool_size,dnn_sizes,dropout=0.1):
    model = keras.Sequential()
    model.add(keras.Input(shape=input_shape))
    for ii in range(len(filters)):
        model.add(layers.Conv2D(filters[ii],kernel_sizes[ii], activation="relu"))
        model.add(layers.MaxPooling2D(pool_size=pool_size))
        model.add(layers.Dropout(dropout))
    model.add(layers.Flatten())
    for ii in range(len(dnn_sizes)-1):
        model.add(layers.Dense(dnn_sizes[ii],activation="relu"))
        model.add(layers.Dropout(dropout))
    model.add(layers.Dense(dnn_sizes[-1],activation="softmax"))
    return model

def RNN_model(rnn_units, input_shape, dnn_sizes, dropout=0.1):
    model = keras.Sequential()
    model.add(keras.Input(shape=input_shape[:-1]))
    for ii in range(len(rnn_units)):
        model.add(layers.LSTM(rnn_units[ii], return_sequences=True))
        model.add(layers.Dropout(dropout))
    model.add(layers.Flatten())
    for ii in range(len(dnn_sizes)-1):
        model.add(layers.Dense(dnn_sizes[ii],activation="relu"))
        model.add(layers.Dropout(dropout))
    model.add(layers.Dense(dnn_sizes[-1],activation="softmax"))
    return model

def representative_dataset():
  for data in testdata.batch(1).take(100):
    yield [data.astype(tf.float32)]

if __name__ == "__main__": 

    # directories 
    srcdir = os.path.dirname(os.path.abspath(__file__))
    datadir = os.path.abspath(os.path.dirname(os.path.join(srcdir, "../data/")))
    maindir = os.path.abspath(os.path.join(srcdir, ".."))

    # classes for prediction 
    CLASSES = ["P-wave","S-wave","Noise"]

    # load train data 
    trainfile = os.path.join(datadir, "train.npz")
    traindata = np.load(trainfile)
    Xtrain = traindata["Xtrain"]
    Ytrain = traindata["Ytrain"]

    #### Training data cleaning steps  ####
    # 1. Remove data with all zeros 
    # 2. Remove data with > 2g
    # 3. Extract log10 amplitude 
    # 4. Normalize ENZ to [-1,1] and preserve relative amplitudes
    # 5. Convert to Int16 
    # 6. Convert to FP32
    # 7. Normalize ENZ to [-1,1] and preserve relative amplitudes
    Xtrain, Ytrain = dataprep.remove_small_amplitude(Xtrain, Ytrain) 
    Xtrain, Ytrain = dataprep.remove_large_amplitude(Xtrain, Ytrain)
    log10_amplitude = np.log10(np.max(np.abs(Xtrain), axis=(1,2,3)))
    Xtrain = dataprep.normalize(Xtrain)
    Xtrain = dataprep.quantize(Xtrain)
    Xtrain = Xtrain.astype(np.float32)
    Xtrain = dataprep.normalize(Xtrain)

    # load training data 
    DATASET_SIZE = Xtrain.shape[0]
    batch_size = 256 
    DATASET_SIZE //= batch_size
    train_size = int(0.8 * DATASET_SIZE)
    val_size = int(0.2 * DATASET_SIZE)
    waves = dataset = tf.data.Dataset.from_tensor_slices((Xtrain,Ytrain)).batch(batch_size)
    val_dataset = waves.take(val_size)
    train_dataset = waves.skip(val_size)

    # load test data 
    testfile = os.path.join(datadir, "test.npz")
    testdata = np.load(testfile)
    Xtest = testdata["Xtest"]
    Ytest = testdata["Ytest"]
    Xtest, Ytest = dataprep.remove_small_amplitude(Xtest, Ytest) 
    Xtest, Ytest = dataprep.remove_large_amplitude(Xtest, Ytest)
    log10_amplitude = np.log10(np.max(np.abs(Xtest), axis=(1,2,3)))
    Xtest = dataprep.normalize(Xtest)
    Xtest = dataprep.quantize(Xtest)
    Xtest = Xtest.astype(np.float32)
    Xtest = dataprep.normalize(Xtest)
    truth = np.argmax(Ytest,axis=-1)

    # directories for tflite and header files 
    DNNDIR = os.path.join(maindir, "DNN")
    CNNDIR = os.path.join(maindir, "CNN")
    DSCNNDIR = os.path.join(maindir, "DSCNN")
    RNNDIR = os.path.join(maindir, "RNN")
    DNNMODELDIR = os.path.join(DNNDIR,"models")
    DNNHEADERDIR = os.path.join(DNNDIR,"headers")
    CNNMODELDIR = os.path.join(CNNDIR,"models")
    CNNHEADERDIR = os.path.join(CNNDIR,"headers")
    # DSCNNMODELDIR = os.path.join(DSCNNDIR,"models")
    # DSCNNHEADERDIR = os.path.join(DSCNNDIR,"headers")
    RNNMODELDIR = os.path.join(RNNDIR,"models")
    RNNHEADERDIR = os.path.join(RNNDIR,"headers")
    for DIR in [DNNMODELDIR,DNNHEADERDIR,CNNMODELDIR,CNNHEADERDIR,RNNMODELDIR,RNNHEADERDIR]:
        if not os.path.isdir(DIR): 
            os.makedirs(DIR)

    # number of epochs for training 
    epochs = 20 

    # create + train DNN models 
    input_shape = (80,3,1)
    # model_sizes = [(2**ii,2**jj,3) for ii in range(4,9) for jj in range(4,ii+1)]
    model_sizes = [(2**ii,2**ii,3) for ii in range(2,9)]
    model_sizes.extend([(2**ii,2**ii, 2**ii,3) for ii in range(2,8, 2)])
    models = {}
    for ii in range(len(model_sizes)):
        model_name = os.path.join(DNNMODELDIR,"model_{}_".format(ii) + "_".join(map(str, model_sizes[ii])) + ".tflite")
        model_header = os.path.join(DNNHEADERDIR,"model_{}_".format(ii) + "_".join(map(str, model_sizes[ii])) + ".h")
        if os.path.isfile(model_name):
            continue
        model_dict = {}
        model = DNN_model(model_sizes[ii],input_shape)
        model.compile(optimizer="adam", loss="categorical_crossentropy",metrics=['mae', 'accuracy'])
        history = model.fit(train_dataset, epochs=epochs, validation_data=val_dataset)
        model_dict["params"] = model.count_params()
        model_dict["history"] = history.history
        model_dict["model_tf_size"] = model_sizes[ii]

        # get precision, recall, f1-score, support
        preds = model.predict(Xtest)
        outpred = np.argmax(preds,axis=-1)
        model_dict["report"] = sklearn.metrics.classification_report(
            truth,
            outpred,
            target_names=CLASSES,
            output_dict=True
        )

        # get confusion matrix 
        model_dict["confusion"] = sklearn.metrics.confusion_matrix(truth,outpred)

        # Convert the model to the TensorFlow Lite format without quantization
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()

        # Save the model to disk
        open(model_name, "wb").write(tflite_model)
        model_dict["model_size"] = os.path.getsize(model_name)

        # save header to disk 
        subprocess.call('echo "const unsigned char model[] __attribute__((aligned(4))) = {{"  > {}'.format(model_header),shell=True)
        subprocess.call("cat {} | xxd -i >> {}".format(model_name,model_header),shell=True)
        subprocess.call('echo "}};" >> {}'.format(model_header),shell=True)

        # get header size 
        model_dict["model_header_size"] = os.path.getsize(model_header)
        models[ii] = model_dict

    # write model history to disk 
    DNN_models = os.path.join(DNNDIR, "DNN_models.pkl")
    if not os.path.isfile(DNN_models):
        with open(DNN_models, 'wb') as file:
            pickle.dump(models, file)

    ## train CNN models
    filter_sizes = [(2**ii,2**jj) for ii in range(1,6) for jj in range(ii,ii+2)]
    filter_sizes.extend([(2**ii,2**jj, 2**kk) for ii in range(1,6) for jj in range(ii,ii+2) for kk in range(jj,jj+1)])
    kernel_sizes = [[(3,1) for jj in range(len(filter_sizes[ii]))] for ii in range(len(filter_sizes))]
    pool_size = (3,1)
    dnn_sizes = (16,3)
    models = {}
    for ii in range(len(filter_sizes)):
        model_name = os.path.join(CNNMODELDIR,"model_{}_".format(ii) + "_".join(map(str, filter_sizes[ii])) + ".tflite")
        model_header = os.path.join(CNNHEADERDIR,"model_{}_".format(ii) + "_".join(map(str, filter_sizes[ii])) + ".h")
        if os.path.isfile(model_name):
            continue
        model_dict = {}
        model = CNN_model(filter_sizes[ii],kernel_sizes[ii],input_shape,pool_size,dnn_sizes)
        model.compile(optimizer="adam", loss="categorical_crossentropy",metrics=['mae', 'accuracy'])
        history = model.fit(train_dataset, epochs=epochs, validation_data=val_dataset)
        model_dict["params"] = model.count_params()
        model_dict["history"] = history.history
        model_dict["filter_size"] = filter_sizes[ii]
        model_dict["kernel_size"] = kernel_sizes[ii]
        model_dict["pool_size"] = pool_size

        # get precision, recall, f1-score, support
        preds = model.predict(Xtest)
        outpred = np.argmax(preds,axis=-1)
        model_dict["report"] = sklearn.metrics.classification_report(
            truth,
            outpred,
            target_names=CLASSES,
            output_dict=True
        )

        # get confusion matrix 
        model_dict["confusion"] = sklearn.metrics.confusion_matrix(truth,outpred)

        # Convert the model to the TensorFlow Lite format without quantization
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()

        # Save the model to disk
        open(model_name, "wb").write(tflite_model)
        model_dict["model_size"] = os.path.getsize(model_name)

        # save header to disk 
        subprocess.call('echo "const unsigned char model[] __attribute__((aligned(4))) = {{"  > {}'.format(model_header),shell=True)
        subprocess.call("cat {} | xxd -i >> {}".format(model_name,model_header),shell=True)
        subprocess.call('echo "}};" >> {}'.format(model_header),shell=True)

        # get header size 
        model_dict["model_header_size"] = os.path.getsize(model_header)
        models[ii] = model_dict

    # write model history to disk 
    CNN_models = os.path.join(CNNDIR, "CNN_models.pkl")
    if not os.path.isfile(CNN_models):
        with open(CNN_models, 'wb') as file:
            pickle.dump(models, file)

    ## train RNN models
    rnn_units = [(2**ii,) for ii in range(0,6)]
    rnn_units.extend([(2**ii,2**ii) for ii in range(0,5)])
    dnn_sizes = (16,3)
    models = {}
    for ii in range(len(rnn_units)):
        model_name = os.path.join(RNNMODELDIR,"model_{}_".format(ii) + "_".join(map(str, rnn_units[ii])) + ".tflite")
        model_header = os.path.join(RNNHEADERDIR,"model_{}_".format(ii) + "_".join(map(str, rnn_units[ii])) + ".h")
        if os.path.isfile(model_name):
            continue
        model_dict = {}
        model = RNN_model(rnn_units[ii],input_shape,dnn_sizes)
        model.compile(optimizer="adam", loss="categorical_crossentropy",metrics=['mae', 'accuracy'])
        history = model.fit(train_dataset, epochs=epochs, validation_data=val_dataset)
        model_dict["params"] = model.count_params()
        model_dict["history"] = history.history
        model_dict["rnn_units"] = rnn_units[ii]
        
        # get precision, recall, f1-score, support
        preds = model.predict(Xtest)
        outpred = np.argmax(preds,axis=-1)
        model_dict["report"] = sklearn.metrics.classification_report(
            truth,
            outpred,
            target_names=CLASSES,
            output_dict=True
        )

        # get confusion matrix 
        model_dict["confusion"] = sklearn.metrics.confusion_matrix(truth,outpred)

        # shenanigans to compile with LSTM 
        run_model = tf.function(lambda x: model(x))
        # This is important, let's fix the input size.
        BATCH_SIZE = 1
        STEPS = 80
        INPUT_SIZE = 3
        concrete_func = run_model.get_concrete_function(
            tf.TensorSpec([BATCH_SIZE, STEPS, INPUT_SIZE], model.inputs[0].dtype))

        # model directory.
        MODEL_DIR = "keras_lstm"
        model.save(MODEL_DIR, save_format="tf", signatures=concrete_func)

        # Convert the model to the TensorFlow Lite format without quantization
        converter = tf.lite.TFLiteConverter.from_saved_model(MODEL_DIR)
        tflite_model = converter.convert()

        # remove directory
        shutil.rmtree(MODEL_DIR)

        # Save the model to disk
        open(model_name, "wb").write(tflite_model)
        model_dict["model_size"] = os.path.getsize(model_name)

        # save header to disk 
        subprocess.call('echo "const unsigned char model[] __attribute__((aligned(4))) = {{"  > {}'.format(model_header),shell=True)
        subprocess.call("cat {} | xxd -i >> {}".format(model_name,model_header),shell=True)
        subprocess.call('echo "}};" >> {}'.format(model_header),shell=True)

        # get header size 
        model_dict["model_header_size"] = os.path.getsize(model_header)
        models[ii] = model_dict

    # write model history to disk 
    RNN_models = os.path.join(RNNDIR, "RNN_models.pkl")
    if not os.path.isfile(RNN_models):
        with open(RNN_models, 'wb') as file:
            pickle.dump(models, file)
