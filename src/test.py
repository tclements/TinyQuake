import dataprep
import glob
import os 
import matplotlib.pyplot as plt 
import numpy as np 
import seaborn as sns
import sklearn.metrics
import tensorflow as tf
import tensorflow.keras as keras 

def test2input(A,input_dir):
    'Convert test data to cc/header file'

    # files to write 
    cc_file = os.path.join(input_dir,"input_data.cc")
    h_file = os.path.join(input_dir,"input_data.h")

    # create header file first 
    h_out = "#ifndef EDE_INPUT_DATA_H_\n" \
            "#define EDE_INPUT_DATA_H_\n\n" \
            "extern const float input_data[];\n" \
            "#endif\n"
    open(h_file, "w").write(h_out)

    # write data to cc file 
    A = A.flatten(order="F")
    cc_out = '#include "input_data.h"\n' \
             "static const int input_data_len = 240;\n" \
             "static const float input_data[240] = {\n"
    arrstring = ""
    for ii in range(A.size-1):
        arrstring += str(A[ii])
        arrstring += ", "
    arrstring += str(A[-1])
    arrstring += "};\n"
    cc_out += arrstring
    open(cc_file, "w").write(cc_out)
    return None

def decision_threshold(Y,threshold):
    Ycop = Y.copy()
    Ycop[np.where(Ycop[:,0:2] < threshold)] = 0
    return np.argmax(Ycop,axis=-1)

if __name__ == "__main__": 

    # load test data 
    srcdir = os.path.dirname(os.path.abspath(__file__))
    datadir = os.path.dirname(os.path.join(srcdir, "../data/"))
    maindir = os.path.abspath(os.path.join(srcdir, ".."))
    figdir = os.path.join(maindir, "FIGURES")
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

    # check that models work on test data 
    maindir = os.path.abspath(os.path.join(srcdir, ".."))
    CNNDIR = os.path.join(maindir, "CNN")
    CNNMODELDIR = os.path.join(CNNDIR,"models")
    model_name = os.path.join(CNNMODELDIR, "model_17_16_32_32.tflite")
    model = tf.lite.Interpreter(model_name)
    model.allocate_tensors()
    model_input_index = model.get_input_details()[0]["index"]
    model_output_index = model.get_output_details()[0]["index"]
    x_value_tensor = tf.convert_to_tensor(Xtest[0:1,:,:,:], dtype=np.float32)
    model.set_tensor(model_input_index, x_value_tensor)
    model.invoke()
    model.get_tensor(model_output_index)

    # convert some test data to data that can be read on the device 
    test2input(Xtest[0:1],srcdir)

    # test precision, recall and F1 score on range on thresholds 
    CLASSES = ["P-wave","S-wave","Noise"]
    preds = np.zeros((Xtest.shape[0],3))
    for ii in range(Xtest.shape[0]):
        x_value_tensor = tf.convert_to_tensor(Xtest[ii:ii+1,:,:,:], dtype=np.float32)
        model.set_tensor(model_input_index, x_value_tensor)
        model.invoke()
        preds[ii,:] = model.get_tensor(model_output_index)
    thresholds = np.linspace(0.33,0.99,60)
    reports = {}
    for ii in range(len(thresholds)): 
        threshpred = decision_threshold(preds,thresholds[ii])
        reports[ii] = sklearn.metrics.classification_report(truth,threshpred,target_names=CLASSES,output_dict=True)

    # extract accuracies 
    accuracy = np.zeros(len(thresholds))
    Precall = np.zeros(len(thresholds))
    Pprecision = np.zeros(len(thresholds))
    Srecall = np.zeros(len(thresholds))
    Sprecision = np.zeros(len(thresholds))
    Nrecall = np.zeros(len(thresholds))
    Nprecision = np.zeros(len(thresholds))
    for ii in range(len(thresholds)):
        accuracy[ii] = reports[ii]["accuracy"]
        Precall[ii] = reports[ii]["P-wave"]["recall"]
        Pprecision[ii] = reports[ii]["P-wave"]["precision"]
        Srecall[ii] = reports[ii]["S-wave"]["recall"]
        Sprecision[ii] = reports[ii]["S-wave"]["precision"]
        Nrecall[ii] = reports[ii]["Noise"]["recall"]
        Nprecision[ii] = reports[ii]["Noise"]["precision"]

    # plot precision vs recall 
    fig,ax = plt.subplots(figsize=(6,6))
    im = ax.scatter(Precall ,Pprecision,100,c=thresholds,alpha=0.85,label="P-wave",edgecolor="k",cmap=plt.cm.inferno)
    ax.scatter(Srecall ,Sprecision,100,c=thresholds,alpha=0.85,label="S-wave",edgecolor="k",marker="^",cmap=plt.cm.inferno)
    ax.legend(loc="center left",fontsize=14,borderpad = 1.)
    ax.set_xlabel("Recall",fontsize=18)
    ax.set_ylabel("Precision",fontsize=18)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_ylim([0.78, 1.0])
    ax.set_xlim([0.2, 1.0])
    ax.spines['left'].set_bounds(0.8, 1.0)
    ax.tick_params(direction='in')
    ax.spines['bottom'].set_bounds(0.2,1.0)
    ax.tick_params(labelsize=12)
    c_ax = ax.inset_axes([0.15, 0.2, 0.5, 0.05])
    cb = fig.colorbar(im, cax=c_ax, orientation="horizontal")
    cb.ax.xaxis.set_ticks_position("top")
    cb.ax.set_xlabel('Detection Threshold',fontsize=14)
    cb.ax.tick_params(labelsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(figdir, "precision-vs-recall.pdf"))
    plt.close()

    # plot confusion matrix 
    ind = np.argmax(accuracy)
    maxpred = decision_threshold(preds,thresholds[ind])
    confusion = sklearn.metrics.confusion_matrix(truth,maxpred)
    plt.figure(figsize=(6, 6))
    ax = sns.heatmap(np.round(confusion / np.sum(confusion,axis=0),decimals=3), xticklabels=CLASSES, yticklabels=CLASSES, 
            annot=True,cmap="Blues",linewidths=.5,cbar=False, annot_kws={"fontsize":18})
    ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize = 16)
    ax.set_yticklabels(ax.get_ymajorticklabels(), fontsize = 16)
    plt.xlabel('Prediction',fontsize=18)
    plt.ylabel('True label',fontsize=18)
    plt.tight_layout()
    plt.savefig(os.path.join(figdir, "confusion-matrix.pdf"))
    plt.close()



