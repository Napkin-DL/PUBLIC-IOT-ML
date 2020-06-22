import tensorflow as tf
import os
import numpy as np
from tensorflow.python.platform import gfile
from PIL import Image
from matplotlib.pyplot import imshow

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import collections
import json
from sklearn.model_selection import StratifiedKFold, train_test_split, ShuffleSplit
from sklearn.metrics import (confusion_matrix, classification_report, accuracy_score, roc_curve, average_precision_score,
                            precision_recall_curve, precision_score, recall_score, f1_score, matthews_corrcoef, auc)

try:
    from joblib import dump, load
except ImportError:
    from sklearn.externals.joblib import dump, load
    

tf.global_variables_initializer()
tf.reset_default_graph()

class MobileNetInference(object):
    
    def __init__(self, model_filepath, class_map):
        self.model_filepath = model_filepath
        self.class_map = class_map
        self.load_graph(model_filepath = model_filepath)
        
        
    def load_graph(self, model_filepath):
        self.graph = tf.Graph()

        # Load the protobuf model file(.pb) and parse it to retrive the graph 
        with tf.gfile.GFile(model_filepath, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())   
        
        # Set default graph as graph
        with self.graph.as_default():
            tf.import_graph_def(graph_def, input_map=None, return_elements=None, name="")   
            self.net_input = self.graph.get_tensor_by_name('input:0')
            self.net_output = self.graph.get_tensor_by_name('MobilenetV1/Predictions/Reshape_1:0')
        
        # Avoid to change graph
        self.graph.finalize()
        
        self.sess = tf.Session(graph = self.graph)
        self.input_node_info = [n for n in graph_def.node if n.op == 'Placeholder']
  

    def get_all_tensors(self):    
        all_tensors = [tensor for op in self.graph.get_operations() for tensor in op.values()]
        return all_tensors
    
    
    def get_input_node_info(self):
        return self.input_node_info[0]
    
    def predict(self, img_filepath, img_size=224, show_image=True):    
        # Open image data and resize it
        img = Image.open(img_filepath)
        img = img.resize((img_size, img_size)) 
        img_arr = np.asarray(img) 

        if show_image: 
            imshow(img_arr)
        img_arr = img_arr.astype('float32')
        img_arr = (img_arr -127.5) * (1.0 / 127.5)
        img_arr = img_arr[np.newaxis, :]


        # Get predictions
        pred_scores = self.sess.run(self.net_output, feed_dict={self.net_input: img_arr} )
        pred_label = np.argmax(pred_scores, axis=1)[0]
        pred_label_str = self.class_map[pred_label]
        pred_score = pred_scores[0][pred_label]
        
        return pred_label, pred_label_str, pred_score


def plot_roc_curve(y_true, y_score, is_single_fig=False):
    """
    Plot ROC Curve and show AUROC score
    """    
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.title('AUROC = {:.4f}'.format(roc_auc))
    plt.plot(fpr, tpr, 'b')
    plt.plot([0,1], [0,1], 'r--')
    plt.xlim([-0.05,1.05])
    plt.ylim([-0.05,1.05])
    plt.ylabel('TPR(True Positive Rate)')
    plt.xlabel('FPR(False Positive Rate)')
    if is_single_fig:
        plt.show()
    
def plot_pr_curve(y_true, y_score, is_single_fig=False):
    """
    Plot Precision Recall Curve and show AUPRC score
    """
    prec, rec, thresh = precision_recall_curve(y_true, y_score)
    avg_prec = average_precision_score(y_true, y_score)
    plt.title('AUPRC = {:.4f}'.format(avg_prec))
    plt.step(rec, prec, color='b', alpha=0.2, where='post')
    plt.fill_between(rec, prec, step='post', alpha=0.2, color='b')
    plt.plot(rec, prec, 'b')
    plt.xlim([-0.05,1.05])
    plt.ylim([-0.05,1.05])
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    if is_single_fig:
        plt.show()

def plot_conf_mtx(y_true, y_score, thresh=0.5, class_labels=['0','1'], is_single_fig=False):
    """
    Plot Confusion matrix
    """    
    y_pred = np.where(y_score >= thresh, 1, 0)
    print("confusion matrix (cutoff={})".format(thresh))
    print(classification_report(y_true, y_pred, target_names=class_labels))
    conf_mtx = confusion_matrix(y_true, y_pred)
    sns.heatmap(conf_mtx, xticklabels=class_labels, yticklabels=class_labels, annot=True, fmt='d')
    plt.title('Confusion Matrix')
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    if is_single_fig:
        plt.show()

def prob_barplot(y_score, bins=np.arange(0.0, 1.11, 0.1), right=False, filename=None, figsize=(10,4), is_single_fig=False):
    """
    Plot barplot by binning predicted scores ranging from 0 to 1
    """    
    c = pd.cut(y_score, bins, right=right)
    counts = c.value_counts()
    percents = 100. * counts / len(c)
    percents.plot.bar(rot=0, figsize=figsize)
    plt.title('Histogram of score')
    print(percents)
    if filename is not None:
        plt.savefig('{}.png'.format(filename))   
    if is_single_fig:
        plt.show()
    
def evaluate_prediction(y_true, y_score, thresh=0.5):
    """
    All-in-one function for evaluation. 
    """    
    plt.figure(figsize=(14,4))
    plt.subplot(1,3,1)
    plot_roc_curve(y_true, y_score)
    plt.subplot(1,3,2)    
    plot_pr_curve(y_true, y_score)
    plt.subplot(1,3,3)    
    plot_conf_mtx(y_true, y_score, thresh) 
    plt.show()

def get_score_df(y_true, y_score, start_score=0.0, end_score=0.7, cutoff_interval=0.05):
    """
    Get a dataframe contains general metrics
    """    
    import warnings
    warnings.filterwarnings("ignore")
    score = []
    
    for cutoff in np.arange(start_score, end_score+0.01, cutoff_interval)[1:]:
        y_pred = np.where(y_score >= cutoff, 1, 0)
        conf_mat = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = conf_mat[0,0], conf_mat[0,1], conf_mat[1,0], conf_mat[1,1]
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        if precision !=0 and recall !=0 :
            f1 = f1_score(y_true, y_pred)
        else:
            f1 = 0     
        mcc = matthews_corrcoef(y_true, y_pred)
        score.append([cutoff, tp, fp, tn, fn, precision, recall, f1, mcc])
        
    score_df = pd.DataFrame(score, columns = ['Cutoff', 'TP', 'FP', 'TN' ,'FN', 'Precision', 'Recall', 'F1', 'MCC'])
    return score_df

def get_test_scores(model, test_img_list, test_img_path):
    num_test = len(test_img_list)
    y_score = np.zeros(num_test)

    for idx, fname in enumerate(test_img_list):
        img_filepath = os.path.join(test_img_path, fname)
        pred_cls, pred_cls_str, pred_score = model.predict(img_filepath, show_image=False)
        y_score[idx] = pred_cls    
        
    return y_score
