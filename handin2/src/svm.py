from sklearn import svm
from sklearn.model_selection import GridSearchCV, cross_val_score
import numpy as np
import pandas as pd
import os
import argparse
from IPython.display import display
from h2_util import load_train_data, load_test_data
from model_stats import export_dataframe, model_stats


def save_model(cv, name):
    """ Simple save the SVC object learned with cross validation, ignore if you do not need it

    Params:
      cv: GridSearchCV fitted on data
      name: str - name to give model, i.e. rbf_kernel or something like that
    """
    print('saving', name)
    np.savez_compressed(os.path.join('model_weights','{0}_best_model.npz'.format(name)), res=cv.best_estimator_)

def load_model(name):
    """ Simple function to load an SVM  model, ignore if you do not need it

    Args:
      name: str - name given to model when saved
    """
    tmp = np.load(os.path.join('model_weights','{0}_best_model.npz'.format(name)))
    return tmp['res'].item()


### YOUR CODE HERE
def cross_validate(X, y, svm_model, parameters):
    # split data into training and validation sets
    grid_search_model = GridSearchCV(svm_model, parameters, n_jobs=1);

    scores = cross_val_score(svm_model, X, y, cv=5)
    return scores.mean()

def print_score(score, kernel):
    print("Accuracy for {0} kernel: {1}".format(kernel, score))

### END CODE

if __name__=="__main__":
    """
    Main code you can use and update as you please if you want to use command line arguments
    Otherwise you are free to ignore it.

    There are some extra functions in model_stats you can use as well if you would like to.
    """
    if not os.path.exists('results'):
        print('create results folder')
        os.mkdir('results')
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    [au_train_images,au_train_labels] = load_train_data()
    [au_test_images,au_test_labels] = load_test_data()
    rp = np.random.permutation(au_train_labels.size)
    digs = au_train_images[rp,:]
    labs = au_train_labels[rp]
    digs = digs[0:1000, :]
    labs = labs[0:1000]

    parser = argparse.ArgumentParser()
    parser.add_argument('-lin', action='store_true', default=False)
    parser.add_argument('-poly2', action='store_true', default=False)
    parser.add_argument('-poly3', action='store_true', default=False)
    parser.add_argument('-rbf', action='store_true', default=False)
    args = parser.parse_args()
    if args.lin:
        print('running linear svm')
        ### YOUR CODE HERE
        svm_model = svm.SVC(kernel='linear', C=1, decision_function_shape="ovr") # Note that ovr is default we only write to make it clear here.
        score = cross_validate(digs, labs, svm_model, parameters)
        print_score(score, "linear")
        ### END CODE
    if args.poly2:
        print('running poly 2 svm')
        ### YOUR CODE HERE
        svm_model = svm.SVC(kernel='poly', degree=2, C=1, decision_function_shape="ovr") # Note that ovr is default we only write to make it clear here.
        score = cross_validate(digs, labs, svm_model, parameters)
        print_score(score, "poly 2")
        ### END CODE
    if args.poly3:
        print('running poly 3 svm')
        #### YOUR CODE HERE
        svm_model = svm.SVC(kernel='poly', degree=3, C=1, decision_function_shape="ovr") # Note that ovr is default we only write to make it clear here.
        score = cross_validate(digs, labs, svm_model, parameters)
        print_score(score, "poly 3")
        ### END CODE
    if args.rbf:
        print('running rbf svm')
        ### YOUR CODE HERE
        svm_model = svm.SVC(kernel='rbf', C=1, decision_function_shape="ovr") # Note that ovr is default we only write to make it clear here.
        score = cross_validate(digs, labs, svm_model, parameters)
        print_score(score, "rbf")
        ### END CODE
