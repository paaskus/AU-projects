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
    grid_search_model = GridSearchCV(svm_model, parameters, n_jobs=-1)
    grid_search_model = grid_search_model.fit(X, y)
    return grid_search_model.cv_results_

def print_score(score, kernel):
    # Create a DataFrame object using the cross validation result and filter out the 
    # relevant information. Finally display/print it. 
    dataframe = pd.DataFrame(score) 
    relevant = dataframe.filter(['mean_test_score', 'mean_train_score', 'std_test_score', 'std_train_score', 'param_C', 'param_coef0', 'param_gamma', 'mean_fit_time']).sort_values(['mean_test_score'])
    display(relevant)

    # Save the data to a file, then load it again and print it. 
    filename = 'results/svc_rbf.csv'
    relevant.to_csv(filename, index=False)

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
    digs = digs[0:2000, :]
    labs = labs[0:2000]

    parser = argparse.ArgumentParser()
    parser.add_argument('-lin', action='store_true', default=False)
    parser.add_argument('-poly2', action='store_true', default=False)
    parser.add_argument('-poly3', action='store_true', default=False)
    parser.add_argument('-rbf', action='store_true', default=False)
    args = parser.parse_args()
    if args.lin:
        print('running linear svm')
        ### YOUR CODE HERE
        parameters = {
            'C': [1, 10, 100, 1000, 10000, 1000000]
        }
        svm_model = svm.SVC(kernel='linear') # Note that ovr is default we only write to make it clear here.
        score = cross_validate(digs, labs, svm_model, parameters)
        print_score(score, "linear")
        ### END CODE
    if args.poly2:
        print('running poly 2 svm')
        ### YOUR CODE HERE
        parameters = {
            'C': [1, 10, 100, 1000],
            'coef0': [0.01, 0.1, 1, 10, 100]
        }
        svm_model = svm.SVC(kernel='poly', degree=2, C=1, decision_function_shape="ovr") # Note that ovr is default we only write to make it clear here.
        score = cross_validate(digs, labs, svm_model, parameters)
        print_score(score, "poly 2")
        ### END CODE
    if args.poly3:
        print('running poly 3 svm')
        #### YOUR CODE HERE
        parameters = {
            'C': [1, 10, 100, 1000],
            'coef0': [0.01, 0.1, 1, 10, 100]
        }
        svm_model = svm.SVC(kernel='poly', degree=3, C=1, decision_function_shape="ovr") # Note that ovr is default we only write to make it clear here.
        score = cross_validate(digs, labs, svm_model, parameters)
        print_score(score, "poly 3")
        ### END CODE
    if args.rbf:
        print('running rbf svm')
        ### YOUR CODE HERE
        parameters = {
            'C': [1, 10, 100, 200, 1000],
            'gamma': [0.008, 0.009, 0.01, 0.1, 1]
        }
        svm_model = svm.SVC(kernel='rbf') # Note that ovr is default we only write to make it clear here.
        score = cross_validate(digs, labs, svm_model, parameters)
        print_score(score, "rbf")
        ### END CODE
