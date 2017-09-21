import glob
import os
import time
import ast

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.model_selection import LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

import ExperimentsRunner as exp
import SupervisedLearning as sl
import utils as u
import NeuralNetwork as nnet

def GetIdForConfig(config):
    return "weights-{0}_neighbors-{1}".format(config['weights'], config['neighbors'])


def RunKNNClassifier(datasets_root_folder, nominal_value_columns=None, positive_class_label=None, metric_fn=None,cv_file=None,cv_scoring='f1'):
    file_extn = "csv"
    testfiles = glob.glob("{0}/*.test.{1}".format(datasets_root_folder, file_extn))
    first = True
    for dataset_dir in u.Get_Subdirectories(datasets_root_folder):
        if(first):
            assert("ts-100" in dataset_dir)
            first = False
        trainfile = glob.glob("{0}/*.train.{1}".format(dataset_dir, file_extn))[0]
        paramfile = glob.glob("{0}/*.params.txt".format(dataset_dir))[0]
        dt_root = u.PreparePath(dataset_dir + "/knn", is_file=False)

        data = pd.read_csv(trainfile)
        testdata = pd.read_csv(testfiles[0])
        train_len = len(data)

        cols_to_ignore = set(
            nominal_value_columns) if nominal_value_columns is not None else set([])
        cols_to_ignore.add(data.columns[-1])
        cols_to_transform = [c for c in data.columns if c not in cols_to_ignore]
        scaler = StandardScaler()
        scaler.fit(data[cols_to_transform])
        data[cols_to_transform] = scaler.transform(data[cols_to_transform])
        testdata[cols_to_transform] = scaler.transform(testdata[cols_to_transform])

        all_data = pd.concat([data, testdata], axis=0, ignore_index=True)
        X_all, Y_all = nnet.PrepareDataAndLabel(
            all_data, positive_class_label, nominal_value_columns)
        X = X_all[0:train_len, :]
        Y = Y_all[0:train_len]
        test_X = X_all[train_len:, :]
        test_Y = Y_all[train_len:]

        param_grid = {'weights': np.array(['uniform', 'distance']), 'n_neighbors': np.array([5, 10, 20, 50])}
        classifier = KNeighborsClassifier()
        if((cv_file is None) or (os.path.isfile(cv_file) == False)):
            gscv = GridSearchCV(classifier,param_grid,scoring=cv_scoring,n_jobs=3)
            gscv.fit(X,Y)
            _D = pd.DataFrame(gscv.cv_results_)
            best_params = gscv.best_params_
        else:
            _D = None

        config_gen = ParameterGrid(
            {'weights': ['uniform'], 'neighbors': [-1]}) # -1 denotes that we need to take the cv results
        for config in config_gen:
            id = GetIdForConfig(config)
            params_info = u.ReadLinesFromFile(paramfile)
            params_info_dict = sl.GetDictionary(params_info)
            run_output_dir = u.PreparePath("{0}/{1}".format(dt_root, id), is_file=False)
            params_output_file = u.PreparePath(
                "{0}/{1}.params.txt".format(run_output_dir, id))
            model_output_file = u.PreparePath(
                "{0}/{1}.model".format(run_output_dir, id))
            train_output_file = u.PreparePath(
                "{0}/{1}.train.predictions.csv".format(run_output_dir, id))
            test_output_file = u.PreparePath(
                "{0}/{1}.test.predictions.csv".format(run_output_dir, id))
            cv_results_file=u.PreparePath(
                "{0}/{1}.grid_search_cv_results.csv".format(run_output_dir,id))
            model_output_file = u.PreparePath(
                "{0}/{1}.model".format(run_output_dir, id))
            scalar_output_file = u.PreparePath(
                "{0}/{1}.scaler".format(run_output_dir, id))
            if(cv_file is not None):
                cv_file = cv_file
            if(_D is not None):
                _D.to_csv(cv_results_file)
            else:
                cv_results = pd.read_csv(cv_file)
                best_params = ast.literal_eval(cv_results[cv_results['rank_test_score']==1].iloc[0]['params'])
    
            # if(os.path.isfile(test_output_file)):
            #	config = config_gen.GetNextConfigAlongWithIdentifier()
            #	continue
            config["trainset"] = trainfile
            config["class"] = "last"
            config["trainpredictionoutputfile"] = train_output_file
            config["predictionoutputfile"] = config["trainpredictionoutputfile"]
            config["modeloutputfile"] = model_output_file
            config["testpredictionoutputfile"] = test_output_file

            config["testset"] = testfiles[0]

            if(config['neighbors'] == -1):
                neighbors = best_params['n_neighbors']
                weights = best_params['weights']
                # _D.to_csv(cv_results_file)
                config['best_neighbors'] = neighbors
                config['best_weights'] = weights
            else:
                neighbors = config['neighbors']
                weights = config['weights']
            if(metric_fn is None):
                classifier = KNeighborsClassifier(neighbors,weights)
            else:
                classifier = KNeighborsClassifier(neighbors,weights,algorithm='brute',metric='pyfunc',metric_params={'func': metric_fn})
            
            loo = LeaveOneOut()
            y_actual = []
            y_predicted = []
            count = 0
            total = len(X)
            for train_idx, test_idx in loo.split(X):
                X_train, X_test = X[train_idx], X[test_idx]
                Y_train, Y_test = Y[train_idx], Y[test_idx]
                classifier.fit(X_train,Y_train)
                Y_test_predicted = classifier.predict(X_test)
                assert(len(Y_test_predicted) == 1)
                y_actual.append(Y_test[0])
                y_predicted.append(Y_test_predicted[0])
                count = count + 1
                if(count % 100 == 0):
                    print(str(count)+" "+str(total))

            start = time.clock()
            classifier.fit(X,Y)
            end = time.clock()
            print(end-start)
            config["modelbuildtimesecs"] = end-start
            # for train performance
            config["trainpredictionoutputfile"]=train_output_file
            #train_predicted_Y = classifier.predict(X)
            output = pd.DataFrame({"actual":y_actual,"predicted":y_predicted})
            output.to_csv(train_output_file,index=False)

            # now for test set
            config["predictionoutputfile"] = test_output_file
        
            start = time.clock()
            predicted_Y = classifier.predict(test_X)
            end = time.clock()
            u.WriteBinaryFile(model_output_file,classifier)
            u.WriteBinaryFile(scalar_output_file,scaler)
            config["modelevaltimesecs"] = end-start
            output = pd.DataFrame({"actual":test_Y,"predicted":predicted_Y})
            output.to_csv(test_output_file,index=False)

            for k in config:
                params_info.append("{0}={1}".format(k,config[k]))
            u.WriteTextArrayToFile(params_output_file,params_info)
        print("DONE dataset : " + dataset_dir)

def MapNominalValuesToIntegers(dataframe, columns_with_nominal_values):
    newdataframe = dataframe
    for col in columns_with_nominal_values:
        unique_keys = dataframe[col].unique()
        unique_values = np.arange(len(unique_keys))
        mapping = dict(zip(unique_keys,unique_values))
        newdataframe = newdataframe.replace({col:mapping})
    return newdataframe

def PrepareDataAndLabel(data,positive_class_label,columns_with_nominal_values):
	label_col = data.columns[-1]
	Y = sl.ConvertLabelsToZeroOne(data[label_col],positive_class_label)
	data = data.drop(label_col,axis=1)
	if(columns_with_nominal_values is not None):
		data = MapNominalValuesToIntegers(data,columns_with_nominal_values)
	X = data.as_matrix()
	return X,Y

def DistanceFuncForCreditScreeningDataset(x,y,
    nominal_attr_indxs=[0,3,4,5,6,8,9,11,12],
    real_attr_indxs=[1,2,7,10,13,14]):

    return np.sum((x[real_attr_indxs] - y[real_attr_indxs])**2) + np.sum(x[nominal_attr_indxs] == y[nominal_attr_indxs])

def RunKnnClassifierOnVowelRecognitionDataset(root=r"C:\Users\shkhandu\OneDrive\Gatech\Courses\ML\DataSets\LetterRecognition"):
    pos_class="v"
    metric_fn = sl.ComputePrecisionRecallForPythonOutputFormat
    keys_to_keep=['dataset_instance','test_split','train_split','random_state','train_split_percent_used','weights','neighbors','modelbuildtimesecs','modelevaltimesecs','best_neighbors','best_weights']
    cv_file = root + r"/i-0_t-80_T-20/i-0_t-80_ts-100/knn/weights-uniform_neighbors--1/weights-uniform_neighbors--1.grid_search_cv_results.csv"
    classifier_fn = lambda x : RunKNNClassifier(x,positive_class_label=pos_class,cv_file=cv_file)
    id="vowel.knn_3_0"
    algo_folder='knn'
    force_computation=True
    exp.RunNEvaluateExperimentsOnDataSet(classifier_fn,root,id,metric_fn,algo_folder,keys_to_keep,pos_class,["i-0"],force_computation)

def RunKnnClassifierOnCreditScreeningDataset(root=r"C:\Users\shkhandu\OneDrive\Gatech\Courses\ML\DataSets\CreditScreeningDataset"):
    pos_class="+"
    metric_fn = sl.ComputePrecisionRecallAccuracyForPythonOutputFormat
    keys_to_keep=['dataset_instance','test_split','train_split','random_state','train_split_percent_used','weights','neighbors','modelbuildtimesecs','modelevaltimesecs','best_neighbors','best_weights']
    cv_file = root + r"/i-0_t-80_T-20/i-0_t-80_ts-100/knn/weights-uniform_neighbors--1/weights-uniform_neighbors--1.grid_search_cv_results.csv"
    classifier_fn = lambda x : RunKNNClassifier(x,['A1','A4','A5','A6','A7','A9','A10','A12','A13'],pos_class,cv_file=cv_file,cv_scoring='accuracy')
    id="credit.knn_3_0"
    algo_folder='knn'
    force_computation=True
    exp.RunNEvaluateExperimentsOnDataSet(classifier_fn,root,id,metric_fn,algo_folder,keys_to_keep,pos_class,["i-0"],force_computation)

def main():
    RunKnnClassifierOnCreditScreeningDataset(r"C:\Users\shkhandu\OneDrive\Gatech\Courses\ML\DataSets\CreditScreeningDataset")
    RunKnnClassifierOnVowelRecognitionDataset(r"C:\Users\shkhandu\OneDrive\Gatech\Courses\ML\DataSets\LetterRecognition")

if __name__ == '__main__':
    main()