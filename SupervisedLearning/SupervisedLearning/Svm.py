import glob
import os
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

import ExperimentsRunner as exp
import SupervisedLearning as sl
import utils as u
import NeuralNetwork as nnet
import ast

def GetIdForConfig(config):
    return "cvresults"

def RunSVMClassifier(datasets_root_folder, nominal_value_columns=None, positive_class_label=None):
    file_extn = "csv"
    testfiles = glob.glob("{0}/*.test.{1}".format(datasets_root_folder, file_extn))
    for dataset_dir in u.Get_Subdirectories(datasets_root_folder):
        trainfile = glob.glob("{0}/*.train.{1}".format(dataset_dir, file_extn))[0]
        paramfile = glob.glob("{0}/*.params.txt".format(dataset_dir))[0]
        dt_root = u.PreparePath(dataset_dir + "/svm", is_file=False)
        params_info = u.ReadLinesFromFile(paramfile)
        params_info_dict=sl.GetDictionary(params_info)

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

        param_grid = [
                        {'C': [0.1, 1, 10, 100, 1000], 'degree' : [2,3,4],'kernel': ['poly']},
                        {'C': [0.1, 1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
                    ]
        classifier = SVC(cache_size=1500, random_state=int(params_info_dict['random_state']))
        gscv = GridSearchCV(classifier,param_grid,scoring='f1',n_jobs=3)
        gscv.fit(X,Y)
        _D = pd.DataFrame(gscv.cv_results_)
        best_params = gscv.best_params_
        config_gen = [{}] 
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

            _D.to_csv(cv_results_file)
            # cv_results = pd.read_csv(cv_results_file)
            # best_params = ast.literal_eval(cv_results[cv_results['rank_test_score']==1].iloc[0]['params'])
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
            config["kernel"] = best_params['kernel']
            config['C'] = best_params['C']
            if(config['kernel'] == 'rbf'):
                config['gamma'] = best_params['gamma']
                classifier = SVC(config['C'],gamma=config['gamma'],kernel=config['kernel'],cache_size=1500, random_state=int(params_info_dict['random_state']))
            else:
                config['degree'] = best_params['degree']
                classifier = SVC(config['C'],kernel=config['kernel'],degree=config['degree'],cache_size=1500, random_state=int(params_info_dict['random_state']))
                
            start = time.clock()
            classifier.fit(X,Y)
            end = time.clock()
            print(end-start)
            config["modelbuildtimesecs"] = end-start
            config['numsupportvectors'] = u.ConcatToStr(';',classifier.n_support_)
            # for train performance
            config["trainpredictionoutputfile"]=train_output_file
            train_predicted_Y = classifier.predict(X)
            output = pd.DataFrame({"actual":Y,"predicted":train_predicted_Y})
            output.to_csv(train_output_file,index=False)

            # now for test set
            config["predictionoutputfile"] = test_output_file
        
            start = time.clock()
            predicted_Y = classifier.predict(test_X)
            end = time.clock()
            config["modelevaltimesecs"] = end-start
            output = pd.DataFrame({"actual":test_Y,"predicted":predicted_Y})
            output.to_csv(test_output_file,index=False)

            for k in config:
                params_info.append("{0}={1}".format(k,config[k]))
            u.WriteTextArrayToFile(params_output_file,params_info)
        print("done dataset : " + dataset_dir)

def RunSvmClassifierOnVowelRecognitionDataset(root=r"C:\Users\shkhandu\OneDrive\Gatech\Courses\ML\DataSets\LetterRecognition"):
    pos_class="v"
    metric_fn = sl.ComputePrecisionRecallForPythonOutputFormat
    keys_to_keep=['dataset_instance','test_split','train_split','random_state','noise_perc','train_split_percent_used','imbalance_perc','kernel','C','gamma','degree','modelbuildtimesecs','modelevaltimesecs','numsupportvectors']
    classifier_fn = lambda x : RunSVMClassifier(x,positive_class_label=pos_class)
    id="vowel.svm_3_0"
    algo_folder='svm'
    force_computation=True
    exp.RunNEvaluateExperimentsOnDataSet(classifier_fn,root,id,metric_fn,algo_folder,keys_to_keep,pos_class,["i-0"],force_computation,evaluate_only=True)

def RunSvmClassifierOnCreditScreeningDataset(root=r"C:\Users\shkhandu\OneDrive\Gatech\Courses\ML\DataSets\CreditScreeningDataset"):
    pos_class="+"
    metric_fn = sl.ComputePrecisionRecallForPythonOutputFormat
    keys_to_keep=['dataset_instance','test_split','train_split','random_state','noise_perc','train_split_percent_used','imbalance_perc','kernel','C','gamma','degree','modelbuildtimesecs','modelevaltimesecs','numsupportvectors']
    classifier_fn = lambda x : RunSVMClassifier(x,['A1','A4','A5','A6','A7','A9','A10','A12','A13'],pos_class)
    id="credit.svm_3_0"
    algo_folder='svm'
    force_computation=True
    exp.RunNEvaluateExperimentsOnDataSet(classifier_fn,root,id,metric_fn,algo_folder,keys_to_keep,pos_class,["i-0"],force_computation,evaluate_only=True)

def main():
    RunSvmClassifierOnVowelRecognitionDataset(r"C:\Users\shwet\OneDrive\Gatech\Courses\ML\DataSets\LetterRecognition")
    RunSvmClassifierOnCreditScreeningDataset(r"C:\Users\shwet\OneDrive\Gatech\Courses\ML\DataSets\CreditScreeningDataset")

if __name__ == '__main__':
    main()