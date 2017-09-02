import glob
import os
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

import ExperimentsRunner as exp
import SupervisedLearning as sl
import utils as u
import NeuralNetwork as nnet

def GetIdForConfig(config):
    return "weights-{0}_neighbors-{1}".format(config['weights'], config['neighbors'])


def RunKNNClassifier(datasets_root_folder, nominal_value_columns=None, positive_class_label=None, metric_fn=None):
    file_extn = "csv"
    testfiles = glob.glob("{0}/*.test.{1}".format(datasets_root_folder, file_extn))
    for dataset_dir in u.Get_Subdirectories(datasets_root_folder):
        trainfile = glob.glob("{0}/*.train.{1}".format(dataset_dir, file_extn))[0]
        paramfile = glob.glob("{0}/*.params.txt".format(dataset_dir))[0]
        dt_root = u.PreparePath(dataset_dir + "/knn", is_file=False)
        config_gen = ParameterGrid(
            {'weights': ['uniform', 'distance'], 'neighbors': [5, 10, 20, 50]})
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
            # if(os.path.isfile(test_output_file)):
            #	config = config_gen.GetNextConfigAlongWithIdentifier()
            #	continue
            config["trainset"] = trainfile
            config["class"] = "last"
            config["trainpredictionoutputfile"] = train_output_file
            config["predictionoutputfile"] = config["trainpredictionoutputfile"]
            config["modeloutputfile"] = model_output_file
            config["testpredictionoutputfile"] = test_output_file

            data = pd.read_csv(trainfile)
            config["testset"] = testfiles[0]
            testdata = pd.read_csv(config["testset"])
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

            if(metric_fn is None):
                classifier = KNeighborsClassifier(config['neighbors'],config['weights'])
            else:
                classifier = KNeighborsClassifier(config['neighbors'],config['weights'],algorithm='auto',metric='pyfunc',metric_params={'func': metric_fn})
            start = time.clock()
            classifier.fit(X,Y)
            end = time.clock()
            print(end-start)
            config["modelbuildtimesecs"] = end-start
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
    keys_to_keep=['dataset_instance','test_split','train_split','random_state','noise_perc','train_split_percent_used','imbalance_perc','weights','neighbors','modelbuildtimesecs','modelevaltimesecs']
    classifier_fn = lambda x : RunKNNClassifier(x,positive_class_label=pos_class)
    id="vowel.knn_1_all"
    algo_folder='knn'
    force_computation=True
    exp.RunNEvaluateExperimentsOnDataSet(classifier_fn,root,id,metric_fn,algo_folder,keys_to_keep,pos_class,[],force_computation)

def RunKnnClassifierOnCreditScreeningDataset(root=r"C:\Users\shkhandu\OneDrive\Gatech\Courses\ML\DataSets\CreditScreeningDataset"):
	pos_class="+"
	metric_fn = sl.ComputePrecisionRecallForPythonOutputFormat
	keys_to_keep=['dataset_instance','test_split','train_split','random_state','noise_perc','train_split_percent_used','imbalance_perc','weights','neighbors','modelbuildtimesecs','modelevaltimesecs']
	classifier_fn = lambda x : RunKNNClassifier(x,['A1','A4','A5','A6','A7','A9','A10','A12','A13'],pos_class)
	id="credit.knn_1_all"
	algo_folder='knn'
	force_computation=True
	exp.RunNEvaluateExperimentsOnDataSet(classifier_fn,root,id,metric_fn,algo_folder,keys_to_keep,pos_class,[],force_computation)

def main():
    RunKnnClassifierOnCreditScreeningDataset(r"C:\Users\shwet\OneDrive\Gatech\Courses\ML\DataSets\CreditScreeningDataset")
    # RunKnnClassifierOnVowelRecognitionDataset(r"C:\Users\shwet\OneDrive\Gatech\Courses\ML\DataSets\LetterRecognition")

if __name__ == '__main__':
    main()