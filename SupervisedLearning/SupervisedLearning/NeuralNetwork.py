import glob
import os
import time
import ast

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

import ExperimentsRunner as exp
import SupervisedLearning as sl
import utils as u
from NeuralNetworkRunConfig import NeuralNetworkRunConfig as nnconfig


def RunNeuralNetClassifier(datasets_root_folder,one_hot_encoding_cols=None, positive_class_label=None,cv_file_format = None):
	file_extn = "csv"
	testfiles = glob.glob("{0}/*.test.{1}".format(datasets_root_folder,file_extn))

	for dataset_dir in u.Get_Subdirectories(datasets_root_folder):
		trainfile = glob.glob("{0}/*.train.{1}".format(dataset_dir,file_extn))[0]
		paramfile = glob.glob("{0}/*.params.txt".format(dataset_dir))[0]
		dt_root = u.PreparePath(dataset_dir+"/nnets",is_file=False)
		config_gen = nnconfig()
		config = config_gen.GetNextConfigAlongWithIdentifier()
		while(config is not None):
			id = config["id"]
			params_info = u.ReadLinesFromFile(paramfile)
			params_info_dict=sl.GetDictionary(params_info)
			run_output_dir = u.PreparePath("{0}/{1}".format(dt_root,id),is_file=False)
			params_output_file=u.PreparePath("{0}/{1}.params.txt".format(run_output_dir,id))
			model_output_file=u.PreparePath("{0}/{1}.model".format(run_output_dir,id))
			train_output_file=u.PreparePath("{0}/{1}.train.predictions.csv".format(run_output_dir,id))
			test_output_file=u.PreparePath("{0}/{1}.test.predictions.csv".format(run_output_dir,id))
			cv_results_file=u.PreparePath("{0}/{1}.grid_search_cv_results.csv".format(run_output_dir,id))
			model_output_file = u.PreparePath(
                "{0}/{1}.model".format(run_output_dir, id))
			# if(os.path.isfile(cv_results_file)):
			# 	config = config_gen.GetNextConfigAlongWithIdentifier()
			# 	continue
			config["trainset"] = trainfile
			config["class"]="last"
			config["trainpredictionoutputfile"]=train_output_file
			config["predictionoutputfile"] = config["trainpredictionoutputfile"]
			config["modeloutputfile"] = model_output_file
			config["testpredictionoutputfile"] = test_output_file

			data = pd.read_csv(trainfile)
			config["testset"] = testfiles[0]
			testdata = pd.read_csv(config["testset"])
			train_len = len(data)

			cols_to_ignore = set(one_hot_encoding_cols) if one_hot_encoding_cols is not None else set([])
			cols_to_ignore.add(data.columns[-1])
			cols_to_transform = [c for c in data.columns if c not in cols_to_ignore]
			scaler = StandardScaler()
			scaler.fit(data[cols_to_transform])
			data[cols_to_transform] = scaler.transform(data[cols_to_transform])
			testdata[cols_to_transform] = scaler.transform(testdata[cols_to_transform])

			all_data = pd.concat([data,testdata], axis=0, ignore_index=True)
			X_all,Y_all = PrepareDataAndLabel(all_data,positive_class_label,one_hot_encoding_cols)
			X = X_all[0:train_len,:]
			Y = Y_all[0:train_len]
			test_X=X_all[train_len:,:]
			test_Y=Y_all[train_len:]

			hidden_layers = [(10,),(30,),(50,),(70,)]
			init_learning_rates = [0.1,0.01,0.001,0.0001]
			alpha =[0.01,0.001,0.0001,0.00001]
			momentum = 0.9
			max_iter = 200
			early_stopping = config["earlystopping"]
			validation_fraction=0.3
			random_state = int(params_info_dict["random_state"])
			solver='sgd'

			#for doing 3-fold CV
			param_grid = {"alpha":alpha,"learning_rate_init":init_learning_rates,"hidden_layer_sizes":hidden_layers}
			classifier = MLPClassifier(
				activation="logistic",
				momentum=momentum,
				early_stopping = early_stopping,
				verbose=False,
				validation_fraction=validation_fraction,
				random_state=random_state,
				solver="sgd",
				max_iter=max_iter)
			cv_file = None
			if(cv_file_format is not None):
				cv_file = cv_file_format.format(id)
			if((cv_file is None) or (os.path.isfile(cv_file) == False)):
				gscv = GridSearchCV(classifier,param_grid,scoring='f1',n_jobs=3)
				gscv.fit(X,Y)
				_D = pd.DataFrame(gscv.cv_results_)
				best_params = gscv.best_params_
				_D.to_csv(cv_results_file)
			else:
				cv_results = pd.read_csv(cv_file)
				best_params = ast.literal_eval(cv_results[cv_results['rank_test_score']==1].iloc[0]['params'])
			# gscv = GridSearchCV(classifier,param_grid,scoring='f1',n_jobs=3)
			# gscv.fit(X,Y)
			# _D = pd.DataFrame(gscv.cv_results_)
			# _D.to_csv(cv_results_file)
			classifier = MLPClassifier(
				hidden_layer_sizes=best_params["hidden_layer_sizes"],
				activation="logistic",
				momentum=momentum,
				early_stopping = early_stopping,
				verbose=True,
				validation_fraction=validation_fraction,
				random_state=random_state,
				solver="sgd",
				max_iter=max_iter,
				learning_rate_init=best_params["learning_rate_init"],
				alpha=best_params["alpha"])
			start = time.clock()
			classifier.fit(X,Y)
			end = time.clock()

			config['momentum']=momentum
			config["hidden_layers"]="10;30;50;70"
			config["alphas"]=u.ConcatToStr(";",alpha)
			config["init_learning_rates"]=u.ConcatToStr(";",init_learning_rates)
			config["total_iter"]=classifier.n_iter_
			config["time_per_iter"]=(end - start) / classifier.n_iter_
			config["best_alpha"] = best_params["alpha"]
			config["best_hidden_layer_sizes"]=best_params["hidden_layer_sizes"][0]
			config["best_init_learning_rate"] = best_params["learning_rate_init"]
			config["loss_curve"] = u.ConcatToStr(";",classifier.loss_curve_)

			config["random_state"] = random_state
			config["modelbuildtimesecs"] = end-start
			# for train performance
			config["trainpredictionoutputfile"]=train_output_file
			train_predicted_Y = classifier.predict(X)
			output = pd.DataFrame({"actual":Y,"predicted":train_predicted_Y})
			output.to_csv(train_output_file,index=False)

			# now for test set
			config["predictionoutputfile"] = test_output_file

			u.WriteBinaryFile(model_output_file, classifier)
			
			#test_X,test_Y = PrepareDataAndLabel(data,positive_class_label,one_hot_encoding_cols)
			predicted_Y = classifier.predict(test_X)
			output = pd.DataFrame({"actual":test_Y,"predicted":predicted_Y})
			output.to_csv(test_output_file,index=False)

			config.pop('random_state',None) # since we already have that in params_info
			for k in config:
				params_info.append("{0}={1}".format(k,config[k]))
			u.WriteTextArrayToFile(params_output_file,params_info)
			config = config_gen.GetNextConfigAlongWithIdentifier()
		print("done dataset : " + dataset_dir)

def PrepareDataAndLabel(data,positive_class_label,one_hot_encoding_cols):
	label_col = data.columns[-1]
	Y = sl.ConvertLabelsToZeroOne(data[label_col],positive_class_label)
	data = data.drop(label_col,axis=1)
	if(one_hot_encoding_cols is not None):
		data = sl.GetOneHotEncodingForDataFrame(data,one_hot_encoding_cols)
	X = data.as_matrix()
	return X,Y

def NeuralNetExperiments():
	root = r"C:\Users\shkhandu\OneDrive\Gatech\Courses\ML\DataSets\CreditScreeningDataset\i-4_t-80_T-20"
	RunNeuralNetClassifier(root,['A1','A4','A5','A6','A7','A9','A10','A12','A13'],"+")

def RunNeuralNetsOnVowelRecognitionDataset(root=r"C:\Users\shkhandu\OneDrive\Gatech\Courses\ML\DataSets\LetterRecognition"):
	pos_class="v"
	metric_fn = sl.ComputePrecisionRecallForPythonOutputFormat
	keys_to_keep=['dataset_instance','test_split','train_split','random_state','train_split_percent_used','prune','modelbuildtimesecs','earlystopping','alphas','init_learning_rates','total_iter','time_per_iter','best_alpha','best_init_learning_rate','loss_curve','momentum','best_hidden_layer_sizes','hidden_layers']
	cv_file = root + r"/i-0_t-80_T-20/i-0_t-80_ts-100/nnets/{0}/{0}.grid_search_cv_results.csv"
	classifier_fn = lambda x : RunNeuralNetClassifier(x,positive_class_label=pos_class,cv_file_format=cv_file)
	id="vowel.nnet_3_0"
	algo_folder='nnets'
	force_computation=True
	exp.RunNEvaluateExperimentsOnDataSet(classifier_fn,root,id,metric_fn,algo_folder,keys_to_keep,pos_class,["i-0"],force_computation)

def RunNeuralNetsOnCreditScreeningDataset(root=r"C:\Users\shkhandu\OneDrive\Gatech\Courses\ML\DataSets\CreditScreeningDataset"):
	pos_class="+"
	metric_fn = sl.ComputePrecisionRecallForPythonOutputFormat
	keys_to_keep=['dataset_instance','test_split','train_split','random_state','train_split_percent_used','prune','modelbuildtimesecs','earlystopping','alphas','init_learning_rates','total_iter','time_per_iter','best_alpha','best_init_learning_rate','loss_curve','momentum','best_hidden_layer_sizes','hidden_layers']
	cv_file = root + r"/i-0_t-80_T-20/i-0_t-80_ts-100/nnets/{0}/{0}.grid_search_cv_results.csv"
	classifier_fn = lambda x : RunNeuralNetClassifier(x,positive_class_label=pos_class,one_hot_encoding_cols=['A1','A4','A5','A6','A7','A9','A10','A12','A13'],cv_file_format=cv_file)
	id="credit.nnet_3_0"
	algo_folder='nnets'
	force_computation=True
	exp.RunNEvaluateExperimentsOnDataSet(classifier_fn,root,id,metric_fn,algo_folder,keys_to_keep,pos_class,["i-0"],force_computation)

def main():
	RunNeuralNetsOnCreditScreeningDataset(r"C:\Users\shwet\OneDrive\Gatech\Courses\ML\DataSets\CreditScreeningDataset")
	RunNeuralNetsOnVowelRecognitionDataset(r"C:\Users\shwet\OneDrive\Gatech\Courses\ML\DataSets\LetterRecognition")

if __name__ == "__main__":
	main()