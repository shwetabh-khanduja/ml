from sklearn.neural_network import MLPClassifier
from NeuralNetworkRunConfig import NeuralNetworkRunConfig as nnconfig
import pandas as pd
import SupervisedLearning as sl
import numpy as np
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import GridSearchCV
import glob
import utils as u
import time
import DecisionTree as dt
import os

def RunNeuralNetClassifier(datasets_root_folder,one_hot_encoding_cols=None, positive_class_label=None):
	file_extn = "csv"
	testfiles = glob.glob("{0}/*.test.{1}".format(datasets_root_folder,file_extn))

	for dataset_dir in u.Get_Subdirectories(datasets_root_folder):
		trainfile = glob.glob("{0}/*.train.{1}".format(dataset_dir,file_extn))[0]
		paramfile = glob.glob("{0}/*.params.txt".format(dataset_dir,file_extn))[0]
		dt_root = u.PreparePath(dataset_dir+"/nnets",is_file=False)
		config_gen = nnconfig()
		config = config_gen.GetNextConfigAlongWithIdentifier()
		while(config is not None):
			id = config["id"]
			params_info = u.ReadLinesFromFile(paramfile)
			run_output_dir = u.PreparePath("{0}/{1}".format(dt_root,id),is_file=False)
			params_output_file=u.PreparePath("{0}/{1}.params.txt".format(run_output_dir,id))
			model_output_file=u.PreparePath("{0}/{1}.model".format(run_output_dir,id))
			train_output_file=u.PreparePath("{0}/{1}.train.predictions.csv".format(run_output_dir,id))
			test_output_file=u.PreparePath("{0}/{1}.test.predictions.csv".format(run_output_dir,id))
			#if(os.path.isfile(test_output_file)):
			#	config = config_gen.GetNextConfigAlongWithIdentifier()
			#	continue
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
			random_state = np.random.randint(0,1000)
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
			gscv = GridSearchCV(classifier,param_grid,n_jobs=2)
			gscv.fit(X,Y)

			classifier = MLPClassifier(
				hidden_layer_sizes=gscv.best_params_["hidden_layer_sizes"],
				activation="logistic",
				momentum=momentum,
				early_stopping = early_stopping,
				verbose=True,
				validation_fraction=validation_fraction,
				random_state=random_state,
				solver="sgd",
				max_iter=max_iter,
				learning_rate_init=gscv.best_params_["learning_rate_init"],
				alpha=gscv.best_params_["alpha"])
			start = time.clock();
			classifier.fit(X,Y)
			end = time.clock();

			config['momentum']=momentum
			config["hidden_layers"]="10;30;50;70"
			config["alphas"]=u.ConcatToStr(";",alpha)
			config["init_learning_rates"]=u.ConcatToStr(";",init_learning_rates)
			config["total_iter"]=classifier.n_iter_
			config["time_per_iter"]=(end - start) / classifier.n_iter_
			config["best_alpha"] = gscv.best_params_["alpha"]
			config["best_hidden_layer_sizes"]=gscv.best_params_["hidden_layer_sizes"][0]
			config["best_init_learning_rate"] = gscv.best_params_["learning_rate_init"]
			config["loss_curve"] = u.ConcatToStr(";",classifier.loss_curve_)

			config["random_state"] = random_state
			config["modelbuildtimesecs"] = 0
			# for train performance
			config["trainpredictionoutputfile"]=train_output_file
			train_predicted_Y = classifier.predict(X)
			output = pd.DataFrame({"actual":Y,"predicted":train_predicted_Y})
			output.to_csv(train_output_file,index=False)

			# now for test set
			config["predictionoutputfile"] = test_output_file
			
			#test_X,test_Y = PrepareDataAndLabel(data,positive_class_label,one_hot_encoding_cols)
			predicted_Y = classifier.predict(test_X)
			output = pd.DataFrame({"actual":test_Y,"predicted":predicted_Y})
			output.to_csv(test_output_file,index=False)

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
	keys_to_keep=['dataset_instance','test_split','train_split','random_state','noise_perc','train_split_percent_used','imbalance_perc','prune','modelbuildtimesecs','earlystopping','alphas','init_learning_rates','total_iter','time_per_iter','best_alpha','best_init_learning_rate','loss_curve','momentum','best_hidden_layer_sizes','hidden_layers']
	classifier_fn = lambda x : RunNeuralNetClassifier(x,positive_class_label=pos_class)
	id="nnet_1_all"
	algo_folder='nnets'
	force_computation=True
	dt.RunNEvaluateExperimentsOnDataSet(classifier_fn,root,id,metric_fn,algo_folder,keys_to_keep,pos_class,[],force_computation)

def RunNeuralNetsOnCreditScreeningDataset(root=r"C:\Users\shkhandu\OneDrive\Gatech\Courses\ML\DataSets\CreditScreeningDataset"):
	pos_class="+"
	metric_fn = sl.ComputePrecisionRecallForPythonOutputFormat
	keys_to_keep=['dataset_instance','test_split','train_split','random_state','noise_perc','train_split_percent_used','imbalance_perc','prune','modelbuildtimesecs','earlystopping','alphas','init_learning_rates','total_iter','time_per_iter','best_alpha','best_init_learning_rate','loss_curve','momentum','best_hidden_layer_sizes','hidden_layers']
	classifier_fn = lambda x : RunNeuralNetClassifier(x,positive_class_label=pos_class,one_hot_encoding_cols=['A1','A4','A5','A6','A7','A9','A10','A12','A13'])
	id="nnet_1_all"
	algo_folder='nnets'
	force_computation=True
	dt.RunNEvaluateExperimentsOnDataSet(classifier_fn,root,id,metric_fn,algo_folder,keys_to_keep,pos_class,[],force_computation)

if __name__ == "__main__":
	RunNeuralNetsOnVowelRecognitionDataset()
	RunNeuralNetsOnCreditScreeningDataset()
	