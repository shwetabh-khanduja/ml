import scipy as sc
import numpy as np
import pandas as pd
import utils as u
import DecisionTreeRunConfig as dtc
from sklearn.model_selection import train_test_split
import glob
import timeit
import os
import subprocess as sb
import random
from sklearn.metrics import precision_recall_fscore_support
import SupervisedLearning as sl

def RunDecisionTrees(datasets_root_folder,weka_jar_path="C:\Program Files\Weka-3-8\weka.jar",use_arff_files=True):
	file_extn = "arff" if use_arff_files else ".csv"
	testfiles = glob.glob("{0}/*.test.{1}".format(datasets_root_folder,file_extn))

	for dataset_dir in u.Get_Subdirectories(datasets_root_folder):
		trainfile = glob.glob("{0}/*.train.{1}".format(dataset_dir,file_extn))[0]
		paramfile = glob.glob("{0}/*.params.txt".format(dataset_dir,file_extn))[0]
		dt_root = u.PreparePath(dataset_dir+"/dt",is_file=False)
		config_gen = dtc.DecisionTreeRunConfig(prune=[True,False],numfolds=[3],minleafsize=[2])
		config = config_gen.GetNextConfigAlongWithIdentifier()
		while(config is not None):
			id = config["id"]
			params_info = u.ReadLinesFromFile(paramfile)
			run_output_dir = u.PreparePath("{0}/{1}".format(dt_root,id),is_file=False)
			params_output_file=u.PreparePath("{0}/{1}.params.txt".format(run_output_dir,id))
			model_output_file=u.PreparePath("{0}/{1}.model".format(run_output_dir,id))
			train_output_file=u.PreparePath("{0}/{1}.train.predictions.csv".format(run_output_dir,id))
			test_output_file=u.PreparePath("{0}/{1}.test.predictions.csv".format(run_output_dir,id))

			config["wekajar"] = weka_jar_path
			config["trainset"] = trainfile
			config["class"]="last"
			config["trainpredictionoutputfile"]=train_output_file
			config["predictionoutputfile"] = config["trainpredictionoutputfile"]
			config["modeloutputfile"] = model_output_file
			config["testpredictionoutputfile"] = test_output_file

			# for every config there has to be a train prediction and test prediction
			cmd = config_gen.GenerateWekaCommandline(config)
			config["modelbuildtimesecs"] = timeit.timeit(lambda: sl.RunCmdWithoutConsoleWindow(cmd),number=1)
			
			# now for test set
			config["predictionoutputfile"] = test_output_file
			config["testset"] = testfiles[0]
			cmd = config_gen.GenerateWekaCommandline(config)
			config["modelevaltimesecs"] = timeit.timeit(lambda : sl.RunCmdWithoutConsoleWindow(cmd),number=1)

			for k in config:
				params_info.append("{0}={1}".format(k,config[k]))
			u.WriteTextArrayToFile(params_output_file,params_info)
			config = config_gen.GetNextConfigAlongWithIdentifier()
		print("done dataset : " + dataset_dir)

def EvaluateDecisionTrees(datasets_root_folder,params_to_keep,positive_class,evaluation_output_filename="performance.csv"):
	headers=[]
	headers.extend(params_to_keep)
	headers.extend(['istrain','p','r','f'])
	headers = ",".join(headers)
	evals = []
	evals.append(headers)
	for directory in u.Get_Subdirectories(datasets_root_folder):
		#each directory is a dataset directory
		dt_output_dir = "{0}/dt".format(directory)
		for run_output_folder in u.Get_Subdirectories(dt_output_dir):
			#read params file
			params_file_path = glob.glob("{0}/*.params.txt".format(run_output_folder))[0]
			params = sl.GetDictionary(u.ReadLinesFromFile(params_file_path))
			values = []
			for k in params_to_keep:
				if(k in params):
					values.append(str(params[k]))
				else:
					values.append(str(np.NaN))
			p,r,f=sl.GetPrecisionRecallForWekaOutputFile(params["trainpredictionoutputfile"],positive_class)
			train_performance_values = ",".join(values)
			train_performance_values = "{0},1,{1},{2},{3}".format(",".join(values),str(p),str(r),str(f))
			p,r,f=sl.GetPrecisionRecallForWekaOutputFile(params["testpredictionoutputfile"],positive_class)
			test_performance_values = ",".join(values)
			test_performance_values = "{0},0,{1},{2},{3}".format(",".join(values),str(p),str(r),str(f))
			evals.append(train_performance_values)
			evals.append(test_performance_values)
	u.WriteTextArrayToFile(u.PreparePath("{0}/{1}".format(datasets_root_folder,evaluation_output_filename)),evals)

def EstimatePerformanceOnCreditScreeningDataset(credit_screening_data_set_root,id,datasets_to_ignore=[],force=False):
	agg_eval_file = "{0}/eval_agg.{1}.csv".format(credit_screening_data_set_root,id)
	eval_files = []
	for dataset in u.Get_Subdirectories(credit_screening_data_set_root):
		ignore=False
		for filter in datasets_to_ignore:
			if(filter in dataset):
				ignore=True
				break
		if(ignore):
			continue
		eval_file = "eval.{0}.csv".format(id)
		eval_file_full = dataset+"/"+eval_file
		eval_files.append(eval_file_full)
		if(os.path.isfile(eval_file_full)):
			if(force == False):
				continue
		RunDecisionTrees(dataset) #this root is for the various configs of the dataset
		EvaluateDecisionTrees(dataset,['dataset_instance','test_split','train_split','random_state','noise_perc','train_split_percent_used','imbalance_perc','prune','modelbuildtimesecs'],'+',eval_file)
	df = None
	for file in eval_files:
		d = pd.read_csv(file)
		df = pd.concat([df,d], axis=0, ignore_index=True) if df is not None else d
	df.to_csv(agg_eval_file,index=False)

def EstimatePerformanceOnLetterRecognitionDataset(letter_recognition_data_set_root,id,datasets_to_ignore=[],force=False):
	agg_eval_file = "{0}/eval_agg.{1}.csv".format(letter_recognition_data_set_root,id)
	eval_files = []
	for dataset in u.Get_Subdirectories(letter_recognition_data_set_root):
		ignore=False
		for filter in datasets_to_ignore:
			if(filter in dataset):
				ignore=True
				break
		if(ignore):
			continue
		eval_file = "eval.{0}.csv".format(id)
		eval_file_full = dataset+"/"+eval_file
		eval_files.append(eval_file_full)
		if(os.path.isfile(eval_file_full)):
			if(force == False):
				continue
		RunDecisionTrees(dataset) #this root is for the various configs of the dataset
		EvaluateDecisionTrees(dataset,['dataset_instance','test_split','train_split','random_state','noise_perc','train_split_percent_used','imbalance_perc','prune','modelbuildtimesecs'],'v',eval_file)
	df = None
	for file in eval_files:
		d = pd.read_csv(file)
		df = pd.concat([df,d], axis=0, ignore_index=True) if df is not None else d
	df.to_csv(agg_eval_file,index=False)

def RunNEvaluateExperimentsOnDataSet(
	classifer_fn,
	dataset_root,
	id,
	metric_calculation_fn,
	algo_folder,
	params_to_keep,
	positive_class,
	datasets_to_run_on=[],
	force=False):
	"""
	classifier_fn : This is the main classifier function that is called by passing in 
					the dataset

	dataset_root :  Is the root directory that contains various dataset instances.
					Dataset instance corresponds to the train_size, noisy and imbalance
					datasets, total 15. Each dataset instance is then further sub divided
					into multiple train sets depending upon the dataset instance parameter
					configuration like noise percentage etc.

	id : Is the unique identifier for this call of RunNEvaluateExperimentsOnDataSet. All the
		 output files created will have this id in them

	metric_calculation_fn : Metrics computation function that is called by passing in the path
							to prediction outputs and the positive class label
					 
	algo_folder : This is the name of the folder inside each train set folder, where the
				  algorithm will store its output

	params_to_keep : List of columns to keep in the final aggregated metrics file

	positive_class : Label of the positive class

	datasets_to_run_on : Is a list of substrings. Only on those datasets we run that contain
						 any of these filters as substrings in
	"""
	agg_eval_file = "{0}/eval_agg.{1}.csv".format(dataset_root,id)
	eval_files = []
	for dataset in u.Get_Subdirectories(dataset_root):
		ignore=True
		if(len(datasets_to_run_on) == 0):
			ignore = False
		else:
			for filter in datasets_to_run_on:
				if(filter in dataset):
					ignore=False
					break
		if(ignore):
			continue
		eval_file = "eval.{0}.csv".format(id)
		eval_file_full = dataset+"/"+eval_file
		eval_files.append(eval_file_full)
		if(os.path.isfile(eval_file_full)):
			if(force == False):
				continue
		classifer_fn(dataset) #this root is for the various configs of the dataset
		EvaluateExperiments(dataset,params_to_keep,positive_class,metric_calculation_fn,eval_file,algo_folder)
	df = None
	for file in eval_files:
		d = pd.read_csv(file)
		df = pd.concat([df,d], axis=0, ignore_index=True) if df is not None else d
	df.to_csv(agg_eval_file,index=False)

def EvaluateExperiments(
	datasets_root_folder,
	params_to_keep,
	positive_class,
	metric_calculation_fn,
	evaluation_output_filename="performance.csv",
	algo_folder="dt"):

	headers=[]
	headers.extend(params_to_keep)
	headers.extend(['istrain','p','r','f'])
	headers = ",".join(headers)
	evals = []
	evals.append(headers)
	for directory in u.Get_Subdirectories(datasets_root_folder):
		#each directory is a dataset directory
		dt_output_dir = "{0}/{1}".format(directory,algo_folder)
		for run_output_folder in u.Get_Subdirectories(dt_output_dir):
			#read params file
			params_file_path = glob.glob("{0}/*.params.txt".format(run_output_folder))[0]
			params = sl.GetDictionary(u.ReadLinesFromFile(params_file_path))
			values = []
			for k in params_to_keep:
				if(k in params):
					values.append(str(params[k]))
				else:
					values.append(str(np.NaN))
			p,r,f=metric_calculation_fn(params["trainpredictionoutputfile"],positive_class)
			train_performance_values = ",".join(values)
			train_performance_values = "{0},1,{1},{2},{3}".format(",".join(values),str(p),str(r),str(f))
			p,r,f=metric_calculation_fn(params["testpredictionoutputfile"],positive_class)
			test_performance_values = ",".join(values)
			test_performance_values = "{0},0,{1},{2},{3}".format(",".join(values),str(p),str(r),str(f))
			evals.append(train_performance_values)
			evals.append(test_performance_values)
	u.WriteTextArrayToFile(u.PreparePath("{0}/{1}".format(datasets_root_folder,evaluation_output_filename)),evals)

def DecisionTreesExperiments():
	root = r"C:\Users\shwet\OneDrive\Gatech\Courses\ML\DataSets"
	#EstimatePerformanceOnCreditScreeningDataset(root+r"\CreditScreeningDataset","all_0_4",[])
	# EstimatePerformanceOnLetterRecognitionDataset(root+r"\LetterRecognition","all_0_4",[],True)
	Datasets = [
			 r"\LetterRecognition\i-0_t-80_T-20",
			 r"\LetterRecognition\i-imb0_t-80_T-20",
			 r"\LetterRecognition\i-noise1_t-80_T-20",
			 r"\CreditScreeningDataset\i-0_t-80_T-20",
			 r"\CreditScreeningDataset\i-imb0_t-80_T-20",
			 r"\CreditScreeningDataset\i-noise1_t-80_T-20"
			 r"\CreditScreeningDataset\i-noise2_t-80_T-20",
			 r"\LetterRecognition\i-noise2_t-80_T-20",
			 ]

	
		
	for dataset in Datasets:
		RunDecisionTrees(root+dataset)

	# EvaluateDecisionTrees(root + r"\CreditScreeningDataset\i-noise2_t-80_T-20",['dataset_instance','test_split','train_split','random_state','noise_perc','prune','modelbuildtimesecs'],'+',"csd_nosie2_performance.csv")
	##EvaluateDecisionTrees(root + r"\CreditScreeningDataset\i-noise1_t-80_T-20",['dataset_instance','test_split','train_split','random_state','noise_perc','prune','modelbuildtimesecs'],'+',"csd_nosie1_performance.csv")
	##EvaluateDecisionTrees(root + r"\CreditScreeningDataset\i-imb0_t-80_T-20",['dataset_instance','test_split','train_split','random_state','imbalance_perc','prune','modelbuildtimesecs'],'+',"csd_imb_performance.csv")
	##EvaluateDecisionTrees(root + r"\CreditScreeningDataset\i-0_t-80_T-20",['dataset_instance','test_split','train_split','random_state','train_split_percent_used','prune','modelbuildtimesecs'],'+',"csd_trainsize_performance.csv")
	
	#EvaluateDecisionTrees(root + r"\LetterRecognition\i-noise2_t-80_T-20",['dataset_instance','test_split','train_split','random_state','noise_perc','prune','modelbuildtimesecs'],'v',"lr_noise2_performance.csv")
	##EvaluateDecisionTrees(root + r"\LetterRecognition\i-0_t-80_T-20",['dataset_instance','test_split','train_split','random_state','train_split_percent_used','prune','modelbuildtimesecs'],'v',"lr_trainsize_performance.csv")
	##EvaluateDecisionTrees(root + r"\LetterRecognition\i-noise1_t-80_T-20",['dataset_instance','test_split','train_split','random_state','noise_perc','prune','modelbuildtimesecs'],'v',"lr_noise1_performance.csv")
	##EvaluateDecisionTrees(root + r"\LetterRecognition\i-imb0_t-80_T-20",['dataset_instance','test_split','train_split','random_state','imbalance_perc','prune','modelbuildtimesecs'],'v',"lr_imb_performance.csv")
	
if __name__=="__main__":
	DecisionTreesExperiments()
