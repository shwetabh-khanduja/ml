"""
java -classpath "C:\Program Files\Weka-3-8\weka.jar" weka.classifiers.trees.J48 -t train.arff -T test.arff -M 2 -c first -U -classifications "weka.classifiers.evaluation.output.prediction.CSV -file c:\temp\out.csv -p 1"
java -classpath "C:\Program Files\Weka-3-8\weka.jar" weka.classifiers.meta.FilteredClassifier -F "weka.filters.unsupervised.instance.RemovePercentage -P 10 -V" -t train.arff -T test.arff -c first -classifications "weka.classifiers.evaluation.output.prediction.CSV -file c:\temp\out.csv -p 1" -W weka.classifiers.trees.J48 -- -M 2 -U
"""

import scipy as sc
import pandas as pd
import utils as u
from sklearn.model_selection import train_test_split
import glob
import DecisionTreeRunConfig as dtc
import timeit
import os

def LoadWineDataSet(root = r"C:\Users\shkhandu\OneDrive\Gatech\Courses\ML\DataSets\WineDataset"):
	datafile = root+r"\wine.csv"
	headerfile = root+r"\schema.txt"
	data = pd.read_csv(datafile,header="infer")
	col_names = u.ReadLinesFromFile(headerfile)[0].split(',')
	data.columns = col_names
	return data

def GenerateStratifiedTrainTestSplits(dataframe,stratification_keys,train_size,test_size,random_state):
	train, test = train_test_split(dataframe,
								train_size=train_size,
								test_size=test_size,
								random_state=random_state,
								stratify=stratification_keys)
	return [train,test]

def CreateTrainTestAndValidationPartitions(
	dataframe,
	stratification_keys_array,
	train_size,
	test_size,
	random_state,
	validation_size=-1,
	train_out_file=None,
	test_out_file=None,
	validation_out_file=None,
	arff_format_predata_lines=None,
	include_header=True):
	"""
	validation size is computed after what is left from train test split i.e. only on the train split
	"""
	train, test = GenerateStratifiedTrainTestSplits(dataframe,dataframe[stratification_keys_array],train_size,test_size,random_state)
	validation = None
	if(validation_size > 0):
		train,validation = GenerateStratifiedTrainTestSplits(train,train[stratification_keys_array],1-validation_size,validation_size,random_state)
	
	def write_to_file(data_to_write,filepath):
		file = u.PreparePath(filepath)
		data_to_write.to_csv(file,index=False,header=(include_header & (arff_format_predata_lines is None)))
		if(arff_format_predata_lines is not None):
			data = []
			data.extend(arff_format_predata_lines)
			data.extend(u.ReadLinesFromFile(file))
			u.WriteTextArrayToFile(file,data)

	if(train_out_file is not None):
		write_to_file(train,train_out_file)
	if(test_out_file is not None):
		write_to_file(test,test_out_file)
	if(validation_out_file is not None):
		write_to_file(validation,validation_out_file)
	return train,test,validation

def LoadCharacterRecognitionDataset(file,arff_attr_file = None):
	data = pd.read_csv(file)
	if(arff_attr_file is not None):
		arff_attrs = u.ReadLinesFromFile(arff_attr_file)
		return data,arff_attrs
	return data,None

def GenerateDatasetSplits(
	rootFolder,
	dataset_folder_prefix,
	dataset,
	test_ratio,
	train_ratio,
	validation_ratio,
	train_size_percentages,
	class_col,
	random_state,
	arff_attr_info=None):
	"""
	train_size_percentages is a list of intergers specifying the
	percent of train set to be taken while preparing the dataset

	test_ratio,train_ratio,validation_ratio : numbers in percentages
	"""
	dataset_root = u.PreparePath("{0}/i-{1}_t-{2}_T-{3}".format(rootFolder,dataset_folder_prefix,train_ratio,test_ratio))
	train,test,validation = CreateTrainTestAndValidationPartitions(dataset,class_col,train_ratio/100,test_ratio/100,random_state,validation_ratio/100)
	test_output_file_csv = u.PreparePath("{0}/i-{1}.test.csv".format(dataset_root,dataset_folder_prefix))
	test.to_csv(test_output_file_csv,index=False)
	if(arff_attr_info is not None):
		test_output_file_arff = u.PreparePath("{0}/i-{1}.test.arff".format(dataset_root,dataset_folder_prefix))
		CreateArffFileFromCsv(arff_attr_info,test_output_file_arff,test_output_file_csv,True,True)

	# now creating the train set partitions
	for train_set_size in train_size_percentages:
		folder_path = u.PreparePath("{0}/i-{1}_t-{2}_ts-{3}".format(dataset_root,dataset_folder_prefix,train_ratio,train_set_size))
		csv_output_file = u.PreparePath("{0}/i-{1}_t-{2}_ts-{3}.train.csv".format(folder_path,dataset_folder_prefix,train_ratio,train_set_size))
		rows_to_keep = int(len(train) * train_set_size / 100)
		train.head(rows_to_keep).to_csv(csv_output_file,index=False)
		if(arff_attr_info is not None):
			arff_output_file = u.PreparePath("{0}/i-{1}_t-{2}_ts-{3}.train.arff".format(folder_path,dataset_folder_prefix,train_ratio,train_set_size))
			CreateArffFileFromCsv(arff_attr_info,arff_output_file,csv_output_file,True,True)
		
		# writing the parameters
		params_info = ["dataset_instance={0}".format(dataset_folder_prefix),
						"test_split={0}".format(test_ratio),
						"train_split={0}".format(train_ratio),
						"random_state={0}".format(random_state),
						"class_col={0}".format(class_col),
						"train_split_percent_used={0}".format(train_set_size)]
		params_out_file = u.PreparePath("{0}/i-{1}_t-{2}_ts-{3}.params.txt".format(folder_path,dataset_folder_prefix,train_ratio,train_set_size))
		u.WriteTextArrayToFile(params_out_file,params_info)

def CreateArffFileFromCsv(arff_attr_info, arff_file_path, data_text_array, isFile=False,hasHeader=True):
	arff_data = []
	arff_data.extend(arff_attr_info)
	data_text_array = u.ReadLinesFromFile(data_text_array) if(isFile) else data_text_array
	data_text_array = data_text_array[1:] if (isFile & hasHeader) else data_text_array
	arff_data.extend(data_text_array)
	file = u.PreparePath(arff_file_path)
	u.WriteTextArrayToFile(file,arff_data)

def GenerateVowelRecognitionDataSetSplits():
	rootFolder=r"C:\Users\shkhandu\OneDrive\Gatech\Courses\ML\DataSets\LetterRecognition"
	id=0
	train_perc=80
	test_perc=20
	vowelDataFile=u.PreparePath("{0}/vowel-recongnition-dataset.csv".format(rootFolder))
	arff_attrs_file = u.PreparePath("{0}/vowel.txt".format(rootFolder))
	data,arff_attrs = LoadCharacterRecognitionDataset(vowelDataFile,arff_attrs_file)
	random=0
	train_size_percs = [20,30,40,50,60,70,80,90,100]
	GenerateDatasetSplits(rootFolder,id,data,test_perc,train_perc,0,train_size_percs,"vowel",random,arff_attrs)

def GenerateCreditScreeningDataSetSplits():
	rootFolder=r"C:\Users\shkhandu\OneDrive\Gatech\Courses\ML\DataSets\CreditScreeningDataset"
	id=0
	train_perc=80
	test_perc=20
	vowelDataFile=u.PreparePath("{0}/data_no_missing_values.csv".format(rootFolder))
	arff_attrs_file = u.PreparePath("{0}/arff_attrs.txt".format(rootFolder))
	data,arff_attrs = LoadCreditScreeningData(vowelDataFile,arff_attrs_file)
	random=0
	train_size_percs = [20,30,40,50,60,70,80,90,100]
	GenerateDatasetSplits(rootFolder,id,data,test_perc,train_perc,0,train_size_percs,"A16",random,arff_attrs)

def RunDecisionTrees(datasets_root_folder,use_arff_files=True):
	file_extn = "arff" if use_arff_files else ".csv"
	testfiles = glob.glob("{0}/*.test.{1}".format(datasets_root_folder,file_extn))
	weka_jar_path = "C:\Program Files\Weka-3-8\weka.jar"

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
			config["modelbuildtimesecs"] = timeit.timeit(lambda: os.system(cmd),number=1)
			
			# now for test set
			config["predictionoutputfile"] = test_output_file
			config["testset"] = testfiles[0]
			cmd = config_gen.GenerateWekaCommandline(config)
			config["modelevaltimesecs"] = timeit.timeit(lambda : os.system(cmd),number=1)

			for k in config:
				params_info.append("{0}={1}".format(k,config[k]))
			u.WriteTextArrayToFile(params_output_file,params_info)
			config = config_gen.GetNextConfigAlongWithIdentifier()
		print("done dataset : " + dataset_dir)

def LoadCreditScreeningData(file,arff_attr_file=None):
	data = pd.read_csv(file)
	if(arff_attr_file is not None):
		arff_attrs = u.ReadLinesFromFile(arff_attr_file)
		return data,arff_attrs
	return data,None

if __name__=="__main__":
	GenerateCreditScreeningDataSetSplits()
	RunDecisionTrees(r"C:\Users\shkhandu\OneDrive\Gatech\Courses\ML\DataSets\LetterRecognition\i-0_t-80_T-20")
	GenerateVowelRecognitionDataSetSplits()
	dataset_root = r"C:\Users\shkhandu\OneDrive\Gatech\Courses\ML\DataSets\WineDataset"
	data = LoadWineDataSet(dataset_root)
	run_instance = 0
	train_out_file = dataset_root + r"\splits\{0}\train.arff".format(run_instance)
	validatation_out_file = dataset_root +  r"\splits\{0}\validatation.arff".format(run_instance)
	test_out_file = dataset_root + r"\splits\{0}\test.arff".format(run_instance)
	arff_sections = u.ReadLinesFromFile(dataset_root+r"\arff_sections.txt")
	splits = CreateTrainTestAndValidationPartitions(data,"class",0.7,0.3,run_instance,0.2,train_out_file,test_out_file,validatation_out_file,arff_sections)