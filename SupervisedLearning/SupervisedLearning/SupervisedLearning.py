import scipy as sc
import numpy as np
import pandas as pd
import utils as u
from sklearn.model_selection import train_test_split
import glob
import DecisionTreeRunConfig as dtc
import timeit
import os
import subprocess as sb
import random
from sklearn.metrics import precision_recall_fscore_support

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
	if(validation is not None):
		validation_output_file_csv = u.PreparePath("{0}/i-{1}.test.csv".format(dataset_root,dataset_folder_prefix))
		validation.to_csv(validation_output_file_csv,index=False)
		test_output_file_csv = u.PreparePath("{0}/i-{1}.realtest.csv".format(dataset_root,dataset_folder_prefix))
		test.to_csv(test_output_file_csv,index=False)
	else:
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

# def GenerateVowelRecognitionDataSetSplits():
# 	rootFolder=r"C:\Users\shkhandu\OneDrive\Gatech\Courses\ML\DataSets\LetterRecognition"
# 	id=0
# 	train_perc=80
# 	test_perc=20
# 	vowelDataFile=u.PreparePath("{0}/vowel-recongnition-dataset.csv".format(rootFolder))
# 	arff_attrs_file = u.PreparePath("{0}/vowel.txt".format(rootFolder))
# 	data,arff_attrs = LoadCharacterRecognitionDataset(vowelDataFile,arff_attrs_file)
# 	r=0
# 	train_size_percs = [20,30,40,50,60,70,80,90,100]
# 	#GenerateDatasetSplits(rootFolder,id,data,test_perc,train_perc,0,train_size_percs,"vowel",random,arff_attrs)
# 	imbalance_percs = [90,10,20,30,40,50,70,5,100]
# 	minority_class = "v"
# 	GenerateDatasetSplitsForClassImbalance(rootFolder,"imb"+str(id),data,test_perc,train_perc,0,imbalance_percs,"vowel",minority_class,500,r,arff_attrs)

def GenerateVowelRecognitionDataSetSplits(
	rootFolder,
	id,
	train_perc,
	test_perc,
	random_state,
	train_size_percs = None,
	imbalance_percs = None,
	noise_percs = None,
	class_col_name="vowel",
	min_minority_class_samples_to_keep=500,
	validation_perc=0):

	vowelDataFile=u.PreparePath("{0}/vowel-recongnition-dataset.csv".format(rootFolder))
	arff_attrs_file = u.PreparePath("{0}/vowel.txt".format(rootFolder))
	data,arff_attrs = LoadCharacterRecognitionDataset(vowelDataFile,arff_attrs_file)

	minority_class = "v"
	flip_fn = lambda x : "c" if(x == "v") else "c"
	if(train_size_percs is not None):
		GenerateDatasetSplits(rootFolder,id,data,test_perc,train_perc,validation_perc,train_size_percs,class_col_name,random_state,arff_attrs)
	if(imbalance_percs is not None):
		GenerateDatasetSplitsForClassImbalance(rootFolder,"imb"+str(id),data,test_perc,train_perc,0,imbalance_percs,class_col_name,minority_class,min_minority_class_samples_to_keep,random_state,arff_attrs)
	if(noise_percs is not None):
		GenerateDatasetSplitsForWithNoise(rootFolder,"noise" + str(id),data,test_perc,train_perc,0,noise_percs,class_col_name,flip_fn,random_state,arff_attrs)

def GenerateCreditScreeningDataSetSplits(
	rootFolder,
	id,
	train_perc,
	test_perc,
	random_state,
	train_size_percs = None,
	imbalance_percs = None,
	noise_percs = None,
	class_col_name="A16",
	min_minority_class_samples_to_keep=10,
	train=None,
	test=None,
	validation_perc = 0):

	#rootFolder=r"C:\Users\shkhandu\OneDrive\Gatech\Courses\ML\DataSets\CreditScreeningDataset"
	#id=0
	#train_perc=80
	#test_perc=20
	vowelDataFile=u.PreparePath("{0}/data_no_missing_values.csv".format(rootFolder))
	arff_attrs_file = u.PreparePath("{0}/arff_attrs.txt".format(rootFolder))
	data,arff_attrs = LoadCreditScreeningData(vowelDataFile,arff_attrs_file)
	#random=0
	#train_size_percs = [20,30,40,50,60,70,80,90,100]
	#imbalance_percs = [90,10,20,30,40,50,70,5,100]
	minority_class = "+"
	flip_fn = lambda x : "-" if(x == "+") else "+"
	if(train_size_percs is not None):
		GenerateDatasetSplits(rootFolder,id,data,test_perc,train_perc,validation_perc,train_size_percs,class_col_name,random_state,arff_attrs)
	if(imbalance_percs is not None):
		GenerateDatasetSplitsForClassImbalance(rootFolder,"imb"+str(id),data,test_perc,train_perc,0,imbalance_percs,class_col_name,minority_class,min_minority_class_samples_to_keep,random_state,arff_attrs)
	if(noise_percs is not None):
		GenerateDatasetSplitsForWithNoise(rootFolder,"noise" + str(id),data,test_perc,train_perc,0,noise_percs,class_col_name,flip_fn,random_state,arff_attrs)

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
			config["modelbuildtimesecs"] = timeit.timeit(lambda: RunCmdWithoutConsoleWindow(cmd),number=1)
			
			# now for test set
			config["predictionoutputfile"] = test_output_file
			config["testset"] = testfiles[0]
			cmd = config_gen.GenerateWekaCommandline(config)
			config["modelevaltimesecs"] = timeit.timeit(lambda : RunCmdWithoutConsoleWindow(cmd),number=1)

			for k in config:
				params_info.append("{0}={1}".format(k,config[k]))
			u.WriteTextArrayToFile(params_output_file,params_info)
			config = config_gen.GetNextConfigAlongWithIdentifier()
		print("done dataset : " + dataset_dir)

def RunCmdWithoutConsoleWindow(cmd):
	CREATE_NO_WINDOW = 0x08000000
	#return sb.Popen(cmd, creationflags=CREATE_NO_WINDOW, stdout=sb.PIPE,shell=True)
	return sb.call(cmd, creationflags=CREATE_NO_WINDOW, shell=True)

def LoadCreditScreeningData(file,arff_attr_file=None):
	data = pd.read_csv(file)
	if(arff_attr_file is not None):
		arff_attrs = u.ReadLinesFromFile(arff_attr_file)
		return data,arff_attrs
	return data,None

def MaintainRatio(minority_class_num, majority_class_num, ratio_to_maintain, min_minority_to_keep):
	"""
	Makes the ratio of minorities / majorities = ratio
	To do this, first tries to remove the minority samples
	while keep a min number of them. Then removes samples 
	from majority to get the ratio
	"""
	minority_samples_to_keep = ratio_to_maintain * majority_class_num
	remove_majority=False
	majority_to_remove = 0
	if((min_minority_to_keep > minority_samples_to_keep) | (minority_samples_to_keep > minority_class_num)):
		minority_samples_to_keep = minority_class_num
		majority_to_keep = minority_samples_to_keep / ratio_to_maintain
		majority_to_remove = majority_class_num - majority_to_keep
		if(majority_to_keep <= 0):
		    raise Exception("cannot maintain the ratio") 
	minority_to_remove = minority_class_num - minority_samples_to_keep
	print("required : {0} current : {1} total_min : {2} total_maj : {3}".format(ratio_to_maintain,minority_samples_to_keep / (majority_class_num - majority_to_remove),minority_samples_to_keep,(majority_class_num - majority_to_remove)))
	return [int(minority_to_remove),int(majority_to_remove)]

def GenerateDatasetSplitsForClassImbalance(
	rootFolder,
	dataset_folder_prefix,
	dataset,
	test_ratio,
	train_ratio,
	validation_ratio,
	imbalance_percentages,
	class_col,
	minority_label,
	min_minority_to_keep,
	random_state,
	arff_attr_info=None,
	train_set=None,
	test_set=None):
	"""
	train_size_percentages is a list of intergers specifying the
	percent of train set to be taken while preparing the dataset

	test_ratio,train_ratio,validation_ratio : numbers in percentages
	"""
	dataset_root = u.PreparePath("{0}/i-{1}_t-{2}_T-{3}".format(rootFolder,dataset_folder_prefix,train_ratio,test_ratio))
	if((train_set is not None) & (test_set is not None)):
		train = train_set
		test = test_set
	else:
		train,test,validation = CreateTrainTestAndValidationPartitions(dataset,class_col,train_ratio/100,test_ratio/100,random_state,validation_ratio/100)
	test_output_file_csv = u.PreparePath("{0}/i-{1}.test.csv".format(dataset_root,dataset_folder_prefix))
	test.to_csv(test_output_file_csv,index=False)
	if(arff_attr_info is not None):
		test_output_file_arff = u.PreparePath("{0}/i-{1}.test.arff".format(dataset_root,dataset_folder_prefix))
		CreateArffFileFromCsv(arff_attr_info,test_output_file_arff,test_output_file_csv,True,True)

	# now creating the train set partitions
	for imbalance_perc in imbalance_percentages:
		folder_path = u.PreparePath("{0}/i-{1}_t-{2}_im-{3}".format(dataset_root,dataset_folder_prefix,train_ratio,imbalance_perc))
		csv_output_file = u.PreparePath("{0}/i-{1}_t-{2}_im-{3}.train.csv".format(folder_path,dataset_folder_prefix,train_ratio,imbalance_perc))
		imbalance_dataset = CreateImbalancedDataSet(train,class_col,minority_label,imbalance_perc/100,min_minority_to_keep,random_state)
		imbalance_dataset.to_csv(csv_output_file,index=False)
		print("done imb : " + str(imbalance_perc))
		if(arff_attr_info is not None):
			arff_output_file = u.PreparePath("{0}/i-{1}_t-{2}_im-{3}.train.arff".format(folder_path,dataset_folder_prefix,train_ratio,imbalance_perc))
			CreateArffFileFromCsv(arff_attr_info,arff_output_file,csv_output_file,True,True)
		
		# writing the parameters
		params_info = ["dataset_instance={0}".format(dataset_folder_prefix),
						"test_split={0}".format(test_ratio),
						"train_split={0}".format(train_ratio),
						"random_state={0}".format(random_state),
						"class_col={0}".format(class_col),
						"minority_label={0}".format(minority_label),
						"imbalance_perc={0}".format(imbalance_perc)]
		params_out_file = u.PreparePath("{0}/i-{1}_t-{2}_im-{3}.params.txt".format(folder_path,dataset_folder_prefix,train_ratio,imbalance_perc))
		u.WriteTextArrayToFile(params_out_file,params_info)

def GenerateDatasetSplitsForWithNoise(
	rootFolder,
	dataset_folder_prefix,
	dataset,
	test_ratio,
	train_ratio,
	validation_ratio,
	noise_percentages,
	class_col,
	flip_fn,
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
	for noise_perc in noise_percentages:
		folder_path = u.PreparePath("{0}/i-{1}_t-{2}_noise-{3}".format(dataset_root,dataset_folder_prefix,train_ratio,noise_perc))
		csv_output_file = u.PreparePath("{0}/i-{1}_t-{2}_noise-{3}.train.csv".format(folder_path,dataset_folder_prefix,train_ratio,noise_perc))
		
		noisy_dataset = CreateNoisyDataset(train,class_col,noise_perc/100,random_state,flip_fn)
		noisy_dataset.to_csv(csv_output_file,index=False)
		
		print("done noisy : " + str(noise_perc))
		if(arff_attr_info is not None):
			arff_output_file = u.PreparePath("{0}/i-{1}_t-{2}_noise-{3}.train.arff".format(folder_path,dataset_folder_prefix,train_ratio,noise_perc))
			CreateArffFileFromCsv(arff_attr_info,arff_output_file,csv_output_file,True,True)
		
		# writing the parameters
		params_info = ["dataset_instance={0}".format(dataset_folder_prefix),
						"test_split={0}".format(test_ratio),
						"train_split={0}".format(train_ratio),
						"random_state={0}".format(random_state),
						"class_col={0}".format(class_col),
						"noise_perc={0}".format(noise_perc)]
		params_out_file = u.PreparePath("{0}/i-{1}_t-{2}_noise-{3}.params.txt".format(folder_path,dataset_folder_prefix,train_ratio,noise_perc))
		u.WriteTextArrayToFile(params_out_file,params_info)

def CreateImbalancedDataSet(data,class_col,minority_label,ratio,min_minority_to_keep,seed):
	total_minority_labels = (data[class_col] == minority_label).sum()
	total_majority_labels = len(data) - total_minority_labels
	remove_min,remove_maj = MaintainRatio(total_minority_labels,total_majority_labels,ratio,min_minority_to_keep)
	new_minority_instances = data[data[class_col] == minority_label]
	new_majority_instances = data[data[class_col] != minority_label]
	
	random.seed(seed)
	new_minority_instances = new_minority_instances.sample(n = len(new_minority_instances) - remove_min)
	new_majority_instances = new_majority_instances.sample(n = len(new_majority_instances) - remove_maj)

	return pd.concat([new_majority_instances,new_minority_instances]).sample(frac=1)

def CreateNoisyDataset(data,class_col,flip_frac,rand,flip_fn):
	random.seed(rand)
	rows_to_flip_idx = random.sample(range(len(data)),int(flip_frac * len(data)))
	assert(len(rows_to_flip_idx) == len(set(rows_to_flip_idx)))
	values = data[class_col].values
	for idx in rows_to_flip_idx:
		values[idx] = flip_fn(values[idx])
	data[class_col] = values
	return data

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
			params = GetDictionary(u.ReadLinesFromFile(params_file_path))
			values = []
			for k in params_to_keep:
				if(k in params):
					values.append(str(params[k]))
				else:
					values.append(str(np.NaN))
			p,r,f=GetPrecisionRecallForWekaOutputFile(params["trainpredictionoutputfile"],positive_class)
			train_performance_values = ",".join(values)
			train_performance_values = "{0},1,{1},{2},{3}".format(",".join(values),str(p),str(r),str(f))
			p,r,f=GetPrecisionRecallForWekaOutputFile(params["testpredictionoutputfile"],positive_class)
			test_performance_values = ",".join(values)
			test_performance_values = "{0},0,{1},{2},{3}".format(",".join(values),str(p),str(r),str(f))
			evals.append(train_performance_values)
			evals.append(test_performance_values)
	u.WriteTextArrayToFile(u.PreparePath("{0}/{1}".format(datasets_root_folder,evaluation_output_filename)),evals)

def GetPrecisionRecallForWekaOutputFile(file,positive_class_label):
	results = pd.read_csv(file)
	actual = results['actual'].map(lambda x : 1 if(x.split(':')[1] == positive_class_label) else 0)
	predicted = results['predicted'].map(lambda x : 1 if(x.split(':')[1] == positive_class_label) else 0)
	p,r,f,s = precision_recall_fscore_support(actual,predicted,average='binary')
	return [p,r,f]

def ComputePrecisionRecallForPythonOutputFormat(file,positive_class_label):
	results = pd.read_csv(file)
	p,r,f,s = precision_recall_fscore_support(results['actual'],results['predicted'],average='binary')
	return [p,r,f]

def GetDictionary(lines_array):
	d = {}
	for line in lines_array:
		tokens = line.split("=")
		d[tokens[0]] = tokens[1]
	return d

def GetOneHotEncodingForDataFrame(dataframe,columns_to_encode):
	for col in columns_to_encode:
		enc = pd.get_dummies(dataframe[col],prefix=col)
		dataframe = dataframe.drop(col,axis=1)
		dataframe=dataframe.join(enc)
	return dataframe

def ConvertLabelsToZeroOne(data, positive_class_label):
	labels_raw = np.array(data)
	labels_raw[labels_raw == positive_class_label] = 1
	labels_raw[labels_raw != 1] = 0
	return labels_raw.astype(int)

if __name__=="__main__":

	root = r"C:\Users\shkhandu\OneDrive\Gatech\Courses\ML\DataSets"
	letter_recognition_dataset_root = root+"/LetterRecognition"
	credit_screening_dataset_root = root+"/CreditScreeningDataSet"
	noise_percs = [0,10,15,20,25,30,50,70]
	train_perc = 80
	test_perc = 20
	r = 1

	GenerateCreditScreeningDataSetSplits(credit_screening_dataset_root,r,train_perc,test_perc,r,noise_percs=noise_percs)
	GenerateVowelRecognitionDataSetSplits(letter_recognition_dataset_root,r,train_perc,test_perc,r,noise_percs=noise_percs)
	r=2
	GenerateCreditScreeningDataSetSplits(credit_screening_dataset_root,r,train_perc,test_perc,r,noise_percs=noise_percs)
	GenerateVowelRecognitionDataSetSplits(letter_recognition_dataset_root,r,train_perc,test_perc,r,noise_percs=noise_percs)