import pandas as pd
import SupervisedLearning as sl
import numpy as np
from sklearn.model_selection import ParameterGrid
import glob
import utils as u
import time
import timeit
import DecisionTree as dt
import os

def GetIdForConfig(config):
    return "prune-{0}_iter-{1}".format(config['prune'],config['iter'])

def GetWekaCommandLineForConfig(config,is_test):
    weka_commandline_template_train_with_pruning = "java -classpath \"{0}\" weka.classifiers.meta.AdaBoostM1 -I {5} -t \"{1}\" -S {2} -d \"{3}\" -no-cv -classifications \"weka.classifiers.evaluation.output.prediction.CSV -file {4}\" -W weka.classifiers.trees.J48 -- -R -N 3"
    weka_commandline_template_test = "java -classpath \"{0}\" weka.classifiers.meta.AdaBoostM1 -T \"{1}\" -l {2} -classifications \"weka.classifiers.evaluation.output.prediction.CSV -file {3}\""
    weka_commandline_template_train_without_pruning = "java -classpath \"{0}\" weka.classifiers.meta.AdaBoostM1 -I {5} -t \"{1}\" -S {2} -d \"{3}\" -no-cv -classifications \"weka.classifiers.evaluation.output.prediction.CSV -file {4}\" -W weka.classifiers.trees.J48 -- -U"
    
    if(is_test):
        return weka_commandline_template_test.format(config['wekajar'],config['testset'],config['modeloutputfile'],config['testpredictionoutputfile'])
    elif config['prune'] == True:
         return weka_commandline_template_train_with_pruning.format(config['wekajar'],config['trainset'],config['random_state'],config['modeloutputfile'],config['trainpredictionoutputfile'],config['iter'])
    else:
         return weka_commandline_template_train_without_pruning.format(config['wekajar'],config['trainset'],config['random_state'],config['modeloutputfile'],config['trainpredictionoutputfile'],config['iter'])

def RunAdaBoostWithDecisionTrees(datasets_root_folder,weka_jar_path,use_arff_files=True):
    """
    #weightThreshold parameter : http://weka.8497.n7.nabble.com/AdaBoost-Parameters-td11830.html    
    """
    file_extn = "arff" if use_arff_files else ".csv"
    testfiles = glob.glob("{0}/*.test.{1}".format(datasets_root_folder,file_extn))
    
    for dataset_dir in u.Get_Subdirectories(datasets_root_folder):
        trainfile = glob.glob("{0}/*.train.{1}".format(dataset_dir,file_extn))[0]
        paramfile = glob.glob("{0}/*.params.txt".format(dataset_dir,file_extn))[0]
        dt_root = u.PreparePath(dataset_dir+"/ada",is_file=False)
        config_gen = ParameterGrid({'prune':[True,False],'iter':[5,10,20,50]})
        for config in config_gen:
            id = GetIdForConfig(config)
            params_info = u.ReadLinesFromFile(paramfile)
            params_info_dict=sl.GetDictionary(params_info)
            run_output_dir = u.PreparePath("{0}/{1}".format(dt_root,id),is_file=False)
            params_output_file=u.PreparePath("{0}/{1}.params.txt".format(run_output_dir,id))
            model_output_file=u.PreparePath("{0}/{1}.model".format(run_output_dir,id))
            train_output_file=u.PreparePath("{0}/{1}.train.predictions.csv".format(run_output_dir,id))
            test_output_file=u.PreparePath("{0}/{1}.test.predictions.csv".format(run_output_dir,id))

            config['random_state'] = params_info_dict['random_state']
            config["wekajar"] = weka_jar_path
            config["trainset"] = trainfile
            config["class"]="last"
            config["trainpredictionoutputfile"]=train_output_file
            config["predictionoutputfile"] = config["trainpredictionoutputfile"]
            config["modeloutputfile"] = model_output_file
            config["testpredictionoutputfile"] = test_output_file

            # for every config there has to be a train prediction and test prediction
            cmd = GetWekaCommandLineForConfig(config,False)
            config["modelbuildtimesecs"] = timeit.timeit(lambda: sl.RunCmdWithoutConsoleWindow(cmd),number=1) / config['iter']
            
            # now for test set
            config["predictionoutputfile"] = test_output_file
            config["testset"] = testfiles[0]
            cmd = GetWekaCommandLineForConfig(config,True)
            config["modelevaltimesecs"] = timeit.timeit(lambda : sl.RunCmdWithoutConsoleWindow(cmd),number=1)

            config.pop('random_state',None) # since we already have that in params_info
            for k in config:
                params_info.append("{0}={1}".format(k,config[k]))
            u.WriteTextArrayToFile(params_output_file,params_info)
        print("done dataset : " + dataset_dir)

def RunAdaBoostOnVowelRecognitionDataset(root=r"C:\Users\shkhandu\OneDrive\Gatech\Courses\ML\DataSets\LetterRecognition",weka_jar_path="C:\Program Files\Weka-3-8\weka.jar"):
	pos_class="v"
	metric_fn = sl.GetPrecisionRecallForWekaOutputFile
	keys_to_keep=['dataset_instance','test_split','train_split','random_state','noise_perc','train_split_percent_used','imbalance_perc','prune','iter','modelbuildtimesecs']
	classifier_fn = lambda x : RunAdaBoostWithDecisionTrees(x,weka_jar_path)
	id="vowel.ada_1_all"
	algo_folder='ada'
	force_computation=True
	dt.RunNEvaluateExperimentsOnDataSet(classifier_fn,root,id,metric_fn,algo_folder,keys_to_keep,pos_class,[],force_computation)

def RunAdaBoostOnCreditScreeningDataset(root=r"C:\Users\shkhandu\OneDrive\Gatech\Courses\ML\DataSets\CreditScreeningDataset",weka_jar_path="C:\Program Files\Weka-3-8\weka.jar"):
	pos_class="+"
	metric_fn = sl.GetPrecisionRecallForWekaOutputFile
	keys_to_keep=['dataset_instance','test_split','train_split','random_state','noise_perc','train_split_percent_used','imbalance_perc','prune','iter','modelbuildtimesecs']
	classifier_fn = lambda x : RunAdaBoostWithDecisionTrees(x,weka_jar_path)
	id="credit.ada_1_all"
	algo_folder='ada'
	force_computation=True
	dt.RunNEvaluateExperimentsOnDataSet(classifier_fn,root,id,metric_fn,algo_folder,keys_to_keep,pos_class,[],force_computation)

def main():
    RunAdaBoostOnCreditScreeningDataset(r"C:\Users\shwet\OneDrive\Gatech\Courses\ML\DataSets\CreditScreeningDataset")

if __name__ == '__main__':
    main()