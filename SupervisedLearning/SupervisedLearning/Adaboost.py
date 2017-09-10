import glob
import os
import time
import timeit

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid

import ExperimentsRunner as exp
import SupervisedLearning as sl
import utils as u
import ntpath

def GetIdForConfig(config):
    return "prune-{0}_iter-{1}".format(config['prune'],config['iter'])

def GetIdForOptConfig(config):
    return "prune-{0}_optiter-{1}".format(config['prune'],config['iter'])

def GetWekaCommandLineForConfig(config,is_test, do_cv = True):
    weka_commandline_template_train_with_pruning = "java -classpath \"{0}\" weka.classifiers.meta.AdaBoostM1 -I {5} -t \"{1}\" -S {2} -d \"{3}\" -no-cv -classifications \"weka.classifiers.evaluation.output.prediction.CSV -file {4}\" -W weka.classifiers.trees.J48 -- -R -N 3"
    weka_commandline_template_train_with_pruning_cv = "java -classpath \"{0}\" weka.classifiers.meta.AdaBoostM1 -I {5} -t \"{1}\" -S {2} -d \"{3}\" -split-percentage 66 -classifications \"weka.classifiers.evaluation.output.prediction.CSV -file {4}\" -W weka.classifiers.trees.J48 -- -R -N 3"
    weka_commandline_template_test = "java -classpath \"{0}\" weka.classifiers.meta.AdaBoostM1 -T \"{1}\" -l {2} -classifications \"weka.classifiers.evaluation.output.prediction.CSV -file {3}\""
    weka_commandline_template_train_without_pruning = "java -classpath \"{0}\" weka.classifiers.meta.AdaBoostM1 -I {5} -t \"{1}\" -S {2} -d \"{3}\" -no-cv -classifications \"weka.classifiers.evaluation.output.prediction.CSV -file {4}\" -W weka.classifiers.trees.J48 -- -U"
    weka_commandline_template_train_without_pruning_cv = "java -classpath \"{0}\" weka.classifiers.meta.AdaBoostM1 -I {5} -t \"{1}\" -S {2} -d \"{3}\" -split-percentage 66 -classifications \"weka.classifiers.evaluation.output.prediction.CSV -file {4}\" -W weka.classifiers.trees.J48 -- -U"

    if(is_test):
        return weka_commandline_template_test.format(config['wekajar'],config['testset'],config['modeloutputfile'],config['testpredictionoutputfile'])
    elif config['prune'] == True:
        if do_cv:
         return weka_commandline_template_train_with_pruning_cv.format(config['wekajar'],config['trainset'],config['random_state'],config['modeloutputfile'],config['trainpredictionoutputfile'],config['iter'])
        else:
         return weka_commandline_template_train_with_pruning.format(config['wekajar'],config['trainset'],config['random_state'],config['modeloutputfile'],config['trainpredictionoutputfile'],config['iter'])
    else:
        if do_cv:
            return weka_commandline_template_train_without_pruning_cv.format(config['wekajar'],config['trainset'],config['random_state'],config['modeloutputfile'],config['trainpredictionoutputfile'],config['iter'])
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
        paramfile = glob.glob("{0}/*.params.txt".format(dataset_dir))[0]
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
            # if(os.path.isfile(train_output_file)):
            #     continue
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

def GetFilterOptions(dataset_name):
    tokens = ntpath.basename(dataset_name).split('_')[2].split('-')
    if(tokens[0] == 'im'):
        return 'imbalance_perc',int(tokens[1])
    elif(tokens[0] == 'noise'):
        return 'noise_perc',int(tokens[1])
    else:
        return 'train_split_percent_used',int(tokens[1])

def RunAdaBoostWithOptimalItersAndDecisionTrees(datasets_root_folder,weka_jar_path, cv_results_file ,use_arff_files=True):
    """
    #weightThreshold parameter : http://weka.8497.n7.nabble.com/AdaBoost-Parameters-td11830.html    
    """
    file_extn = "arff" if use_arff_files else ".csv"
    testfiles = glob.glob("{0}/*.test.{1}".format(datasets_root_folder,file_extn))
    cv_results = pd.read_csv( datasets_root_folder+"/"+ cv_results_file)
    for dataset_dir in u.Get_Subdirectories(datasets_root_folder):
        trainfile = glob.glob("{0}/*.train.{1}".format(dataset_dir,file_extn))[0]
        paramfile = glob.glob("{0}/*.params.txt".format(dataset_dir))[0]
        dt_root = u.PreparePath(dataset_dir+"/ada",is_file=False)
        filter_name,filter_val = GetFilterOptions(dataset_dir)
        config_gen = ParameterGrid({'prune':[True,False]})
        for config in config_gen:

            filter = lambda x : (x['prune'] == config['prune']) & (x[filter_name] == filter_val) & (x['istrain'] == 1)
            filtered_rows = u.FilterRows(cv_results,filter)
            a = filtered_rows['f']
            if(len(a) == 0):
                print("ignoring : {0}".format(dataset_dir))
                continue
            b = np.max(filtered_rows['f'])
            indxs = np.isclose(a,b)
            best_iters = filtered_rows[indxs]
            best_iters = best_iters.iloc[0]['iter']
            config['iter'] = best_iters

            id = GetIdForOptConfig(config)
            params_info = u.ReadLinesFromFile(paramfile)
            params_info_dict=sl.GetDictionary(params_info)
            run_output_dir = u.PreparePath("{0}/{1}".format(dt_root,id),is_file=False)
            params_output_file=u.PreparePath("{0}/{1}.params.txt".format(run_output_dir,id))
            model_output_file=u.PreparePath("{0}/{1}.model".format(run_output_dir,id))
            train_output_file=u.PreparePath("{0}/{1}.train.predictions.csv".format(run_output_dir,id))
            test_output_file=u.PreparePath("{0}/{1}.test.predictions.csv".format(run_output_dir,id))
            # if(os.path.isfile(train_output_file)):
            #     continue
            config['random_state'] = params_info_dict['random_state']
            config["wekajar"] = weka_jar_path
            config["trainset"] = trainfile
            config["class"]="last"
            config["trainpredictionoutputfile"]=train_output_file
            config["predictionoutputfile"] = config["trainpredictionoutputfile"]
            config["modeloutputfile"] = model_output_file
            config["testpredictionoutputfile"] = test_output_file

            # for every config there has to be a train prediction and test prediction
            cmd = GetWekaCommandLineForConfig(config,False,False)
            config["modelbuildtimesecs"] = timeit.timeit(lambda: sl.RunCmdWithoutConsoleWindow(cmd),number=1) / config['iter']
            
            # now for test set
            config["predictionoutputfile"] = test_output_file
            config["testset"] = testfiles[0]
            cmd = GetWekaCommandLineForConfig(config,True,False)
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
    algo_folder='ada'
    force_computation=True

    cv_id="vowel.ada_2_cv"
    cv_eval_fn = lambda x : "optiter" not in x
    classifier_fn = lambda x : RunAdaBoostWithDecisionTrees(x,weka_jar_path)
    exp.RunNEvaluateExperimentsOnDataSet(classifier_fn,root,cv_id,metric_fn,algo_folder,keys_to_keep,pos_class,[],force_computation,cv_eval_fn)

    opt_eval_fn = lambda x : "optiter" in x
    opt_id = "vowel.ada_2_all"
    file = 'eval.vowel.ada_2_cv.csv'
    classifier_fn = lambda x : RunAdaBoostWithOptimalItersAndDecisionTrees(x,weka_jar_path,file)
    exp.RunNEvaluateExperimentsOnDataSet(classifier_fn,root,opt_id,metric_fn,algo_folder,keys_to_keep,pos_class,[],force_computation,opt_eval_fn)

def RunAdaBoostOnCreditScreeningDataset(root=r"C:\Users\shkhandu\OneDrive\Gatech\Courses\ML\DataSets\CreditScreeningDataset",weka_jar_path="C:\Program Files\Weka-3-8\weka.jar"):
    pos_class="+"
    metric_fn = sl.GetPrecisionRecallForWekaOutputFile
    keys_to_keep=['dataset_instance','test_split','train_split','random_state','noise_perc','train_split_percent_used','imbalance_perc','prune','iter','modelbuildtimesecs']
    algo_folder='ada'
    force_computation=True

    # doing cross validation
    cv_id="credit.ada_2_cv"
    cv_eval_fn = lambda x : "optiter" not in x
    classifier_fn = lambda x : RunAdaBoostWithDecisionTrees(x,weka_jar_path)
    exp.RunNEvaluateExperimentsOnDataSet(classifier_fn,root,cv_id,metric_fn,algo_folder,keys_to_keep,pos_class,[],force_computation,cv_eval_fn)

    # running the experiment with best params
    opt_eval_fn = lambda x : "optiter" in x
    opt_id = "credit.ada_2_all"
    file = 'eval.credit.ada_2_cv.csv'
    classifier_fn = lambda x : RunAdaBoostWithOptimalItersAndDecisionTrees(x,weka_jar_path,file)
    exp.RunNEvaluateExperimentsOnDataSet(classifier_fn,root,opt_id,metric_fn,algo_folder,keys_to_keep,pos_class,[],force_computation,opt_eval_fn)

def main():
    RunAdaBoostOnCreditScreeningDataset(r"C:\Users\shwet\OneDrive\Gatech\Courses\ML\DataSets\CreditScreeningDataset")
    RunAdaBoostOnVowelRecognitionDataset(r"C:\Users\shwet\OneDrive\Gatech\Courses\ML\DataSets\LetterRecognition")

if __name__ == '__main__':
    main()
