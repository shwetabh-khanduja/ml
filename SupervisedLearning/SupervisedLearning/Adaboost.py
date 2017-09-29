import glob
import os
import time
import timeit
import shutil

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
    weka_commandline_template_train_with_pruning = "java -classpath \"{0}\" weka.classifiers.meta.AdaBoostM1 -I {5} -t \"{1}\" -S {2} -d \"{3}\" -no-cv -classifications \"weka.classifiers.evaluation.output.prediction.CSV -file {4}\" -W weka.classifiers.trees.J48 -- -R -N 3 -M {6}"
    weka_commandline_template_train_with_pruning_cv = "java -classpath \"{0}\" weka.classifiers.meta.AdaBoostM1 -I {5} -t \"{1}\" -S {2} -d \"{3}\" -split-percentage 66 -classifications \"weka.classifiers.evaluation.output.prediction.CSV -file {4}\" -W weka.classifiers.trees.J48 -- -R -N 3 -M {6}"
    weka_commandline_template_test = "java -classpath \"{0}\" weka.classifiers.meta.AdaBoostM1 -T \"{1}\" -l {2} -classifications \"weka.classifiers.evaluation.output.prediction.CSV -file {3}\""
    weka_commandline_template_train_without_pruning = "java -classpath \"{0}\" weka.classifiers.meta.AdaBoostM1 -I {5} -t \"{1}\" -S {2} -d \"{3}\" -no-cv -classifications \"weka.classifiers.evaluation.output.prediction.CSV -file {4}\" -W weka.classifiers.trees.J48 -- -U -M {6}"
    weka_commandline_template_train_without_pruning_cv = "java -classpath \"{0}\" weka.classifiers.meta.AdaBoostM1 -I {5} -t \"{1}\" -S {2} -d \"{3}\" -split-percentage 66 -classifications \"weka.classifiers.evaluation.output.prediction.CSV -file {4}\" -W weka.classifiers.trees.J48 -- -U -M {6}"
    min_instances_per_leaf = config["inst"] if "inst" in config else 2
    if(is_test):
        return weka_commandline_template_test.format(config['wekajar'],config['testset'],config['modeloutputfile'],config['testpredictionoutputfile'])
    elif config['prune'] == True:
        if do_cv:
         return weka_commandline_template_train_with_pruning_cv.format(config['wekajar'],config['trainset'],config['random_state'],config['modeloutputfile'],config['trainpredictionoutputfile'],config['iter'], min_instances_per_leaf)
        else:
         return weka_commandline_template_train_with_pruning.format(config['wekajar'],config['trainset'],config['random_state'],config['modeloutputfile'],config['trainpredictionoutputfile'],config['iter'], min_instances_per_leaf)
    else:
        if do_cv:
            return weka_commandline_template_train_without_pruning_cv.format(config['wekajar'],config['trainset'],config['random_state'],config['modeloutputfile'],config['trainpredictionoutputfile'],config['iter'], min_instances_per_leaf)
        else:
            return weka_commandline_template_train_without_pruning.format(config['wekajar'],config['trainset'],config['random_state'],config['modeloutputfile'],config['trainpredictionoutputfile'],config['iter'], min_instances_per_leaf)

def RunAdaBoostWithDecisionTreesToGeneratePerIterationMetrics(datasets_root_folder,weka_jar_path,dataset_filter,iters,inst,use_arff_files=True):
    """
    #weightThreshold parameter : http://weka.8497.n7.nabble.com/AdaBoost-Parameters-td11830.html    
    """
    file_extn = "arff" if use_arff_files else ".csv"
    testfiles = glob.glob("{0}/*.test.{1}".format(datasets_root_folder,file_extn))
    first = True
    for dataset_dir in u.Get_Subdirectories(datasets_root_folder):
        if(dataset_filter not in dataset_dir):
            continue
        trainfile = glob.glob("{0}/*.train.{1}".format(dataset_dir,file_extn))[0]
        paramfile = glob.glob("{0}/*.params.txt".format(dataset_dir))[0]
        dt_root = u.PreparePath(dataset_dir+"/ada",is_file=False)
        config_gen = ParameterGrid({'prune':[True,False],'iter':iters})
        for config in config_gen:
            id = GetIdForConfig(config)
            config["inst"] = inst
            params_info = u.ReadLinesFromFile(paramfile)
            params_info_dict=sl.GetDictionary(params_info)
            run_output_dir = u.PreparePath("{0}/{1}".format(dt_root,id),is_file=False)
            params_output_file=u.PreparePath("{0}/{1}.params.txt".format(run_output_dir,id))
            model_output_file=u.PreparePath("{0}/{1}.model".format(run_output_dir,id))
            train_output_file=u.PreparePath("{0}/{1}.train.predictions.csv".format(run_output_dir,id))
            full_train_output_file=u.PreparePath("{0}/{1}.fulltrain.predictions.csv".format(run_output_dir,id))
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
            config["modelbuildtimesecs"] = timeit.timeit(lambda: sl.RunCmdWithoutConsoleWindow(cmd),number=1)

            config["testpredictionoutputfile"] = test_output_file
            config["testset"] = testfiles[0]
            cmd = GetWekaCommandLineForConfig(config,True)
            config["modelevaltimesecs"] = timeit.timeit(lambda : sl.RunCmdWithoutConsoleWindow(cmd),number=1)
            os.remove(model_output_file)

            config.pop('random_state',None) # since we already have that in params_info
            for k in config:
                params_info.append("{0}={1}".format(k,config[k]))
            u.WriteTextArrayToFile(params_output_file,params_info)
        print("done dataset : " + dataset_dir)

def RunAdaBoostWithDecisionTrees(datasets_root_folder,weka_jar_path,inst,use_arff_files=True):
    """
    #weightThreshold parameter : http://weka.8497.n7.nabble.com/AdaBoost-Parameters-td11830.html    
    """
    file_extn = "arff" if use_arff_files else ".csv"
    testfiles = glob.glob("{0}/*.test.{1}".format(datasets_root_folder,file_extn))
    first = True
    for dataset_dir in u.Get_Subdirectories(datasets_root_folder):
        if(first):
            assert("ts-100" in dataset_dir)
            first = False
        else:
            break
        trainfile = glob.glob("{0}/*.train.{1}".format(dataset_dir,file_extn))[0]
        paramfile = glob.glob("{0}/*.params.txt".format(dataset_dir))[0]
        dir_path = dataset_dir+"/ada"
        if(os.path.isdir(dir_path)):
            shutil.rmtree(dir_path)
        dt_root = u.PreparePath(dir_path,is_file=False)
        config_gen = ParameterGrid({'prune':[False],'iter':[2,5,10,20,25,30,50]})
        for config in config_gen:
            config["inst"] = inst
            id = GetIdForConfig(config)
            params_info = u.ReadLinesFromFile(paramfile)
            params_info_dict=sl.GetDictionary(params_info)
            run_output_dir = u.PreparePath("{0}/{1}".format(dt_root,id),is_file=False)
            params_output_file=u.PreparePath("{0}/{1}.params.txt".format(run_output_dir,id))
            model_output_file=u.PreparePath("{0}/{1}.model".format(run_output_dir,id))
            train_output_file=u.PreparePath("{0}/{1}.train.predictions.csv".format(run_output_dir,id))
            full_train_output_file=u.PreparePath("{0}/{1}.fulltrain.predictions.csv".format(run_output_dir,id))
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
            config["modelbuildtimesecs"] = timeit.timeit(lambda: sl.RunCmdWithoutConsoleWindow(cmd),number=1)

            # now for test set
            config["testpredictionoutputfile"] = full_train_output_file
            config["testset"] = trainfile
            cmd = GetWekaCommandLineForConfig(config,True)
            sl.RunCmdWithoutConsoleWindow(cmd)

            config["testpredictionoutputfile"] = test_output_file
            config["testset"] = testfiles[0]
            cmd = GetWekaCommandLineForConfig(config,True)
            config["modelevaltimesecs"] = timeit.timeit(lambda : sl.RunCmdWithoutConsoleWindow(cmd),number=1)
            os.remove(model_output_file)
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
        return 'train_split_percent_used',100

def RunAdaBoostWithOptimalItersAndDecisionTrees(datasets_root_folder,weka_jar_path, cv_results_file,inst ,use_arff_files=True):
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
            config["inst"] = inst
            filter = lambda x : (x['prune'] == False) & (x[filter_name] == filter_val) & (x['istrain'] == 1)
            filtered_rows = u.FilterRows(cv_results,filter)
            a = filtered_rows['m']
            if(len(a) == 0):
                print("ignoring : {0}".format(dataset_dir))
                continue
            b = np.max(filtered_rows['m'])
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
            config["modelbuildtimesecs"] = timeit.timeit(lambda: sl.RunCmdWithoutConsoleWindow(cmd),number=1)
            
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

def GetPerIterationMetricsForVowelRecognitionDataset(root, weka_jar_path, dataset_size_filter,iters):
    pos_class="v"
    metric_fn = sl.GetPrecisionRecallForWekaOutputFile
    keys_to_keep=['dataset_instance','test_split','train_split','random_state','train_split_percent_used','prune','iter','modelbuildtimesecs']
    algo_folder='ada'
    force_computation=True
    inst = 2
    cv_id="vowel.ada_3_{0}".format(dataset_size_filter)
    cv_eval_fn = lambda x : ("optiter" not in x) and (dataset_size_filter in x)
    classifier_fn = lambda x : RunAdaBoostWithDecisionTreesToGeneratePerIterationMetrics(x,weka_jar_path,dataset_size_filter,iters,inst)
    exp.RunNEvaluateExperimentsOnDataSet(classifier_fn,root,cv_id,metric_fn,algo_folder,keys_to_keep,pos_class,["i-0"],force_computation,cv_eval_fn)

def GetPerIterationMetricsForCreditScreeningDataset(root, weka_jar_path,dataset_size_filter,iters):
    pos_class="+"
    metric_fn = sl.GetPrecisionRecallForWekaOutputFile
    keys_to_keep=['dataset_instance','test_split','train_split','random_state','train_split_percent_used','prune','iter','modelbuildtimesecs']
    algo_folder='ada'
    force_computation=True
    inst = 2
    # doing cross validation
    cv_id="credit.ada_3_{0}".format(dataset_size_filter)
    cv_eval_fn = lambda x : ("optiter" not in x) and (dataset_size_filter in x)
    classifier_fn = lambda x : RunAdaBoostWithDecisionTreesToGeneratePerIterationMetrics(x,weka_jar_path,dataset_size_filter,iters,inst)
    exp.RunNEvaluateExperimentsOnDataSet(classifier_fn,root,cv_id,metric_fn,algo_folder,keys_to_keep,pos_class,["i-0"],force_computation,cv_eval_fn)

def RunAdaBoostOnVowelRecognitionDataset(root=r"C:\Users\shkhandu\OneDrive\Gatech\Courses\ML\DataSets\LetterRecognition",weka_jar_path="C:\Program Files\Weka-3-8\weka.jar"):
    pos_class="v"
    metric_fn = sl.GetPrecisionRecallForWekaOutputFile
    keys_to_keep=['dataset_instance','test_split','train_split','random_state','train_split_percent_used','prune','iter','modelbuildtimesecs']
    algo_folder='ada'
    force_computation=True
    inst = 2
    cv_id="vowel.ada_3_cv"
    cv_eval_fn = lambda x : ("optiter" not in x) and ("ts-100" in x)
    classifier_fn = lambda x : RunAdaBoostWithDecisionTrees(x,weka_jar_path,inst)
    exp.RunNEvaluateExperimentsOnDataSet(classifier_fn,root,cv_id,metric_fn,algo_folder,keys_to_keep,pos_class,["i-0"],force_computation,cv_eval_fn)

    opt_eval_fn = lambda x : "optiter" in x
    opt_id = "vowel.ada_3_0"
    file = 'eval.vowel.ada_3_cv.csv'
    classifier_fn = lambda x : RunAdaBoostWithOptimalItersAndDecisionTrees(x,weka_jar_path,file,inst)
    exp.RunNEvaluateExperimentsOnDataSet(classifier_fn,root,opt_id,metric_fn,algo_folder,keys_to_keep,pos_class,["i-0"],force_computation,opt_eval_fn)

def RunAdaBoostOnCreditScreeningDataset(root=r"C:\Users\shkhandu\OneDrive\Gatech\Courses\ML\DataSets\CreditScreeningDataset",weka_jar_path=r"C:\Program Files\Weka-3-8\weka.jar"):
    pos_class="+"
    metric_fn = sl.GetPrecisionRecallAccuracyForWekaOutputFile
    keys_to_keep=['dataset_instance','test_split','train_split','random_state','train_split_percent_used','prune','iter','modelbuildtimesecs']
    algo_folder='ada'
    force_computation=True
    inst = 2
    # doing cross validation
    cv_id="credit.ada_3_cv"
    cv_eval_fn = lambda x : ("optiter" not in x) and ("ts-100" in x)
    classifier_fn = lambda x : RunAdaBoostWithDecisionTrees(x,weka_jar_path,inst)
    exp.RunNEvaluateExperimentsOnDataSet(classifier_fn,root,cv_id,metric_fn,algo_folder,keys_to_keep,pos_class,["i-0"],force_computation,cv_eval_fn)

    # running the experiment with best params
    opt_eval_fn = lambda x : "optiter" in x
    opt_id = "credit.ada_3_0"
    file = 'eval.credit.ada_3_cv.csv'
    classifier_fn = lambda x : RunAdaBoostWithOptimalItersAndDecisionTrees(x,weka_jar_path,file,inst)
    exp.RunNEvaluateExperimentsOnDataSet(classifier_fn,root,opt_id,metric_fn,algo_folder,keys_to_keep,pos_class,["i-0"],force_computation,opt_eval_fn)

def main():
    #RunAdaBoostOnCreditScreeningDataset(r"C:\Users\shkhandu\OneDrive\Gatech\Courses\ML\DataSets\CreditScreeningDataset")
    iters = [2,5,10,15,20]
    ts = [20,30,40,50,60,70,80,90,100]
    for _ts in ts:
        GetPerIterationMetricsForCreditScreeningDataset(r"C:\Users\shkhandu\OneDrive\Gatech\Courses\ML\DataSets\CreditScreeningDataset",r"C:\Program Files\Weka-3-8\weka.jar","ts-{0}".format(_ts),iters)
    #RunAdaBoostOnVowelRecognitionDataset(r"C:\Users\shwet\OneDrive\Gatech\Courses\ML\DataSets\LetterRecognition")

if __name__ == '__main__':
    main()
