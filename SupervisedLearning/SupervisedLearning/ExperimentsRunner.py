import pandas as pd
import utils as u
import numpy as np
import os
import glob
import SupervisedLearning as sl

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
                            to prediction outputs and the positive class label. This function is
                            expected to return precision, recall and f-measure

    algo_folder : This is the name of the folder inside each train set folder, where the
                  algorithm will store its output

    params_to_keep : List of columns to keep in the final aggregated metrics file

    positive_class : Label of the positive class

    datasets_to_run_on : Is a list of substrings. Only on those datasets we run that contain
                         any of these filters as substrings in
    """
    agg_eval_file = "{0}/eval_agg.{1}.csv".format(dataset_root, id)
    eval_files = []
    for dataset in u.Get_Subdirectories(dataset_root):
        if('Plots' in dataset):
            continue
        ignore = True
        if(len(datasets_to_run_on) == 0):
            ignore = False
        else:
            for filter in datasets_to_run_on:
                if(filter in dataset):
                    ignore = False
                    break
        if(ignore):
            continue
        eval_file = "eval.{0}.csv".format(id)
        eval_file_full = dataset + "/" + eval_file
        eval_files.append(eval_file_full)
        if(os.path.isfile(eval_file_full)):
            if(force == False):
                continue
        # this root is for the various configs of the dataset
        classifer_fn(dataset)
        EvaluateExperiments(dataset, params_to_keep, positive_class,
                            metric_calculation_fn, eval_file, algo_folder)
    df = None
    for file in eval_files:
        d = pd.read_csv(file)
        df = pd.concat([df, d], axis=0,
                       ignore_index=True) if df is not None else d
    df.to_csv(agg_eval_file, index=False)

def EvaluateExperiments(
        datasets_root_folder,
        params_to_keep,
        positive_class,
        metric_calculation_fn,
        evaluation_output_filename="performance.csv",
        algo_folder="dt"):

    headers = []
    headers.extend(params_to_keep)
    headers.extend(['istrain', 'p', 'r', 'f'])
    headers = ",".join(headers)
    evals = []
    evals.append(headers)
    for directory in u.Get_Subdirectories(datasets_root_folder):
        # each directory is a dataset directory
        dt_output_dir = "{0}/{1}".format(directory, algo_folder)
        for run_output_folder in u.Get_Subdirectories(dt_output_dir):
            # read params file
            params_file_path = glob.glob(
                "{0}/*.params.txt".format(run_output_folder))[0]
            params = sl.GetDictionary(u.ReadLinesFromFile(params_file_path))
            values = []
            for k in params_to_keep:
                if(k in params):
                    values.append(str(params[k]))
                else:
                    values.append(str(np.NaN))
            p, r, f = metric_calculation_fn(
                params["trainpredictionoutputfile"], positive_class)
            train_performance_values = ",".join(values)
            train_performance_values = "{0},1,{1},{2},{3}".format(
                ",".join(values), str(p), str(r), str(f))
            p, r, f = metric_calculation_fn(
                params["testpredictionoutputfile"], positive_class)
            test_performance_values = ",".join(values)
            test_performance_values = "{0},0,{1},{2},{3}".format(
                ",".join(values), str(p), str(r), str(f))
            evals.append(train_performance_values)
            evals.append(test_performance_values)
    u.WriteTextArrayToFile(u.PreparePath(
        "{0}/{1}".format(datasets_root_folder, evaluation_output_filename)), evals)