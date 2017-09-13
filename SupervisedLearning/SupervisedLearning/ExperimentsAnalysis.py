import utils as u
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import ParameterGrid
import SupervisedLearning as sl
import random
import ast

def TestPlotting():
    y1 = u.YSeries(np.arange(10) * 2, line_style='-',
                   points_marker='o', line_color='r', plot_legend_label='x^2')
    y2 = u.YSeries(np.arange(10), line_style='-',
                   points_marker='x', line_color='b', plot_legend_label='x')
    x = np.arange(10)
    fig, ax = u.SaveDataPlotWithLegends([y1, y2], x, r"c:/temp/testfig.png", dispose_fig=False,
                                        x_axis_name="x values", y1_axis_name="y values", title="x square")
    plt.show(fig)


def GetAggMetrics(data, gpby=['train_split_percent_used', 'prune', 'istrain'], dataset_type_col_idx=0,
                  col_funcs={'p': ['mean', 'std'], 'r': ['mean', 'std'], 'f': [
        'mean', 'std'], 'modelbuildtimesecs': ['mean', 'std']}):
    groupby_cols = gpby
    data_grouped = data.groupby(groupby_cols, as_index=False)
    data_agg = data_grouped.agg(col_funcs)
    data_agg.columns = ['_'.join(col).strip() if(
        col[1] != '') else col[0] for col in data_agg.columns.values]
    return data_agg.sort_values(gpby[dataset_type_col_idx])

def AdaBoostAnalysis(
    output_root,
    output_file_prefix,
    metrics_file
    ):
    data_all = pd.read_csv(metrics_file)
    dataset_types = ['train_split_percent_used',
                     'imbalance_perc', 'noise_perc']
    col_funcs = {'p': ['mean', 'std'], 'r': ['mean', 'std'], 'f': [
        'mean', 'std']}

    mapping_output_words = {
        'p': 'Precision',
        'r': 'Recall',
        'f': 'F-Measure',
        dataset_types[0]: 'Train size % used',
        dataset_types[1]: 'Fraction of postives to negatives',
        dataset_types[2]: 'Noise %',
        'modelbuildtimesecs': 'Time to build AdaBoost model (sec)'}

    for dataset_type in dataset_types:
        def filter_query(x): return (~np.isnan(x[dataset_type]))
        data = FilterRows(data_all, filter_query)
        data_agg = GetAggMetrics(data, col_funcs=col_funcs, gpby=[
            dataset_type, 'prune', 'istrain', 'iter'])
        for metric, v in col_funcs.items():
            for agg in v:
                iterations = np.sort(data_agg['iter'].unique())
                prune_vals = data_agg['prune'].unique()
                dataset_type_values = data_agg[dataset_type].unique()
                for type_val in dataset_type_values:
                    for prune_val in prune_vals:
                        metric_col = metric + "_" + agg
                        y_test = []
                        y_train = []
                        for i in iterations:
                            filtered_data = data_agg[(data_agg['prune'] == prune_val) & (
                                data_agg['iter'] == i) & (data_agg[dataset_type] == type_val)]
                            train_data = filtered_data[filtered_data['istrain'] == 1]
                            assert(len(train_data) == 1)
                            y_train.append(train_data[metric_col].iloc[0])

                            test_data = filtered_data[filtered_data['istrain'] == 0]
                            assert(len(test_data) == 1)
                            y_test.append(test_data[metric_col].iloc[0])
                        # now we can plot since we have test and train values for each iter
                        output_file_name = u.PreparePath(
                            "{4}/{0}.{1}.prune-{5}.{6}-{7}.{2}.{3}.png".format(output_file_prefix, dataset_type, metric, agg, output_root, prune_val,dataset_type,type_val))
                        y_train_series = u.YSeries(
                            y_train, line_color='r', plot_legend_label='train')
                        y_test_series = u.YSeries(
                            y_test, line_color='b', plot_legend_label='test')
                        if(~os.path.isfile(output_file_name)):
                            u.SaveDataPlotWithLegends([y_train_series, y_test_series], iterations, output_file_name, True,
                                                    "num of iterations", mapping_output_words[metric], "AdaBoost Performance ({0})".format(agg))
                        print(output_file_name)

def KnnAnalysis(
    output_root,
    output_file_prefix,
    metrics_file
    ):
    data_all = pd.read_csv(metrics_file)
    dataset_types = ['train_split_percent_used']
    col_funcs = {'p': ['mean'], 'r': ['mean'], 'f': [
        'mean'], 'modelevaltimesecs': ['mean']}

    mapping_output_words = {
        'p': 'Precision',
        'r': 'Recall',
        'f': 'F-Measure',
        dataset_types[0]: 'Train size % used',
        dataset_types[1]: 'Fraction of postives to negatives',
        dataset_types[2]: 'Noise %',
        'modelevaltimesecs': 'Time to run Knn model (sec)'}

    for dataset_type in dataset_types:

        def filter_query(x): return (
            ~np.isnan(x[dataset_type]) & (x['istrain'] == 0))

        def distance_weights_filter(x): return x['weights'] == 'distance'

        def uniform_weights_filter(x): return x['weights'] == 'uniform'

        data = FilterRows(data_all, filter_query)
        data_agg = GetAggMetrics(data, col_funcs=col_funcs, gpby=[
            dataset_type, 'weights', 'neighbors'])
        x = data_agg[dataset_type].unique()
        for k, v in col_funcs.items():
            for agg in v:
                data_for_distance_based_weighting = FilterRows(
                    data_agg, distance_weights_filter)
                nneighbors = [5, 10, 20, 50]
                marker_and_color_map = {5: ('g', 'o'), 10: (
                    'r', '+'), 20: ('b', 'x'), 50: ('k', 'd')}
                y_series = []
                for n in nneighbors:
                    d = data_for_distance_based_weighting[data_for_distance_based_weighting['neighbors'] == n]
                    y = u.YSeries(d[k + "_" + agg], line_color=marker_and_color_map[n][0],
                                  points_marker=marker_and_color_map[n][1], plot_legend_label="k = " + str(n))
                    y_series.append(y)
                output_file_name = u.PreparePath(
                    "{4}/{0}.{1}.weighted.{2}.{3}.png".format(output_file_prefix, dataset_type, k, agg, output_root))
                f, ax = u.SaveDataPlotWithLegends(y_series, x, output_file_name,
                                                  True, mapping_output_words[dataset_type], mapping_output_words[k], 'K Nearest Neighbor'.format(agg))

                data_for_distance_based_weighting = FilterRows(
                    data_agg, uniform_weights_filter)
                y_series = []
                for n in nneighbors:
                    d = data_for_distance_based_weighting[data_for_distance_based_weighting['neighbors'] == n]
                    y = u.YSeries(d[k + "_" + agg], line_color=marker_and_color_map[n][0],
                                  points_marker=marker_and_color_map[n][1], plot_legend_label="k = " + str(n))
                    y_series.append(y)
                output_file_name = u.PreparePath(
                    "{4}/{0}.{1}.uniform.{2}.{3}.png".format(output_file_prefix, dataset_type, k, agg, output_root))
                f, ax = u.SaveDataPlotWithLegends(y_series, x, output_file_name,
                                                  True, mapping_output_words[dataset_type], mapping_output_words[k], 'K Nearest Neighbor'.format(agg))
    return data_agg

def DecisionTreeAnalysis(
        output_root,
        output_file_prefix,
        metrics_file,
        dataset_filter_fn = None,
        plt_title = "Decision Trees Performance"):

    data_all = pd.read_csv(metrics_file)
    dataset_types = ['train_split_percent_used']
    col_funcs = {'p': ['mean'], 'r': ['mean'], 'f': [
        'mean'], 'modelbuildtimesecs': ['mean']}

    mapping_output_words = {
        'p': 'Precision',
        'r': 'Recall',
        'f': 'F-Measure',
        dataset_types[0]: 'Train size % used',
        'modelbuildtimesecs': 'Time to build model (sec)'}

    for dataset_type in dataset_types:

        def filter_query(x): return ~np.isnan(x[dataset_type])

        def train_prune_filter(x): return x['prune'] & (x['istrain'] == 1)

        def train_no_prune_filter(x): return (
            x['prune'] == False) & (x['istrain'] == 1)

        def test_prune_filter(x): return x['prune'] & (x['istrain'] == 0)

        def test_no_prune_filter(x): return (
            x['prune'] == False) & (x['istrain'] == 0)

        if(dataset_filter_fn is not None):
            data_all = FilterRows(data_all, dataset_filter_fn)
        data = FilterRows(data_all, filter_query)
        data_agg = GetAggMetrics(data, col_funcs=col_funcs, gpby=[
            dataset_type, 'prune', 'istrain'])
        x = data_agg[dataset_type].unique()

        for k, v in col_funcs.items():
            for agg in v:
                y_train_prune = u.YSeries(FilterRows(data_agg, train_prune_filter)[k + "_" + agg], line_color='r',
                                          points_marker='o', plot_legend_label="Train_with_pruning")
                y_train_no_prune = u.YSeries(FilterRows(data_agg, train_no_prune_filter)[k + "_" + agg], line_color='r',
                                             points_marker='x', plot_legend_label="Train_without_pruning")
                y_test_prune = u.YSeries(FilterRows(data_agg, test_prune_filter)[k + "_" + agg], line_color='b',
                                         points_marker='o', plot_legend_label="Test_with_pruning")
                y_no_test_prune = u.YSeries(FilterRows(data_agg, test_no_prune_filter)[k + "_" + agg], line_color='b',
                                            points_marker='x', plot_legend_label="Test_without_pruning")

                if((k=='modelbuildtimesecs')):
                    y_series = [y_train_no_prune, y_train_prune]
                else:
                    y_series = [y_test_prune, y_no_test_prune, y_train_no_prune, y_train_prune]

                output_file_name = u.PreparePath(
                    "{3}/{0}.{4}.{1}.{2}.png".format(output_file_prefix, k, agg, output_root, dataset_type))
                f, ax = u.SaveDataPlotWithLegends(y_series, x, output_file_name,
                                                  True, mapping_output_words[dataset_type], mapping_output_words[k], plt_title)
    return data_agg

def KnnAnalysisOptK(
        output_root,
        output_file_prefix,
        metrics_file,
        dataset_filter_fn = None):

    data_all = pd.read_csv(metrics_file)
    dataset_types = ['train_split_percent_used']
    col_funcs = {'p': ['mean'], 'r': ['mean'], 'f': [
        'mean'], 'modelevaltimesecs': ['mean']}

    mapping_output_words = {
        'p': 'Precision',
        'r': 'Recall',
        'f': 'F-Measure',
        dataset_types[0]: 'Train size % used',
        'modelevaltimesecs': 'Time to build model (sec)'}

    for dataset_type in dataset_types:

        def filter_query(x): return ~np.isnan(x[dataset_type])

        def train_filter(x): return (x['istrain'] == 1)

        def test_filter(x): return (x['istrain'] == 0)

        if(dataset_filter_fn is not None):
            data_all = FilterRows(data_all, dataset_filter_fn)
        data = FilterRows(data_all, filter_query)
        data_agg = GetAggMetrics(data, col_funcs=col_funcs, gpby=[
            dataset_type, 'istrain'])
        x = data_agg[dataset_type].unique()

        for k, v in col_funcs.items():
            for agg in v:
                y_train = u.YSeries(FilterRows(data_agg, train_filter)[k + "_" + agg], line_color='r',
                                          points_marker='o', plot_legend_label="Train")
                y_test = u.YSeries(FilterRows(data_agg, test_filter)[k + "_" + agg], line_color='b',
                                         points_marker='o', plot_legend_label='validation')
                if((k=='modelevaltimesecs')):
                    y_series = [y_train]
                else:
                    y_series = [y_test, y_train]

                output_file_name = u.PreparePath(
                    "{3}/{0}.{4}.{1}.{2}.png".format(output_file_prefix, k, agg, output_root, dataset_type))
                f, ax = u.SaveDataPlotWithLegends(y_series, x, output_file_name,
                                                  True, mapping_output_words[dataset_type], mapping_output_words[k], 'Knn Performance')
    return data_agg

def SvmAnalysis(
        output_root,
        output_file_prefix,
        metrics_file,
        dataset_filter_fn = None):

    def ComputeTotalSupportVectors(s):
        return np.array([int(t) for t in s.split(';')]).sum()

    data_all = pd.read_csv(metrics_file)
    data_all['numsupportvectors'] = data_all['numsupportvectors'].apply(ComputeTotalSupportVectors)
    dataset_types = ['train_split_percent_used']
    col_funcs = {'p': ['mean'], 'r': ['mean'], 'f': [
        'mean'], 'modelbuildtimesecs': ['mean']}

    mapping_output_words = {
        'p': 'Precision',
        'r': 'Recall',
        'f': 'F-Measure',
        dataset_types[0]: 'Train size % used',
        'modelbuildtimesecs': 'Time to build model (sec)',
        'numsupportvectors': 'Number of Support Vectors'}

    for dataset_type in dataset_types:

        def filter_query(x): return ~np.isnan(x[dataset_type])

        def train_filter(x): return (x['istrain'] == 1)

        def test_filter(x): return (x['istrain'] == 0)

        if(dataset_filter_fn is not None):
            data_all = FilterRows(data_all, dataset_filter_fn)
        data = FilterRows(data_all, filter_query)
        data_agg = GetAggMetrics(data, col_funcs=col_funcs, gpby=[
            dataset_type, 'istrain'])
        x = data_agg[dataset_type].unique()

        for k, v in col_funcs.items():
            for agg in v:
                y_train = u.YSeries(FilterRows(data_agg, train_filter)[k + "_" + agg], line_color='r',
                                          points_marker='o', plot_legend_label="Train")
                y_test = u.YSeries(FilterRows(data_agg, test_filter)[k + "_" + agg], line_color='b',
                                         points_marker='o', plot_legend_label='validation')
                if((k=='numsupportvectors') | (k=='modelbuildtimesecs')):
                    y_series = [y_train]
                else:
                    y_series = [y_test, y_train]

                output_file_name = u.PreparePath(
                    "{3}/{0}.{4}.{1}.{2}.png".format(output_file_prefix, k, agg, output_root, dataset_type))
                f, ax = u.SaveDataPlotWithLegends(y_series, x, output_file_name,
                                                  True, mapping_output_words[dataset_type], mapping_output_words[k], 'SVM Performance'.format(agg))
    return data_agg

def PlotSupportVectorsOverlap(root, output_file, data_file = None):
    file_template = root + '/i-0_t-80_T-20/i-0_t-80_ts-{0}/svm/cvresults/cvresults.model';

    y = []
    x = []
    for i in np.arange(30,110,10):
        file1 = file_template.format(str(i-10))
        file2 = file_template.format(str(i))
        s1 = u.ReadBinaryFile(file1).support_
        s2 = u.ReadBinaryFile(file2).support_
        _y = len(set(s1).intersection(s2)) / len(s1)
        _x = i
        y.append(_y)
        x.append(_x)
    outputfile = root + "/" + output_file
    u.SaveDataPlotWithLegends([u.YSeries(y)],x,outputfile,x_axis_name="Train size % used",y1_axis_name="Common support vectors fraction wrt previous size %",y_limits=[0,1])
    if(data_file is not None):
        pd.DataFrame({'size %':x, 'overlap':y}).to_csv(root + '/' + data_file,index = False)

def PlotCrossValidationCurvesForNNets(rootfolder):
    
    roots = [rootfolder + r'/CreditScreeningDataset',rootfolder + r'/LetterRecognition']
    for root in roots:
        instance = r'i-0_t-80_T-20'
        stopping = 'earlystop-False'
        dataset_instance_root = root + '/' + instance
        plot_output_file = u.PreparePath(root+ r'/Plots/nnets/cv.{0}.nnets.{1}.png'.format(stopping, instance))
        cv_save_file = u.PreparePath(dataset_instance_root + "/nnets.{0}.{1}.model_complexity_curves.csv".format(instance,stopping))
        x_axis_name = 'Train size % used'
        parameter_name = 'train_split_percent_used'
        y_axis_name = 'F-Measure'
        title = 'CV Peformance'
        def parameter_getter(path):
            paramfile = "{0}/nnets/{1}/{1}.params.txt".format(path,stopping)
            params_info = u.ReadLinesFromFile(paramfile)
            params_info_dict=sl.GetDictionary(params_info)
            return int(params_info_dict[parameter_name])

        def cv_getter(path):
            return "{0}/nnets/{1}/{1}.grid_search_cv_results.csv".format(path,stopping)

        PlotCrossValidationCurves(dataset_instance_root,plot_output_file,x_axis_name,y_axis_name,title,parameter_getter,cv_getter,cv_save_file)
        plot_fn = lambda x : (("0.0001" in x) | ("0.0001" in x)) & (("70" in x) | ("50" in x))
        plot_output_file = root+ r'/Plots/nnets/cv.small.{0}.nnets.{1}.png'.format(stopping, instance)
        PlotCrossValidationCurves(dataset_instance_root,plot_output_file,x_axis_name,y_axis_name,title,parameter_getter,cv_getter,should_plot=plot_fn)

def PlotCrossValidationCurvesForSvm(rootfolder):
    # root = r'C:/Users/shwet/OneDrive/Gatech/Courses/ML/DataSets/LetterRecognition'
    #root = r'C:/Users/shkhandu/OneDrive/Gatech/Courses/ML/DataSets/CreditScreeningDataset'
    roots = [rootfolder + r'/CreditScreeningDataset',rootfolder + r'/LetterRecognition']
    for root in roots:    
        instance = r'i-0_t-80_T-20'
        dataset_instance_root = root + "/" + instance
        plot_output_file = u.PreparePath(root+ r'/Plots/svm/cv.svm.{0}.png'.format(instance))
        cv_save_file = u.PreparePath(dataset_instance_root + "/svm.{0}.model_complexity_curves.csv".format(instance))
        x_axis_name = 'Train size % used'
        y_axis_name = 'F-Measure'
        title = 'CV Peformance'
        def parameter_getter(path):
            paramfile = "{0}/svm/cvresults/cvresults.params.txt".format(path)
            params_info = u.ReadLinesFromFile(paramfile)
            params_info_dict=sl.GetDictionary(params_info)
            return int(params_info_dict['train_split_percent_used'])

        def cv_getter(path):
            return "{0}/svm/cvresults/cvresults.grid_search_cv_results.csv".format(path)
        PlotCrossValidationCurves(dataset_instance_root,plot_output_file,x_axis_name,y_axis_name,title,parameter_getter,cv_getter,cv_save_file)
def PlotCrossValidationCurvesForWeka(
    cv_file,
    model_complexity_param_name,
    metric_name,
    plt_output_file,
    title,
    x_axis_name,
    y_axis_name,
    rows_filter_fn = None):
    data = pd.read_csv(cv_file)
    if(rows_filter_fn is not None):
        data = FilterRows(data,rows_filter_fn)
    metric_vals = data[[model_complexity_param_name,metric_name]].set_index(model_complexity_param_name).sort_index()
    x = metric_vals.index
    y = metric_vals[metric_name]
    y = u.YSeries(y)
    u.SaveDataPlotWithLegends([y],x,plt_output_file,True,x_axis_name,y_axis_name,title)

def PlotCrossValidationCurvesForKnn(rootfolder):
    # root = r'C:/Users/shwet/OneDrive/Gatech/Courses/ML/DataSets/LetterRecognition'
    #root = r'C:/Users/shkhandu/OneDrive/Gatech/Courses/ML/DataSets/CreditScreeningDataset'
    roots = [rootfolder + r'/CreditScreeningDataset',rootfolder + r'/LetterRecognition']
    for root in roots:    
        instance = r'i-0_t-80_T-20'
        dataset_instance_root = root + "/" + instance
        plot_output_file = u.PreparePath(root+ r'/Plots/knn/cv.knn.{0}.png'.format(instance))
        cv_save_file = u.PreparePath(dataset_instance_root + "/knn.{0}.model_complexity_curves.csv".format(instance))
        x_axis_name = 'Model complexity'
        y_axis_name = 'F-Measure'
        title = 'CV Peformance'
        def parameter_getter(path):
            paramfile = "{0}/knn/weights-uniform_neighbors--1/weights-uniform_neighbors--1.params.txt".format(path)
            params_info = u.ReadLinesFromFile(paramfile)
            params_info_dict=sl.GetDictionary(params_info)
            return int(params_info_dict['train_split_percent_used'])

        def knn_label_maker(l):
            p = ast.literal_eval(l)
            return "n{0}w{1}".format(p['n_neighbors'],p['weights'][0])
        def cv_getter(path):
            return "{0}/knn/weights-uniform_neighbors--1/weights-uniform_neighbors--1.grid_search_cv_results.csv".format(path)
        PlotCrossValidationCurves2(dataset_instance_root,plot_output_file,x_axis_name,y_axis_name,title,parameter_getter,cv_getter,cv_save_file,label_maker=knn_label_maker)    

def PlotCrossValidationCurves2(
    dataset_instance_root,
    plot_output_file,
    x_axis_name,
    y_axis_name,
    title,
    parameter_value_getter_fn,
    cv_results_file_getter_fn,
    cv_save_file=None,
    should_plot = lambda x : True,
    label_maker = lambda x : x):
    grid = ParameterGrid([{'marker':['o','x','d','^','+','v','8','s','p','>','<'], 'color':['orange','red','blue','green','black','saddlebrown','violet','darkcyan','maroon','lightcoral']}])
    combinations = [p for p in grid]
    random.seed(30)
    random.shuffle(combinations)
    param_dict = {}
    x_value_dict = {}
    for parameter_value_dataset in u.Get_Subdirectories(dataset_instance_root):
        cv_file_path = cv_results_file_getter_fn(parameter_value_dataset)
        if(os.path.isfile(cv_file_path) == False):
            continue
        cv_results = pd.read_csv(cv_file_path)
        parameter_value = parameter_value_getter_fn(parameter_value_dataset)
        for i in range(len(cv_results)):
            #param_dict = {param1 : series_1}
            param = cv_results.iloc[i]['params']
            s = pd.Series({parameter_value : cv_results.iloc[i]['mean_test_score']})
            if param in param_dict:
                param_dict[param] = param_dict[param].append(s)
            else:
                param_dict[param] = s
    yseries = []
    x = []
    for name,value in param_dict.items():
        if(should_plot(name) == False):
            continue
        theme = combinations.pop()
        y = u.YSeries(value.sort_index().values,points_marker=theme['marker'],line_color=theme['color'],plot_legend_label=name)
        yseries.append(y)
        x = value.sort_index().index
    transpose_data = pd.DataFrame(param_dict).transpose()
    x_values = transpose_data.index.values
    x_values = list(map(label_maker,x_values))
    col = transpose_data.columns[0]
    y_values = transpose_data[col]
    yseries = [u.YSeries(y_values)]
    u.SaveDataPlotWithLegends(yseries,x_values,plot_output_file,True,x_axis_name,y_axis_name)
    if(cv_save_file is not None):
        pd.DataFrame(param_dict).transpose().to_csv(cv_save_file)

def PlotCrossValidationCurves(
    dataset_instance_root,
    plot_output_file,
    x_axis_name,
    y_axis_name,
    title,
    parameter_value_getter_fn,
    cv_results_file_getter_fn,
    cv_save_file=None,
    should_plot = lambda x : True):
    grid = ParameterGrid([{'marker':['o','x','d','^','+','v','8','s','p','>','<'], 'color':['orange','red','blue','green','black','saddlebrown','violet','darkcyan','maroon','lightcoral']}])
    combinations = [p for p in grid]
    random.seed(30)
    random.shuffle(combinations)
    param_dict = {}
    x_value_dict = {}
    for parameter_value_dataset in u.Get_Subdirectories(dataset_instance_root):
        cv_file_path = cv_results_file_getter_fn(parameter_value_dataset)
        if(os.path.isfile(cv_file_path) == False):
            continue
        cv_results = pd.read_csv(cv_file_path)
        parameter_value = parameter_value_getter_fn(parameter_value_dataset)
        for i in range(len(cv_results)):
            #param_dict = {param1 : series_1}
            param = cv_results.iloc[i]['params']
            s = pd.Series({parameter_value : cv_results.iloc[i]['mean_test_score']})
            if param in param_dict:
                param_dict[param] = param_dict[param].append(s)
            else:
                param_dict[param] = s
    yseries = []
    x = []
    for name,value in param_dict.items():
        if(should_plot(name) == False):
            continue
        theme = combinations.pop()
        y = u.YSeries(value.sort_index().values,points_marker=theme['marker'],line_color=theme['color'],plot_legend_label=name)
        yseries.append(y)
        x = value.sort_index().index
    u.SaveDataPlotWithLegends(yseries,x,plot_output_file,True,x_axis_name,y_axis_name)
    if(cv_save_file is not None):
        pd.DataFrame(param_dict).transpose().to_csv(cv_save_file)

def GetBestResultsForSvms(metrics_file):
    dataset_types = ['train_split_percent_used',
                     'imbalance_perc', 'noise_perc']
    col_funcs = {'p': ['mean', 'std'], 'r': ['mean', 'std'], 'f': [
        'mean', 'std'], 'modelbuildtimesecs': ['mean', 'std']}
    dataset = {'train_split_percent_used' : 'Train size % used', 'imbalance_perc' : 'fraction of positives to negatives', 'noise_perc' : 'noise %'}
    allresults = None
    for parameter in dataset_types:
            def filter_fn(x) : return (x['istrain'] == 0)
            results = _GetBestResults(metrics_file,dataset[parameter],parameter,"svm", filter_fn,col_funcs)
            results = results.drop(parameter,axis=1)
            if allresults is None:
                allresults = results
            else:
                allresults = pd.concat([allresults, results], axis=0, ignore_index=True)
    return allresults

def GetBestResultsForDecisionTrees(metrics_file):
    dataset_types = ['train_split_percent_used',
                     'imbalance_perc', 'noise_perc']
    prune = [True,False]
    col_funcs = {'p': ['mean', 'std'], 'r': ['mean', 'std'], 'f': [
        'mean', 'std'], 'modelbuildtimesecs': ['mean', 'std']}
    dataset = {'train_split_percent_used' : 'Train size % used', 'imbalance_perc' : 'fraction of positives to negatives', 'noise_perc' : 'noise %'}
    allresults = None
    for es in prune:
        for parameter in dataset_types:
            def filter_fn(x) : return (x['prune'] == es) & (x['istrain'] == 0)
            results = _GetBestResults(metrics_file,dataset[parameter],parameter,"dts_prune-{0}".format(es),filter_fn,col_funcs)
            results = results.drop(parameter,axis=1)
            if allresults is None:
                allresults = results
            else:
                allresults = pd.concat([allresults, results], axis=0, ignore_index=True)
    return allresults

def GetBestResultsForAdaboost(metrics_file):
    dataset_types = ['train_split_percent_used',
                     'imbalance_perc', 'noise_perc']
    prune = [True,False]
    col_funcs = {'p': ['mean', 'std'], 'r': ['mean', 'std'], 'f': [
        'mean', 'std'], 'modelbuildtimesecs': ['mean', 'std']}
    dataset = {'train_split_percent_used' : 'Train size % used', 'imbalance_perc' : 'fraction of positives to negatives', 'noise_perc' : 'noise %'}
    allresults = None
    alldata = pd.read_csv(metrics_file)
    for es in prune:
        for parameter in dataset_types:
            """
            beacuse i haven't done cross validation, i have to choose that iteration for each
            data set that gives the maximum f1 score
            """
            def param_filter_fn(x) : return ~(np.isnan(x[parameter]))
            data = FilterRows(alldata,param_filter_fn)
            best_results = None
            filtered_data = data
            for ds_instance in data['dataset_instance'].unique():
                for p_instance in data[parameter].unique():
                    def filter_fn(x) : return (x['prune'] == es) & (x['istrain'] == 0) & (x['dataset_instance'] == ds_instance) & (x[parameter] == p_instance)
                    filtered_rows = FilterRows(data,filter_fn)
                    a = filtered_rows['f']
                    if(len(a) == 0):
                        print("ignoring : {0} {1}".format(ds_instance,p_instance))
                        continue
                    b = np.max(filtered_rows['f'])
                    indxs = np.isclose(a,b)
                    best_iters = filtered_rows[indxs]
                    best_iters = best_iters.iloc[0]['iter']
                    print("{0} {1} {2} {3}".format(ds_instance,p_instance,str(best_iters),es))
                    def data_filter_fn(x) : return (x['prune'] == es) & (x['istrain'] == 0) & (x['dataset_instance'] == ds_instance) & (x[parameter] == p_instance) & (x['iter'] == best_iters)
                    filtered_data = FilterRows(filtered_rows,data_filter_fn)
                    if best_results is None:
                        best_results = filtered_data
                    else:
                        best_results=pd.concat([best_results,filtered_data],axis = 0,ignore_index=True)
                    print(len(best_results))

            results = _GetBestResults(metrics_file,dataset[parameter],parameter,"adaboost_prune-{0}".format(es),lambda x : True, col_funcs,data=best_results)
            results = results.drop(parameter,axis=1)
            if allresults is None:
                allresults = results
            else:
                allresults = pd.concat([allresults, results], axis=0, ignore_index=True)
    return allresults

def GetBestResultsForNNets(metrics_file):
    dataset_types = ['train_split_percent_used',
                     'imbalance_perc', 'noise_perc']
    earlystopping = [True,False]
    col_funcs = {'p': ['mean', 'std'], 'r': ['mean', 'std'], 'f': [
        'mean', 'std'], 'modelbuildtimesecs': ['mean', 'std']}
    dataset = {'train_split_percent_used' : 'Train size % used', 'imbalance_perc' : 'fraction of positives to negatives', 'noise_perc' : 'noise %'}
    allresults = None
    for es in earlystopping:
        for parameter in dataset_types:
            def filter_fn(x) : return (x['earlystopping'] == es) & (x['istrain'] == 0)
            results = _GetBestResults(metrics_file,dataset[parameter],parameter,"nnets_earlyStopping-{0}".format(es),filter_fn,col_funcs)
            results = results.drop(parameter,axis=1)
            if allresults is None:
                allresults = results
            else:
                allresults = pd.concat([allresults, results], axis=0, ignore_index=True)
    return allresults

def GetBestResultsForKnn(metrics_file):
    dataset_types = ['train_split_percent_used',
                     'imbalance_perc', 'noise_perc']
    col_funcs = {'p': ['mean', 'std'], 'r': ['mean', 'std'], 'f': [
        'mean', 'std'], 'modelevaltimesecs': ['mean', 'std']}
    dataset = {'train_split_percent_used' : 'Train size % used', 'imbalance_perc' : 'fraction of positives to negatives', 'noise_perc' : 'noise %'}
    allresults = None
    for parameter in dataset_types:
            def filter_fn(x) : return (x['weights'] == 'distance') & (x['istrain'] == 0) & (x['neighbors'] == -1)
            results = _GetBestResults(metrics_file,dataset[parameter],parameter,"knn",filter_fn,col_funcs)
            results['modelbuildtimesecs_mean'] = results['modelevaltimesecs_mean']
            results['modelbuildtimesecs_std'] = results['modelevaltimesecs_std']
            results = results.drop([parameter,'modelevaltimesecs_mean','modelevaltimesecs_std'],axis=1)
            if allresults is None:
                allresults = results
            else:
                allresults = pd.concat([allresults, results], axis=0, ignore_index=True)
    return allresults

def GetBestResultsForVowelRecognitionDataset(root_folder, output_path_relvative_to_root="all_performance.csv"):
    ada_metrics_file = root_folder + r'/eval_agg.vowel.ada_1_all.csv'
    ada_results = GetBestResultsForAdaboost(ada_metrics_file)

    svm_metrics_file = root_folder + r'/eval_agg.vowel.svm_2_all.csv'
    svm_results = GetBestResultsForSvms(svm_metrics_file)

    dts_metrics_file = root_folder + r'/eval_agg.vowel.dt_1_all.csv'
    dts_results = GetBestResultsForDecisionTrees(dts_metrics_file)

    nnets_metrics_file = root_folder + r'/eval_agg.vowel.nnet_2_all.csv'
    nnets_results = GetBestResultsForNNets(nnets_metrics_file)

    knn_metrics_file = root_folder + r'/eval_agg.vowel.knn_3_all.csv'
    knn_results = GetBestResultsForKnn(knn_metrics_file)

    output = pd.concat([nnets_results, knn_results, dts_results, svm_results, ada_results], axis=0, ignore_index=True)
    output_file = u.PreparePath(root_folder+"/"+output_path_relvative_to_root)
    output.to_csv(output_file,index=False)

    for dataset in output['dataset'].unique():
        yseries = []
        markers = u.GetMarkerColorCombinations()
        metric_plots = {}
        x = []
        for algo in output['algorithm'].unique():
            filter_fn = lambda x : (x['algorithm'] == algo) & (x['dataset'] == dataset)
            filtered_data = FilterRows(output,filter_fn)
            data = filtered_data.set_index('parameter')
            data= data.drop(['algorithm','dataset'],axis=1)
            data = data.sort_index()
            x = data.index.values
            theme = markers.pop()
            for column in data.columns:
                y = u.YSeries(data[column].values,points_marker=theme['marker'],line_color=theme['color'],plot_legend_label = algo)
                if column in metric_plots:
                    metric_plots[column].append(y)
                else : 
                    metric_plots[column] = [y]
        token_map = {'f' : 'F-Measure','r' : 'Recall','p' : 'Precision','modelbuildtimesecs': 'Time to build model (sec)'}
        for metric,series in metric_plots.items():
            outputfilename = u.PreparePath(root_folder+"/Plots/all/{0}-{1}.png".format(dataset,metric))
            tokens = metric.split('_')
            y_axis_name = token_map[tokens[0]]
            x_axis_name = dataset
            title = dataset+" ({0})".format(tokens[1])
            u.SaveDataPlotWithLegends(series,x,outputfilename,x_axis_name = x_axis_name,y1_axis_name = y_axis_name,title=title)

def _GetBestResults(metrics_file,dataset,parameter,algorithm,rows_filter_fn,col_fns, data = None):
    """
    The final output schema should be : 
    dataset,parameter,algorithm,f_mean,p_mean,r_mean,time_mean
    TrainSize,20,earlystop-NNets,0.8,0.8,0.8,1.2
    """
    data = pd.read_csv(metrics_file) if data is None else data
    filtered_data = FilterRows(data,rows_filter_fn)
    gped_data = GetAggMetrics(filtered_data,gpby=[parameter],col_funcs = col_fns)
    gped_data['dataset'] = dataset
    gped_data['parameter'] = gped_data[parameter]
    gped_data['algorithm'] = algorithm
    return gped_data

def FilterRows(data, filter_fn):
    return data[data.apply(filter_fn, axis=1)]

def NNetAnalysis(
    output_root,
    output_file_prefix,
    metrics_file,
    iters_to_ignore):
    data_all = pd.read_csv(metrics_file)
    dataset_types = ['train_split_percent_used']
    col_funcs = {'p': ['mean'], 'r': ['mean'], 'f': [
        'mean'], 'modelbuildtimesecs': ['mean']}

    mapping_output_words = {
        'p': 'Precision',
        'r': 'Recall',
        'f': 'F-Measure',
        dataset_types[0]: 'Train size % used',
        'modelbuildtimesecs': 'Time to build model (sec)'}

    for dataset_type in dataset_types:

        def filter_query(x): return (~np.isnan(x[dataset_type]) & (x['total_iter'] > iters_to_ignore))

        def train_earlystopping_filter(x): return x['earlystopping'] & (x['istrain'] == 1)

        def train_no_earlystopping_filter(x): return (
            x['earlystopping'] == False) & (x['istrain'] == 1)

        def test_earlystopping_filter(x): return x['earlystopping'] & (x['istrain'] == 0)

        def test_no_earlystopping_filter(x): return (
            x['earlystopping'] == False) & (x['istrain'] == 0)

        data = FilterRows(data_all, filter_query)
        data_agg = GetAggMetrics(data, col_funcs=col_funcs, gpby=[
            dataset_type, 'earlystopping', 'istrain'])
        x = data_agg[dataset_type].unique()

        def MissingValuesHandler(curr_values_frame,keyCol,valueCol, required_values):
            data = dict(zip(curr_values_frame[keyCol],curr_values_frame[valueCol]))
            y = []
            for v in required_values:
                if(v in data):
                    y.append(data[v])
                else:
                    y.append(0)
            return y

        for k, v in col_funcs.items():
            for agg in v:
                mvh = lambda df : MissingValuesHandler(df,dataset_type,k+"_"+agg,x)
                y_train_earlystopping = u.YSeries(mvh(FilterRows(data_agg, train_earlystopping_filter)), line_color='r',
                                          points_marker='o', plot_legend_label="Train_with_earlystopping")
                y_train_no_earlystopping = u.YSeries(mvh(FilterRows(data_agg, train_no_earlystopping_filter)), line_color='r',
                                             points_marker='x', plot_legend_label="Train_without_earlystopping")
                y_test_earlystopping = u.YSeries(mvh(FilterRows(data_agg, test_earlystopping_filter)), line_color='b',
                                         points_marker='o', plot_legend_label="Test_with_earlystopping")
                y_no_test_earlystopping = u.YSeries(mvh(FilterRows(data_agg, test_no_earlystopping_filter)), line_color='b',
                                            points_marker='x', plot_legend_label="Test_without_earlystopping")

                output_file_name = u.PreparePath(
                    "{3}/{0}.{4}.{1}.{2}.png".format(output_file_prefix, k, agg, output_root, dataset_type))
                f, ax = u.SaveDataPlotWithLegends([y_test_earlystopping, y_no_test_earlystopping, y_train_no_earlystopping, y_train_earlystopping], x, output_file_name,
                                                  True, mapping_output_words[dataset_type], mapping_output_words[k], 'Neural Nets Performance'.format(agg))
    return data_agg

def main():
    root = r"C:/Users/shkhandu/OneDrive/Gatech/Courses/ML/DataSets"
    PlotSupportVectorsOverlap(root + "/LetterRecognition",r"Plots/svm/vowel.support_overlap.png","i-0_t-80_T-20/vowel.support_overlap.csv")
    PlotSupportVectorsOverlap(root + "/CreditScreeningDataset",r"Plots/svm/credit.support_overlap.png","i-0_t-80_T-20/credit.support_overlap.csv")
    #GetBestResultsForVowelRecognitionDataset(root+'/LetterRecognition')
    PlotCrossValidationCurvesForKnn(root)
    PlotCrossValidationCurvesForSvm(root)
    PlotCrossValidationCurvesForNNets(root)

    
    KnnAnalysisOptK(root + r'/CreditScreeningDataset/Plots/knn', r'dt.creditscreening',
                root + r"/CreditScreeningDataset/eval_agg.credit.knn_3_0.csv")
    KnnAnalysisOptK(root + r'/LetterRecognition/Plots/knn', r'dt.vowelrecognition',
                root + r"/LetterRecognition/eval_agg.vowel.knn_3_0.csv")

    # KnnAnalysis(root + r'/CreditScreeningDataset/Plots/knn', r'dt.creditscreening',
    #             root + r"/CreditScreeningDataset/eval_agg.credit.knn_1_all.csv")
    # KnnAnalysis(root + r'/LetterRecognition/Plots/knn', r'dt.vowelrecognition',
                # root + r"/LetterRecognition/eval_agg.vowel.knn_1_all.csv")

    #DecisionTreeAnalysis(
    #    root + r'/CreditScreeningDataset/Plots/dt', 
    #    r'dt.creditscreening',
    #    root + r"/CreditScreeningDataset/eval_agg.credit.dt_2_all.csv")
    #DecisionTreeAnalysis(
    #    root + r'/LetterRecognition/Plots/dt', 
    #    r'dt.vowelrecognition',
    #    root + r"/LetterRecognition/eval_agg.vowel.dt_1_all.csv")

#Neural net Analysis : We ignore results corresponding to some min num of iterations
# since for those the algorithm did not converge, mainly 8 or less iterations
    NNetAnalysis(
        root + r'/CreditScreeningDataset/Plots/nnets',
        'dt.creditscreening',
        root + r'/CreditScreeningDataset/eval_agg.credit.nnet_3_0.csv',
        0)

    NNetAnalysis(
        root + r'/LetterRecognition/Plots/nnets',
        'dt.vowelrecognition',
        root + r'/LetterRecognition/eval_agg.vowel.nnet_3_0.csv',
        0)

# Svm Analysis
    SvmAnalysis(
        root + r'/LetterRecognition/Plots/svm', 
        r'dt.vowelrecognition.svm',
        root + r"/LetterRecognition/eval_agg.vowel.svm_3_0.csv",
        None)

    SvmAnalysis(
        root + r'/CreditScreeningDataset/Plots/svm', 
        r'dt.creditscreening.svm',
        root + r"/CreditScreeningDataset/eval_agg.credit.svm_3_0.csv",
        None)

#Adaboost Analysis : First couple of functions generate plots showing how train/test error varies
# as iterations increase. Last 2 functions plot learning curves for a fixed number of iterations
    # AdaBoostAnalysis(
    #             root + r'/CreditScreeningDataset/Plots/ada/cv', 
    #             'dt.creditscreening',
    #             root + r"/CreditScreeningDataset/eval_agg.credit.ada_2_cv.csv")
    # AdaBoostAnalysis(
    #             root + r'/LetterRecognition/Plots/ada', 
    #             'dt.vowelrecognition',
    #             root + r"/LetterRecognition/eval_agg.vowel.ada_1_all.csv")

    # DecisionTreeAnalysis(
    #     root + r'/CreditScreeningDataset/Plots/ada_10_iters', 
    #     r'dt.creditscreening.ada_10_iters',
    #     root + r"/CreditScreeningDataset/eval_agg.credit.ada_1_all.csv",
    #     lambda x : x['iter'] == 10)

    # DecisionTreeAnalysis(
    #     root + r'/LetterRecognition/Plots/ada_50_iters', 
    #     r'dt.vowelrecognition.ada_50_iters',
    #     root + r"/LetterRecognition/eval_agg.vowel.ada_1_all.csv",
    #     lambda x : x['iter'] == 50)

if __name__ == '__main__':
    main()
