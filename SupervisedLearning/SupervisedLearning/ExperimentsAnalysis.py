import utils as u
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def TestPlotting():
    y1 = u.YSeries(np.arange(10) * 2, line_style='-',
                   points_marker='o', line_color='r', plot_legend_label='x^2')
    y2 = u.YSeries(np.arange(10), line_style='-',
                   points_marker='x', line_color='b', plot_legend_label='x')
    x = np.arange(10)
    fig, ax = u.SaveDataPlotWithLegends([y1, y2], x, r"c:\temp\testfig.png", dispose_fig=False,
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
    dataset_types = ['train_split_percent_used',
                     'imbalance_perc', 'noise_perc']
    col_funcs = {'p': ['mean', 'std'], 'r': ['mean', 'std'], 'f': [
        'mean', 'std'], 'modelevaltimesecs': ['mean', 'std']}

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
                                                  True, mapping_output_words[dataset_type], mapping_output_words[k], 'K Nearest Neighbor ({0})'.format(agg))

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
                                                  True, mapping_output_words[dataset_type], mapping_output_words[k], 'K Nearest Neighbor ({0})'.format(agg))
    return data_agg


def DecisionTreeAnalysis(
        output_root,
        output_file_prefix,
        metrics_file,
        dataset_filter_fn = None):

    data_all = pd.read_csv(metrics_file)
    dataset_types = ['train_split_percent_used',
                     'imbalance_perc', 'noise_perc']
    col_funcs = {'p': ['mean', 'std'], 'r': ['mean', 'std'], 'f': [
        'mean', 'std'], 'modelbuildtimesecs': ['mean', 'std']}

    mapping_output_words = {
        'p': 'Precision',
        'r': 'Recall',
        'f': 'F-Measure',
        dataset_types[0]: 'Train size % used',
        dataset_types[1]: 'Fraction of postives to negatives',
        dataset_types[2]: 'Noise %',
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

                output_file_name = u.PreparePath(
                    "{3}/{0}.{4}.{1}.{2}.png".format(output_file_prefix, k, agg, output_root, dataset_type))
                f, ax = u.SaveDataPlotWithLegends([y_test_prune, y_no_test_prune, y_train_no_prune, y_train_prune], x, output_file_name,
                                                  True, mapping_output_words[dataset_type], mapping_output_words[k], 'Decision Trees Performance ({0})'.format(agg))
    return data_agg


def FilterRows(data, filter_fn):
    return data[data.apply(filter_fn, axis=1)]

def NNetAnalysis(
    output_root,
    output_file_prefix,
    metrics_file,
    iters_to_ignore):
    data_all = pd.read_csv(metrics_file)
    dataset_types = ['train_split_percent_used',
                     'imbalance_perc', 'noise_perc']
    col_funcs = {'p': ['mean', 'std'], 'r': ['mean', 'std'], 'f': [
        'mean', 'std'], 'modelbuildtimesecs': ['mean', 'std']}

    mapping_output_words = {
        'p': 'Precision',
        'r': 'Recall',
        'f': 'F-Measure',
        dataset_types[0]: 'Train size % used',
        dataset_types[1]: 'Fraction of postives to negatives',
        dataset_types[2]: 'Noise %',
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
                                                  True, mapping_output_words[dataset_type], mapping_output_words[k], 'Neural Nets Performance ({0})'.format(agg))
    return data_agg

def main():
    root = r"C:\Users\shwet\OneDrive\Gatech\Courses\ML\DataSets"
    
    # KnnAnalysis(root + r'\CreditScreeningDataset\Plots\knn', r'dt.creditscreening',
    #             root + r"\CreditScreeningDataset\eval_agg.credit.knn_1_all.csv")
    # KnnAnalysis(root + r'\LetterRecognition\Plots\knn', r'dt.vowelrecognition',
                # root + r"\LetterRecognition\eval_agg.vowel.knn_1_all.csv")

    # DecisionTreeAnalysis(root + r'\CreditScreeningDataset\Plots\dt', r'dt.creditscreening',root + r"\CreditScreeningDataset\eval_agg.credit.dt_2_all.csv")
    # DecisionTreeAnalysis(root + r'\LetterRecognition\Plots\dt', r'dt.vowelrecognition',root + r"\LetterRecognition\eval_agg.vowel.dt_1_all.csv")

    #Neural net Analysis : We ignore results corresponding to some min num of iterations
    # since for those the algorithm did not converge, mainly 8 or less iterations
    NNetAnalysis(
        root + r'\CreditScreeningDataset\Plots\nnets',
        'dt.creditscreening',
        root + r'\CreditScreeningDataset\eval_agg.credit.nnet_1_all.csv',
        8)

    NNetAnalysis(
        root + r'\LetterRecognition\Plots\nnets',
        'dt.vowelrecognition',
        root + r'\LetterRecognition\eval_agg.vowel.nnet_1_all.csv',
        8)


#Adaboost Analysis : First couple of functions generate plots showing how train/test error varies
# as iterations increase. Last 2 functions plot learning curves for a fixed number of iterations
    # AdaBoostAnalysis(
    #             root + r'\CreditScreeningDataset\Plots\ada', 
    #             'dt.creditscreening',
    #             root + r"\CreditScreeningDataset\eval_agg.credit.ada_1_all.csv")
    # AdaBoostAnalysis(
    #             root + r'\LetterRecognition\Plots\ada', 
    #             'dt.vowelrecognition',
    #             root + r"\LetterRecognition\eval_agg.vowel.ada_1_all.csv")

    # DecisionTreeAnalysis(
    #     root + r'\CreditScreeningDataset\Plots\ada_10_iters', 
    #     r'dt.creditscreening.ada_10_iters',
    #     root + r"\CreditScreeningDataset\eval_agg.credit.ada_1_all.csv",
    #     lambda x : x['iter'] == 10)

    # DecisionTreeAnalysis(
    #     root + r'\LetterRecognition\Plots\ada_50_iters', 
    #     r'dt.vowelrecognition.ada_50_iters',
    #     root + r"\LetterRecognition\eval_agg.vowel.ada_1_all.csv",
    #     lambda x : x['iter'] == 50)

if __name__ == '__main__':
    main()
