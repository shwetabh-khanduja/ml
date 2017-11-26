import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
import utils as u
import SupervisedLearning as sl
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from numpy.linalg import norm
from scipy.stats import kurtosis as kt
from sklearn.random_projection import GaussianRandomProjection
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif,SelectKBest,mutual_info_regression
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.neural_network import MLPClassifier
import time
from sklearn.model_selection import train_test_split
from sklearn import cluster
from scipy.spatial import distance
from sklearn.feature_selection import chi2

def GetOneHotEncodingCols():
    return ['A1','A4','A5','A6','A7','A9','A10','A12','A13']

def ReadVowelRecognitionData(rootfolder):
    file = rootfolder+"/LetterRecognition/vowel-recongnition-dataset.csv"
    data = pd.read_csv(file)
    return data

def ReadLetterRecognitionData(rootfolder):
    file = rootfolder+"/LetterRecognition/letter-recognition.data.csv"
    fulldata = pd.read_csv(file)
    label_col_name = "lettr"
    ReplaceLabelsWithIntegers(fulldata,label_col_name)
    DoStandardScaling(fulldata,lambda x : x != label_col_name)
    X = fulldata.loc[:,fulldata.columns != label_col_name].as_matrix()
    Y = fulldata.loc[:,fulldata.columns == label_col_name]
    Y = np.array(Y[Y.columns[0]].values,dtype=int)
    X,X_test,Y,Y_test = train_test_split(X,Y,stratify=Y,train_size=0.80,test_size=0.20,random_state=0)
    return X_test,Y_test

def ReadCreditScreeningData(rootfolder):
    file = rootfolder + "/CreditScreeningDataset/data_no_missing_values.csv"
    lrd = pd.read_csv(file)
    lrd = PreProcessCreditScreeningData(lrd)
    label_col_name = "A16"
    X = lrd.loc[:,lrd.columns != label_col_name].as_matrix()
    Y = lrd.loc[:,lrd.columns == label_col_name]
    Y = np.array(Y[Y.columns[0]].values,dtype=int)
    return X,Y

def ReadCreditScreeningDataRaw(rootfolder):
    file = rootfolder + "/CreditScreeningDataset/data_no_missing_values.csv"
    lrd = pd.read_csv(file)
    return lrd
    lrd = PreProcessCreditScreeningData(lrd)
    label_col_name = "A16"
    X = lrd.loc[:,lrd.columns != label_col_name].as_matrix()
    Y = lrd.loc[:,lrd.columns == label_col_name]
    Y = np.array(Y[Y.columns[0]].values,dtype=int)
    return X,Y

def ReadGeneData(rootfolder):
    data = pd.read_csv(rootfolder + "/genedataset/data.csv",index_col=0)
    labels = pd.read_csv(rootfolder + "/genedataset/labels.csv",index_col=0)
    fulldata = pd.concat([data,labels],axis=1)
    label_col_name = "Class"
    ReplaceLabelsWithIntegers(fulldata,label_col_name)
    DoStandardScaling(fulldata,lambda x : x != label_col_name)
    X = fulldata.loc[:,fulldata.columns != label_col_name].as_matrix()
    Y = fulldata.loc[:,fulldata.columns == label_col_name]
    Y = np.array(Y[Y.columns[0]].values,dtype=int)
    return X,Y

def ReadVoiceData(rootfolder):
    datafile = rootfolder+r'/VoiceDataset/LSVT_voice_rehabilitation.csv'
    fulldata = pd.read_csv(datafile)
    label_col_name = "label"
    ReplaceLabelsWithIntegers(fulldata,label_col_name)
    DoStandardScaling(fulldata,lambda x : x != label_col_name)
    X = fulldata.loc[:,fulldata.columns != label_col_name].as_matrix()
    Y = fulldata.loc[:,fulldata.columns == label_col_name]
    Y = np.array(Y[Y.columns[0]].values,dtype=int)
    return X,Y

def Compute_ARI(actual, predicted):
    return metrics.adjusted_rand_score(actual,predicted)

def ComputeClusteringMetrics(actual,predicted, X):
    metrics_dict = {}
    metrics_dict['ari'] = metrics.adjusted_rand_score(actual,predicted)
    metrics_dict['ami'] = metrics.adjusted_mutual_info_score(actual,predicted)
    metrics_dict['nmi'] = metrics.normalized_mutual_info_score(actual,predicted)
    metrics_dict['sl'] = metrics.silhouette_score(X,predicted)
    metrics_dict["h"] = metrics.homogeneity_score(actual,predicted)
    metrics_dict["c"] = metrics.completeness_score(actual,predicted)
    metrics_dict["v"] = metrics.homogeneity_completeness_v_measure(actual,predicted)
    return metrics_dict

def RunKMeansOnCreditScreeningDataset(rootfolder):
    data = ReadCreditScreeningData(rootfolder)
    data = PreProcessCreditScreeningData(data)
    X = data.loc[:,data.columns != "A16"]
    Y = data.loc[:,data.columns == "A16"]
    resuls = RunClustering(X,Y,[2,5,8,9,10,20],10,["kmeans","gmm"])
    print("done credit screening")

def PreProcessCreditScreeningData(data, normalize=True, label_col = "A16"):
    cols_not_to_scale = set(GetOneHotEncodingCols())
    cols_not_to_scale.add(label_col)
    if(normalize):
        DoStandardScaling(data,lambda x : x not in cols_not_to_scale)
    ReplaceLabelsWithIntegers(data,label_col,{'+':1,'-':0})
    d = sl.GetOneHotEncodingForDataFrame(data,GetOneHotEncodingCols())
    return d

def RunKMeansOnLetterRecognitionDataset(rootfolder):
    vowel_data = ReadLetterRecognitionData(rootfolder)
    ReplaceLabelsWithIntegers(vowel_data,"lettr")
    DoStandardScaling(vowel_data,lambda x : x != "lettr")
    X = vowel_data.loc[:,vowel_data.columns != "lettr"]
    Y = vowel_data.loc[:,vowel_data.columns == "lettr"]
    results = RunClustering(X,Y,[10,26,30,50,80,100],10,["gmm"])
    print("done letter recognition")
    return results

def RunKMeansOnVowelRecognitionDataset(rootfolder):
    vowel_data = ReadVowelRecognitionData(rootfolder)
    ReplaceLabelsWithIntegers(vowel_data,"vowel",{"c":0,"v":1})
    DoStandardScaling(vowel_data,lambda x : x != "vowel")

    X = vowel_data.loc[:,vowel_data.columns != "vowel"]
    Y = vowel_data.loc[:,vowel_data.columns == "vowel"]
    results = RunClustering(X,Y,[2,30,40,50,60],10,["kmeans","gmm"])
    print("done vowel recognition")
    return results

def RunKMeans(X,Y,num_clusters_to_try, random_state, dim_red = True):
    results = {}
    for k in num_clusters_to_try:
        prefix = "kmeans_"+str(k)+"_"
        algo = KMeans(k,random_state = random_state,n_jobs = -2, max_iter = 1000,precompute_distances=False)
        algo.fit(X)
        predicted = algo.labels_
        ari = Compute_ARI(Y,predicted)
        metrics = ComputeClusteringMetrics(Y,predicted,X)
        print(metrics)
        results[prefix+"metrics"]=metrics
        results[prefix+"ari"] = ari
        results[prefix+"algo"] = algo
        results[prefix+"predicted"] = predicted
        bic = compute_bic(algo,X)
        results[prefix+"bic"] = bic
        if( dim_red is not None):
            new_data = None
            for i in np.arange(k):
                d = np.linalg.norm(X-algo.cluster_centers_[i],axis=1)
                if(new_data is None):
                    new_data = d
                else:
                    new_data = np.column_stack((new_data,d))
            results[prefix+"new_data"] = new_data

        print("done for {0} : {1}".format(str(k),str(bic)))

    return results

def RunGmm(X,Y,num_clusters_to_try, random_state):
    results = {}
    for k in num_clusters_to_try:
        prefix = "gmm_" + str(k)+"_"
        algo = GaussianMixture(n_components=k,verbose=0,max_iter = 1000)
        algo.fit(X)
        p = algo._estimate_weighted_log_prob(X)
        predicted = algo.predict(X)
        predicted_prob = algo.predict_proba(X)
        ari = Compute_ARI(Y,predicted)
        metrics = ComputeClusteringMetrics(Y,predicted,X)
        print(metrics)
        results[prefix+"metrics"]=metrics
        results[prefix+"ari"] = ari
        results[prefix+"algo"] = algo
        results[prefix+"predicted"] = predicted
        results[prefix+"prob"] = predicted_prob
        results[prefix+"new_data"] = predicted_prob
        results[prefix+"bic"] = algo.bic(X)
        print("done({2}) for {0} : {1}".format(str(k),str(ari),algo.converged_))

    return results

def RunClustering(X,Y,num_clusters_to_try, random_state, algos,dim_red=None):
    results = {}
    for algo in algos:
        if(algo == "kmeans"):
            results[algo] = RunKMeans(X,Y,num_clusters_to_try,random_state,dim_red)
        else:
            results[algo] = RunGmm(X,Y,num_clusters_to_try,random_state)
    return results

def DoStandardScaling(data, cols_to_transform):
    cols = []
    if(callable(cols_to_transform)):
        for col in data.columns:
            if(cols_to_transform(col)):
                cols.append(col)
    else:
        cols = cols_to_transform

    scaler = StandardScaler()
    scaler.fit(data[cols])
    data[cols] = scaler.transform(data[cols])
    return data

def DoStandardScalingNumpyArray(data):
    return data
    scaler = StandardScaler()
    scaler.fit(data)
    scaler.transform(data)

def ReplaceLabelsWithIntegers(data, label_col, label_dict = None):
    if(label_dict is None):
        labels = np.array(data[label_col].unique())
        label_dict = {}
        for i in range(len(labels)):
            label_dict[labels[i]] = i
    for label in label_dict.keys():
        data.loc[data[label_col] == label, label_col] = label_dict[label]
    return data

def PerformRandomProjections(X,Y,num_components,random_state):
    """
    For each num_components, random_state number of times
    random projection is done and that projection is kept
    that gives minimum reconstruction error
    """
    result = {}
    recons_errs = []
    for n in num_components:
        prefix = "rp_" + str(n) + "_"
        best_grp = None
        best_reconstruction_error = np.Infinity;
        reconstruction_errors = []
        for i in np.arange(random_state) + 1:
            grp = GaussianRandomProjection(n,random_state=i)
            grp.fit(X)
            _x = grp.transform(X)
            p_inv = np.linalg.pinv(grp.components_)
            X_recons = np.dot(p_inv,_x.T).T
            recons_err = ComputeReconstructionSSE(X,X_recons)
            reconstruction_errors.append(recons_err)
            #print(r"n = {0} i ={1} error = {2}".format(n,i,recons_err))
            if(best_grp is None or best_reconstruction_error > recons_err):
                best_grp = grp
                best_reconstruction_error = recons_err
        result[prefix+"data"] = best_grp.transform(X)
        result[prefix+"reconstruction_errors_all"] = reconstruction_errors
        result[prefix+"reconstruction_error"] = best_reconstruction_error
    return result

def PerformLda(X,Y,num_components,random_state):
    result = {}
    for n in num_components:
        prefix = "lda_" + str(n) + "_"
        algo = LDA(n_components=n)
        algo.fit(X,Y)
        result[prefix+"algo"] = algo
        _x = algo.transform(X)
        result[prefix+"data"] = _x
        result[prefix+"total_explained_var_ratio"] = algo.explained_variance_ratio_.sum()
        print("LDA num dim {0} : variance explained {1}".format(str(n),str(algo.explained_variance_ratio_.sum())))
        print(algo.explained_variance_ratio_)
    return result

def PerformPca(X,Y,num_components,random_state):
    result = {}
    for n in num_components:
        prefix = "pca_"+str(n) + "_"
        algo = PCA(n,random_state=random_state)
        algo.fit(X)
        result[prefix+"algo"] = algo
        _x = algo.transform(X)
        X_recons = algo.inverse_transform(_x)
        result[prefix+"data"] = _x
        result[prefix+"reconstruction_error"] = ComputeReconstructionSSE(X,X_recons)
        #print("PCA num dim {0} : reconstruction error {1} variance explained {2}".format(str(n),str(result[prefix+"reconstruction_error"]),str(algo.explained_variance_ratio_.sum())))
        #print(algo.explained_variance_)
        result[prefix+"explained_var_ratio"] = algo.explained_variance_ratio_
    return result

def PerformIca2(X,Y,num_components,random_state):
    result = {}
    for n in num_components:
        prefix = "ica_" + str(n) + "_"
        algo = FastICA(n_components=n,random_state=random_state)
        algo.fit(X)
        result[prefix+"algo"] = algo
        _x = algo.transform(X)
        X_recons = algo.inverse_transform(_x)
        result[prefix+"reconstruction_error"] = ComputeReconstructionSSE(X,X_recons)
        kt_value = np.abs(kt(_x))
        avg_kt = kt_value.mean()
        print("ICA num dim {0} : reconstruction error {1} avg kt {2}".format(str(n),str(result[prefix+"reconstruction_error"]),str(avg_kt)))
        print(np.sort(kt_value))
    return result

def PerformIca(X,Y,num_components,random_state):
    result = {}
    algo = FastICA(random_state=random_state,max_iter=800)
    algo.fit(X)
    full_mixing_matrix = algo.mixing_
    full_unmixing_matrix = algo.components_
    _x = algo.transform(X)
    kt_value = np.abs(kt(_x))
    largest_kt_values_idx = np.argsort(kt_value)[::-1]
    result["ica_kt_all"] = kt_value

    for n in num_components:
        prefix = "ica_" + str(n) + "_"
        component_idx_to_select = largest_kt_values_idx[0:n]
        mixing_matrix = full_mixing_matrix.T[component_idx_to_select,:].T
        unmixing_matrix = full_unmixing_matrix[component_idx_to_select,:]
        algo.components_ = unmixing_matrix
        algo.mixing_ = mixing_matrix

        result[prefix+"mm"] = mixing_matrix
        result[prefix+"umm"] = unmixing_matrix

        _x = algo.transform(X)
        result[prefix+"data"] = _x
        X_recons = algo.inverse_transform(_x)
        result[prefix+"reconstruction_error"] = ComputeReconstructionSSE(X,X_recons)
        n_kt_value = kt_value[component_idx_to_select]
        avg_kt = n_kt_value.mean()
        #print("ICA num dim {0} : reconstruction error {1} avg kt {2}".format(str(n),str(result[prefix+"reconstruction_error"]),str(avg_kt)))
        #print(np.sort(n_kt_value))
    return result

def ComputeMIBtwVars(X,Y,rand):
    if(X.shape[1] > 40):
        x1 = X[:,0:7]
        x2 = X[:,7:]
        s1 = mutual_info_classif(x1,Y,discrete_features=False,random_state=rand)
        s2 = mutual_info_classif(x2,Y,discrete_features=True,random_state=rand)
        return np.append(s1,s2)
    else:
        return mutual_info_classif(X,Y,discrete_features=False,random_state=rand)

def PerformMiBasedFeatureSelection(X,Y,num_components,random_state):
    result = {}
    all_scores = None
    max_n = 0
    for n in num_components:
        prefix = "mi_" + str(n) + "_"
        #algo = SelectKBest(chi2,n)
        algo = SelectKBest(lambda _x,_y : ComputeMIBtwVars(_x,_y,rand=random_state),n)
        result[prefix+"algo"] = algo
        s = MinMaxScaler(copy=True,feature_range=(0,1))
        s.fit(X)
        algo.fit(s.transform(X),Y)
        
        _x = algo.transform(X)
        result[prefix+"reconstruction_error"] = 0
        result[prefix+"data"] = _x
        if(all_scores is None or max_n < n):
            result["scores"] = algo.scores_
            max_n = n

    return result

def ComputeReconstructionSSE(X_orig, X_reconstructed):
    error = norm(X_orig-X_reconstructed)
    return error

def ScatterPlotForClustering(predicted_cluster,actual_cluster,output_file,dot_size = 1):
    markers = u.GetMarkerColorCombinations(10)
    labels = np.unique(actual_cluster)
    f = plt.figure()
    for i in np.arange(labels.size):
        label = labels[i]
        x_idx = np.where(actual_cluster == label)
        predicted_for_actual_label = predicted_cluster[x_idx]
        plt.scatter(x_idx,predicted_for_actual_label,c=markers[i]['color'],s=dot_size,marker=markers[i]['marker'],label=label)
        i = i + 1
    if(output_file is not None):
        f.savefig(output_file)

def ScatterPlotForClusteringData(X,Y,predicted_cluster,actual_cluster,output_file,dot_size = 1):
    markers = u.GetMarkerColorCombinations(10)
    labels = np.unique(actual_cluster)
    predicted_unique = np.unique(predicted_cluster.astype(int))
    print(str(predicted_unique.size) +" " +str(predicted_cluster.size))
    decorations = {}

    for predicted in predicted_unique:
        colors = np.random.rand(3,1)
        for label in labels:
            marker = "${0}$".format(str(label))
            #if(label == 1):
            #    marker = "+"
            #else:
            #    marker = "o"
            decorations[(predicted,label)] = (colors, marker)

    f = plt.figure()
    for key in decorations.keys():
        idx = list(set(np.where(actual_cluster == key[1])[0]) 
                 & set(np.where(predicted_cluster == key[0])[0]))
        plt.scatter(X[idx],Y[idx],c=decorations[key][0],marker=decorations[key][1])

    if(output_file is not None):
        f.savefig(output_file)
    plt.close(f)

def ScatterPlot(X,Y_df,output):
    Y = Y_df
    markers = u.GetMarkerColorCombinations(10)
    labels = np.unique(Y)
    y_ser = []
    i = 0
    for label in labels:
        label_data = X[Y == label]
        ser = u.YSeries(
            label_data[:,0],
            xvalues=label_data[:,1] if label_data.shape[1] > 1 else np.arange(label_data.shape[0])+1,
            line_style = ".",
            points_marker = markers[i]["marker"],
            line_color=markers[i]["color"],
            plot_legend_label = label)
        plt.scatter(ser.values,ser.xvalues,c = markers[i]["color"],marker=markers[i]["marker"],label=label)
        y_ser.append(ser)
        i = i + 1
    plt.show()
    #u.SaveDataPlotWithLegends(y_ser,filename=output,x_axis_name="dim1",y1_axis_name="dim2")

def RunExperiments(X,Y,rootfolder,clusters,dims,compute_acc=None):
    datasets = {}
    datasets["raw"] = (X,Y)
    err_series = []
    decorations = {}
    decorations["pca"] = ("o","r","pca")
    decorations["ica"] = ("x","b","ica")
    decorations["rp"] = ("+","g","rp")
    decorations["mi"] = ("o","k","mi")
    flags = [True,True,True,True]
    nn_output_lines = []
    nn_output_file = rootfolder + "/nn.csv"
    if(compute_acc is not None):
        h,l = CreateOutputLineForNN(RunNeuralNetwork(X,Y,10,compute_acc,False),"raw")
        nn_output_lines.append(h)
        nn_output_lines.append(l)

    best_bic = None
    ################### PCA #####################
    if(flags[0]):
        pca_results = PerformPca(X,Y,dims,0)
        pca_var_explained_plot = u.PreparePath(rootfolder + "/plots/pca/var.png")
        recons_err_plot = u.PreparePath(rootfolder + "/plots/err.png")
        recons_err_dict = []
        var_y = []
        err_y = []

        for dim in dims:
            key = "pca_{0}_".format(str(dim))
            datasets[key] = (DoStandardScalingNumpyArray(pca_results["{0}data".format(key)]),Y)
            err_y.append(pca_results[key+"reconstruction_error"])
            var_y = pca_results[key+"explained_var_ratio"]
            #if(compute_acc is not None and dim == 2):
            #    h,l = CreateOutputLineForNN(RunNeuralNetwork(datasets[key][0],datasets[key][1],10,compute_acc),"pca")
            #    #nn_output_lines.append(h)
            #    nn_output_lines.append(l)

        ser = u.YSeries(err_y,xvalues = dims,points_marker=decorations["pca"][0],line_color=decorations["pca"][1],plot_legend_label=decorations["pca"][2])
        recons_err_dict.append(ser)
        ser = u.YSeries(var_y,xvalues = np.arange(len(var_y)) + 1,points_marker=decorations["pca"][0],line_color=decorations["pca"][1],plot_legend_label=decorations["pca"][2])
        u.SaveDataPlotWithLegends([ser],x_axis_name="dimensions",y1_axis_name="% explained variance",filename=pca_var_explained_plot)

    ################### ICA #####################

    if(flags[1]):
        ica_kt_plot = u.PreparePath(rootfolder + "/plots/ica/kt.png")
        err_y = []
        ica_results = PerformIca(X,Y,dims,0)
        for dim in dims:
            key = "ica_{0}_".format(str(dim))
            datasets[key] = (DoStandardScalingNumpyArray(ica_results[key+"data"]),Y)
            err_y.append(ica_results[key+"reconstruction_error"])
            #if(compute_acc is not None and dim == 2):
            #    h,l = CreateOutputLineForNN(RunNeuralNetwork(datasets[key][0],datasets[key][1],10,compute_acc),"ica")
            #    nn_output_lines.append(l)

        var_y = ica_results["ica_kt_all"]
        ser = u.YSeries(err_y,xvalues = dims,points_marker=decorations["ica"][0],line_color=decorations["ica"][1],plot_legend_label=decorations["ica"][2])
        recons_err_dict.append(ser)
        ser = u.YSeries(var_y,xvalues = np.arange(len(var_y)) + 1,points_marker=decorations["ica"][0],line_color=decorations["ica"][1],plot_legend_label=decorations["ica"][2])
        u.SaveDataPlotWithLegends([ser],x_axis_name="components",y1_axis_name="kurtosis",filename=ica_kt_plot)

    ################### RP #####################
    if(flags[2]):
        rp_runs_plot = u.PreparePath(rootfolder + "/plots/rp/runs.png")
        err_y = []
        runs = 10
        rp_results = PerformRandomProjections(X,Y,dims,runs)
        runs_series = []
        markers = u.GetColorCombinations(10)
        i=0
        for dim in dims:
            key = "rp_{0}_".format(str(dim))
            datasets[key] = (DoStandardScalingNumpyArray(rp_results[key+"data"]),Y)
            err_y.append(rp_results[key+"reconstruction_error"])
            runs_ser = u.YSeries(rp_results[key+"reconstruction_errors_all"],xvalues=np.arange(runs)+1,points_marker = "o",line_color = markers[i]["color"],plot_legend_label="proj dims = "+str(dim))
            runs_series.append(runs_ser)
            i = i + 1
            #if(compute_acc is not None and dim == 2):
            #    h,l = CreateOutputLineForNN(RunNeuralNetwork(datasets[key][0],datasets[key][1],10,compute_acc),"rp")
            #    nn_output_lines.append(l)

        ser = u.YSeries(err_y,xvalues = dims,points_marker=decorations["rp"][0],line_color=decorations["rp"][1],plot_legend_label=decorations["rp"][2])
        recons_err_dict.append(ser)
        u.SaveDataPlotWithLegends(runs_series,x_axis_name="run number",y1_axis_name="reconstruction err",filename=rp_runs_plot)

        u.SaveDataPlotWithLegends(recons_err_dict,x_axis_name="dimensions",y1_axis_name="reconstruction_error",filename=recons_err_plot)

    ###################### MI Feature Selection #########################
    if(flags[3]):
        mi_results = PerformMiBasedFeatureSelection(X,Y,dims,10)
        mi_plot = u.PreparePath(rootfolder + "/plots/mi/scores.png")
        for dim in dims:
            key = "mi_{0}_".format(str(dim))
            datasets[key] = (DoStandardScalingNumpyArray(mi_results[key+"data"]),Y)
            #if(compute_acc is not None and dim == 2):
            #    h,l = CreateOutputLineForNN(RunNeuralNetwork(datasets[key][0],datasets[key][1],10,compute_acc),"mi")
            #    nn_output_lines.append(l)
        ser = u.YSeries(mi_results["scores"],xvalues = np.arange(len(mi_results["scores"])) + 1,points_marker=decorations["mi"][0],line_color=decorations["mi"][1],plot_legend_label=decorations["mi"][2])
        u.SaveDataPlotWithLegends([ser],x_axis_name="feature number",y1_axis_name="mutual information", filename=mi_plot)

    ###################### CLUSTERING #########################
    clustering_output_file = rootfolder + "/clustering.csv"
    clustering_plots_output_root = u.PreparePath(rootfolder + "/plots")
    lines = []
    lines.append("clustering,dim_red_method,k,p,ami_raw,ami_true,sc,bic")
    raw_clustering_results = {}
    best_bic_raw_clustering = {}
    curr_best_bic = {}
    actual_labels = Y
    for dim in dims:
        for algo in ["raw","ica","rp","mi","pca"]:
            raw_data_plot_done = False
            key = "{0}_{1}_".format(algo,str(dim))
            if(algo == "raw"):
                key = "raw"
            dataset = datasets[key]
            for cluster in clusters:
                for mthd in ["kmeans","gmm"]:
                    raw_key = "{0}_{1}".format(str(cluster),mthd)
                    print("doing clustering for dim = {0} {1} k = {2} {3}".format(str(dim),algo,str(cluster), mthd))
                    c_key = "{0}_{1}_predicted".format(mthd,str(cluster))
                    c_key1 = "{0}_{1}_".format(mthd,str(cluster))
                    if(algo == "raw" and raw_key in raw_clustering_results):
                        results = raw_clustering_results[raw_key]
                    else:
                        #if(algo == "raw" and cluster == 2 and compute_acc):
                        #    results = RunClustering(dataset[0],dataset[1],[cluster],0,[mthd],dim)[mthd]
                        #    h,l = CreateOutputLineForNN(RunNeuralNetwork(results[c_key.replace("predicted","new_data")],dataset[1],10,compute_acc),mthd)
                        #    nn_output_lines.append(l)
                        #else:
                        results = RunClustering(dataset[0],dataset[1],[cluster],0,[mthd],dim)[mthd]
                        if(algo == "raw"):
                           raw_clustering_results[raw_key] = results
                        if(compute_acc):
                            mthd_key = mthd+algo if algo == "raw" else mthd+algo+str(cluster)+str(dim)
                            if((mthd_key not in curr_best_bic) or (curr_best_bic[mthd_key] > results[c_key1+"bic"])):
                                curr_best_bic[mthd_key] = results[c_key1+"bic"]
                                best_bic_raw_clustering[mthd_key] = (results[c_key1+"new_data"],dataset[1],results[c_key1+"metrics"]["ami"],results[c_key1+"bic"])
                                print("new best {0} {1}".format(c_key1,str(results[c_key1+"bic"])))

                    clustering_prediction_file = u.PreparePath(rootfolder + "/clustering_output/mthd={0}_k={1}_d={2}_algo={3}.csv".format(mthd,str(cluster),str(dim),algo))
                    np.savetxt(clustering_prediction_file,results[c_key])
                    bic = c_key.replace("predicted","bic")
                    bic = results[bic]
                    act = ComputeClusteringMetrics(actual_labels,results[c_key],dataset[0])
                    raw = ComputeClusteringMetrics(raw_clustering_results[raw_key][c_key],results[c_key],dataset[0])
                    line = "{0},{1},{2},{3},{4},{5},{6},{7}".format(mthd,algo,str(cluster),str(dim),str(raw["ami"]),str(act["ami"]),str(raw["sl"]),str(bic))
                    print(line)
                    plot_output_file = clustering_plots_output_root + "/{0}_{1}_{2}_{3}.png".format(mthd,str(cluster),algo,str(dim))
                    #if(mthd == "gmm"):
                    #    prob_output_file = rootfolder + "/{0}_{1}_{2}_{3}.csv".format(mthd,str(cluster),algo,str(dim))
                    #    np.savetxt(prob_output_file,results[c_key.replace("predicted","prob")],delimiter=",")
                    ScatterPlotForClustering(results[c_key],actual_labels,plot_output_file)
                    if(dim == 2 and algo != "raw"):
                        if(raw_data_plot_done == False):
                            plot_output_file = clustering_plots_output_root + "/{0}_{1}_data.png".format(mthd,algo)
                            ScatterPlotForClusteringData(dataset[0][:,0],dataset[0][:,1],np.zeros_like(actual_labels),actual_labels,plot_output_file)
                            raw_data_plot_done = True
                        plot_output_file = clustering_plots_output_root + "/{0}_{1}_{2}_data.png".format(mthd,str(cluster),algo)
                        ScatterPlotForClusteringData(dataset[0][:,0],dataset[0][:,1],results[c_key],actual_labels,plot_output_file)
                    lines.append(line)

    #if(compute_acc):
    #    keys_to_output = {"kmeansraw":"kmeans","gmmraw":"gmm","gmmpca":"pca","gmmica":"ica","gmmrp":"rp","gmmmi":"mi"}
    #    for key in keys_to_output.keys():
    #        if("raw" not in key):
    #            curr_best = None
    #            for cluster in clusters:
    #                datakey = key+str(cluster)
    #                if(curr_best is None or best_bic_raw_clustering[datakey][2] > curr_best):
    #                    curr_best = best_bic_raw_clustering[datakey][2]
    #                    _X = best_bic_raw_clustering[datakey][0]
    #                    _Y = best_bic_raw_clustering[datakey][1]
    #        else:
    #            _X = best_bic_raw_clustering[key][0]
    #            _Y = best_bic_raw_clustering[key][1]

    #        h,l = CreateOutputLineForNN(RunNeuralNetwork(_X,_Y,10,compute_acc,scale=False if "gmmraw" == key else True),keys_to_output[key])
    #        nn_output_lines.append(l)
    #    u.WriteTextArrayToFile(nn_output_file,nn_output_lines)

    if(compute_acc):
        keys_to_output = {"kmeansraw":"kmeans","gmmraw":"gmm","pca":"pca","ica":"ica","rp":"rp","mi":"mi"}
        for key in keys_to_output.keys():
            if("raw" not in key):
                dim_best_val = None
                dim_result = None
                for dim in dims:
                    best = {} # {x,y,p,k,bic,ami}
                    for cluster_mthd in ["kmeans","gmm"]:
                        for cluster in clusters:
                            datakey = cluster_mthd+key+str(cluster)+str(dim)
                            if(cluster_mthd not in best or best_bic_raw_clustering[datakey][2] > best[cluster_mthd][4]):
                                best[cluster_mthd] = (best_bic_raw_clustering[datakey][0],best_bic_raw_clustering[datakey][1],dim,cluster,best_bic_raw_clustering[datakey][3],best_bic_raw_clustering[datakey][2])
                    curr_val = (best["kmeans"][5] + best["gmm"][5]) / 2
                    if(dim_best_val is None or dim_best_val < curr_val):
                        dim_best_val = curr_val
                        dim_result = best

                _X = dim_result["gmm"][0]
                _Y = dim_result["gmm"][1]
            else:
                _X = best_bic_raw_clustering[key][0]
                _Y = best_bic_raw_clustering[key][1]

            h,l = CreateOutputLineForNN(RunNeuralNetwork(_X,_Y,10,compute_acc,scale=False if "gmmraw" == key else True),keys_to_output[key])
            nn_output_lines.append(l)
        u.WriteTextArrayToFile(nn_output_file,nn_output_lines)

    u.WriteTextArrayToFile(clustering_output_file,lines)

def RunNeuralNetwork(X,Y,random_state,compute_accuracy=False,scale=True):
    X,X_test,Y,Y_test = train_test_split(X,Y,stratify=Y,train_size=0.70,test_size=0.30,random_state=random_state)
    if(scale):
        s = StandardScaler()
        s.fit(X)
        X = s.transform(X)
        X_test = s.transform(X_test)
    hidden_layers = [(30,)]
    init_learning_rates = [0.1,0.01,0.001,0.0001]
    alpha =[0.01,0.1,1,10,100]
    momentum = 0.9
    max_iter = 500

    #for doing 3-fold CV
    param_grid = {"alpha":alpha,"learning_rate_init":init_learning_rates,"hidden_layer_sizes":hidden_layers}
    classifier = MLPClassifier(
        activation="logistic",
        momentum=momentum,
        early_stopping = False,
        verbose=False,
        random_state=random_state,
        solver="sgd",
        max_iter=max_iter,tol = 0.000001)

    gscv = GridSearchCV(classifier,param_grid,n_jobs=3)
    gscv.fit(X,Y)
    best_params = gscv.best_params_
    classifier = MLPClassifier(
        hidden_layer_sizes=best_params["hidden_layer_sizes"],
        activation="logistic",
        momentum=momentum,
        early_stopping = False,
        verbose=True,
        random_state=random_state,
        solver="sgd",
        max_iter=max_iter,
        learning_rate_init=best_params["learning_rate_init"],
        alpha=best_params["alpha"],tol = 0.000001)
    start = time.clock()
    classifier.fit(X,Y)
    end = time.clock()
    Y_predicted = classifier.predict(X_test)
    metrics = sl.ComputePrecisionRecallForPythonOutputFormat(pd.DataFrame({'actual':Y_test, 'predicted':Y_predicted}),1,False,compute_accuracy)
    result = {}
    result['test_acc'] = metrics[2]

    Y_predicted = classifier.predict(X)
    metrics = sl.ComputePrecisionRecallForPythonOutputFormat(pd.DataFrame({'actual':Y, 'predicted':Y_predicted}),1,False,compute_accuracy)
    result['train_acc'] = metrics[2]
    result['time'] = end-start
    result['iter'] = classifier.n_iter_
    return result

def CreateOutputLineForNN(result, algo):
    header = "algo,train_acc,test_acc,time,iter"
    line = "{0},{1},{2},{3},{4}".format(algo,str(result["train_acc"]),str(result["test_acc"]),str(result["time"]),str(result["iter"]))
    print("{0} train : {1} test : {2}".format(algo,result["train_acc"],result["test_acc"]))
    return header,line

def ComputeFinalResults(rootfolder):
    clustering_stats = pd.read_csv(rootfolder+"/clustering.csv")
    final_output_file = rootfolder+"/best_metrics.csv"
    data = u.FilterRows(clustering_stats, lambda x : x['p'] == 2)
    dim_red = ["ica","pca","rp","mi"]
    clustering = ["kmeans","gmm"]
    lines = []
    lines.append("clustering,dim_red,k,p,ami_raw,ami_true,sc,bic")
    raw_predictions = {}
    for c in clustering:
        d = data.loc[(data['dim_red_method'] == "raw") & (data['clustering'] == c),:]
        d = d.loc[d['bic'] == np.min(d['bic']),:]
        clusters_file = rootfolder + "/clustering_output/mthd={0}_k={1}_d=2_algo=raw.csv".format(c,d.iloc[0]['k'])
        raw_predictions[c] = np.loadtxt(clusters_file,delimiter=',')

    for dr in dim_red:
        for c in clustering:
            d = data.loc[(data['dim_red_method'] == dr) & (data['clustering'] == c),:]
            d = d.loc[d['bic'] == np.min(d['bic']),:]
            clusters_file = rootfolder + "/clustering_output/mthd={0}_k={1}_d=2_algo={2}.csv".format(c,d.iloc[0]['k'],dr)
            predicted = np.loadtxt(clusters_file,delimiter=',')
            ami = metrics.adjusted_mutual_info_score(raw_predictions[c],predicted)
            lines.append(u.ConcatToStr(",",[c,dr,d.iloc[0]['k'],ami,d.iloc[0]['ami_true'],d.iloc[0]['sc'],d.iloc[0]['bic']]))

    u.WriteTextArrayToFile(final_output_file,lines)

def GetBestRawClustering(rootfolder):
    clustering_stats = pd.read_csv(rootfolder+"/clustering.csv")
    final_output_file = rootfolder+"/best_metrics.csv"
    data = u.FilterRows(clustering_stats, lambda x : x['p'] == 2)
    dim_red = ["ica","pca","rp","mi"]
    clustering = ["kmeans","gmm"]
    lines = []
    lines.append("clustering,dim_red,k,p,ami_raw,ami_true,sc,bic")
    raw_predictions = {}
    for c in clustering:
        d = data.loc[(data['dim_red_method'] == "raw") & (data['clustering'] == c),:]
        d = d.loc[d['bic'] == np.min(d['bic']),:]
        clusters_file = rootfolder + "/clustering_output/mthd={0}_k={1}_d=2_algo=raw.csv".format(c,d.iloc[0]['k'])
        raw_predictions[c] = (np.loadtxt(clusters_file,delimiter=','),d.copy())
    return raw_predictions

def GetAmiWithRawPredictions(rootfolder,rawpredictions,dr,p,k,mthd):
    ami_scores = {}
    for cluster_mthd in [mthd]:
        clusters_file = rootfolder + "/clustering_output/mthd={0}_k={1}_d={2}_algo={3}.csv".format(cluster_mthd,k,p,dr)
        predicted = np.loadtxt(clusters_file,delimiter=',')
        ami_scores[cluster_mthd] = metrics.adjusted_mutual_info_score(rawpredictions[cluster_mthd][0],predicted)
    return ami_scores


def ComputeFinalResults1(rootfolder,clusters,dims):
    raw_results = GetBestRawClustering(rootfolder)
    clustering_results = pd.read_csv(rootfolder+"/clustering.csv")
    final_output_file = rootfolder+"/best_metrics.csv"
    best_raw_clustering_output = rootfolder+"/best_raw_clustering.csv"
    o = pd.concat([raw_results["kmeans"][1], raw_results["gmm"][1]])
    o.to_csv(best_raw_clustering_output)
    dim_red = ["mi","pca","ica","rp"]
    lines = []
    lines.append("clustering,dim_red,k,p,ami_raw,ami_true,sc,bic")
    raw_predictions = {}

    output = None
    for dr in dim_red:
        data = u.FilterRows(clustering_results,lambda x : x["dim_red_method"] == dr)
        dim_best_val = None
        dim_result = None
        for dim in dims:
            best = {} # {p,k,ami}
            for cluster_mthd in ["kmeans","gmm"]:
                for cluster in clusters:
                    print("{0},{1},{2},{3}".format(dr,str(dim),cluster_mthd,str(cluster)))
                    d = data.loc[(data['clustering'] == cluster_mthd)&(data['k'] == cluster) & (data['p'] == dim)]
                    row = d.head(1).copy()
                    if(cluster_mthd not in best or best[cluster_mthd]['bic'].iloc[0] > row['bic'].iloc[0]):
                        best[cluster_mthd] = row
            curr_val = (best["kmeans"]['ami_true'].iloc[0] + best["gmm"]['ami_true'].iloc[0]) / 2
            #curr_val = (best["kmeans"]['ami_true'].iloc[0] + best["gmm"]['ami_true'].iloc[0]) / 2
            #curr_val = np.minimum(best["kmeans"]['ami_true'].iloc[0], best["gmm"]['ami_true'].iloc[0])
            if(dim_best_val is None or dim_best_val < curr_val):
                dim_best_val = curr_val
                dim_result = best.copy()
        for c in ["kmeans","gmm"]:
            ami_raw = GetAmiWithRawPredictions(rootfolder,raw_results,dr,dim_result[c].iloc[0]["p"],dim_result[c].iloc[0]["k"],c)
            lines.append("{0},{1},{2},{3},{4},{5},{6},{7}".format(c,str(dim_result[c].iloc[0]["dim_red_method"]),str(dim_result[c].iloc[0]["k"]),str(dim_result[c].iloc[0]["p"]),str(ami_raw[c]),str(dim_result[c].iloc[0]["ami_true"]),str(dim_result[c].iloc[0]["sc"]),str(dim_result[c].iloc[0]["bic"])))
        #if(output is None):
        #    output = pd.concat([dim_result["kmeans"],dim_result["gmm"]])
        #else:
        #    output = pd.concat([output,dim_result["kmeans"],dim_result["gmm"]])

    u.WriteTextArrayToFile(final_output_file,lines)
    #output.to_csv(final_output_file)


def PlotClusteringMetricsForDimsAndDimRed(rootfolder, data, dims, dim_reds,k):
    colors = {"ica":'r','pca':'b','rp':'g','mi':'k','raw':'orange'}
    markers = {"kmeans":'o',"gmm":'x'}
    metrics = ["ami_raw","ami_true","sc","bic"]
    for _k in k:
        filter = lambda x : x['k'] == _k
        filtered_data = u.FilterRows(data,filter)
        for metric in metrics:
            ser = []
            outputfile = u.PreparePath(rootfolder + "/plots/metrics/dr_{0}_k={1}.png".format(metric,str(_k)))
            for dim_red in dim_reds:
                d = data.loc[(data['dim_red_method'] == dim_red) & (data['k'] == _k) & (data['clustering'] == 'kmeans') ,:]
                ser.append(u.YSeries(d[metric],xvalues=d['p'],line_color=colors[dim_red],points_marker=markers['kmeans'],plot_legend_label="{0}-{1}".format(dim_red,'kmeans')))
                d = data.loc[(data['dim_red_method'] == dim_red) & (data['k'] == _k) & (data['clustering'] == 'gmm') ,:]
                ser.append(u.YSeries(d[metric],xvalues=d['p'],line_color=colors[dim_red],points_marker=markers['gmm'],plot_legend_label="{0}-{1}".format(dim_red,'gmm')))
            u.SaveDataPlotWithLegends(ser,x_axis_name="dimensions",y1_axis_name=metric,filename = outputfile)

def PlotClusteringMetricsForDimsAndBic(rootfolder, data, dims, dim_reds,k,metrics = ["ami_raw","ami_true","sc","bic"]):
    colors = {"ica":'r','pca':'b','rp':'g','mi':'k','raw':'orange'}
    markers = {"kmeans":'o',"gmm":'x'}
    for _k in dims:
        for metric in metrics:
            for dim_red in dim_reds:
                ser = []
                outputfile = u.PreparePath(rootfolder + "/plots/metrics/dr_{0}_p={1}_{2}.png".format(metric,str(_k),dim_red))
                d = data.loc[(data['dim_red_method'] == dim_red) & (data['p'] == _k) & (data['clustering'] == 'kmeans') ,:]
                ser.append(u.YSeries(d[metric],xvalues=d['k'],line_color=colors[dim_red],points_marker=markers['kmeans'],plot_legend_label="{0}-{1}".format(dim_red,'kmeans')))
                d = data.loc[(data['dim_red_method'] == dim_red) & (data['p'] == _k) & (data['clustering'] == 'gmm') ,:]
                ser.append(u.YSeries(d[metric],xvalues=d['k'],line_color=colors[dim_red],points_marker=markers['gmm'],plot_legend_label="{0}-{1}".format(dim_red,'gmm')))
                u.SaveDataPlotWithLegends(ser,x_axis_name="k",y1_axis_name=metric,filename = outputfile)

def PlotClusteringMetrics(rootfolder, data, k,dim='raw',p=2):
    filter = lambda x : x['dim_red_method'] == dim and x['p'] == p
    filtered_data = u.FilterRows(data,filter)
    metrics = ["ami_raw","ami_true","sc","bic"]
    gmm_data = filtered_data.loc[filtered_data['clustering'] == "gmm",:]
    kmeans_data = filtered_data.loc[filtered_data['clustering'] == "kmeans",:]
    d = {"kmeans":('o','b','kmeans'),"gmm":('x','r','gmm')}
    for metric in metrics:
        outputfile = u.PreparePath(rootfolder + "/plots/metrics/{0}_{1}_p={2}.png".format(metric,dim,str(p)))
        kmeans_ser = u.YSeries(kmeans_data[metric],xvalues=kmeans_data["k"],points_marker = d["kmeans"][0],line_color=d["kmeans"][1],plot_legend_label=d["kmeans"][2])
        gmm_ser = u.YSeries(gmm_data[metric],xvalues=gmm_data["k"],points_marker = d["gmm"][0],line_color=d["gmm"][1],plot_legend_label=d["gmm"][2])
        u.SaveDataPlotWithLegends([kmeans_ser,gmm_ser],x_axis_name="number of clusters",y1_axis_name=metric,filename=outputfile)
    # computing avg prob of belonging to a cluster
    #avg_assign_probs = {}
    #for _k in k:
    #    file = rootfolder + "/gmm_{0}_raw_2.csv".format(str(_k))
    #    probs = pd.read_csv(file)
    #    best_prob = probs.apply(np.max,axis = 1)
    #    print("k = {0} {1}".format(str(_k),u.ConcatToStr(",",np.percentile(best_prob,[10,30,50,90,99]))))
    #    avg_assign_probs[_k] = np.mean(best_prob)
    #lines = []
    #headers = []
    #values = []
    #for key in k:
    #    headers.append(str(key))
    #    values.append(str(avg_assign_probs[key]))
    #outputfile = rootfolder + "/gmm_avg_raw.csv"
    #u.WriteTextArrayToFile(outputfile,[",".join(headers),",".join(values)])


def compute_bic(kmeans,X):
    """
    Computes the BIC metric for a given clusters

    Parameters:
    -----------------------------------------
    kmeans:  List of clustering object from scikit learn

    X     :  multidimension np array of data points

    Returns:
    -----------------------------------------
    BIC value
    """
    # assign centers and labels
    centers = [kmeans.cluster_centers_]
    labels  = kmeans.labels_
    #number of clusters
    m = kmeans.n_clusters
    # size of the clusters
    n = np.bincount(labels)
    #size of data set
    N, d = X.shape

    #compute variance for all clusters beforehand
    cl_var = (1.0 / (N - m) / d) * sum([sum(distance.cdist(X[np.where(labels == i)], [centers[0][i]], 
             'euclidean')**2) for i in range(np.minimum(m,n.size))])

    const_term = 0.5 * m * np.log(N) * (d+1)

    BIC = np.sum([n[i] * np.log(n[i]) -
               n[i] * np.log(N) -
             ((n[i] * d) / 2) * np.log(2*np.pi*cl_var) -
             ((n[i] - 1) * d/ 2) for i in range(np.minimum(m,n.size))]) - const_term

    return(-BIC)

def main():
    print("running")
    rootfolder = r"C:\Users\shkhandu\OneDrive\Gatech\Courses\ML\Assignment3\skhanduja7\data"

    output = rootfolder
    output_lr = u.PreparePath(output + "/lr")
    X,Y = ReadLetterRecognitionData(rootfolder)
    RunExperiments(X,Y,output_lr,[10,15,26,35,50],[2,4,8,12,16])
    PlotClusteringMetricsForDimsAndBic(output_lr,pd.read_csv(output_lr + '/clustering.csv'),[2,4,8,12,16],["raw","pca","ica","rp","mi"],[10,15,26,35,50],metrics=["bic"])
    PlotClusteringMetrics(output_lr,pd.read_csv(output_lr + '/clustering.csv'),[])
    ComputeFinalResults1(output_lr,[10,15,26,35,50],[2,4,8,12])

    output_lr = u.PreparePath(output + "/cs")
    X,Y = ReadCreditScreeningData(rootfolder)
    RunExperiments(X,Y,output_lr,[2,5,10,15,20],[2,5,10,20,30],True)
    PlotClusteringMetricsForDimsAndBic(output_lr,pd.read_csv(output_lr + '/clustering.csv'),[2,5,10,20,30],["raw","pca","ica","rp","mi"],[2,5,10,15,20],metrics=["bic"])
    PlotClusteringMetrics(output_lr,pd.read_csv(output_lr + '/clustering.csv'),[])
    ComputeFinalResults1(output_lr,[2,5,10,15,20],[2,5,10,30,20])

if __name__ == "__main__":
    main()
 

