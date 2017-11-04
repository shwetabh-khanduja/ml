import pandas as pd
import numpy as np
import cmath
import utils as u
from sklearn.decomposition import PCA

def CompteStats(rootfolder,files = ["mimic.csv","ga.csv","sa.csv","rhc.csv"], ids = [0,1,2,3,4]):
    stats_agg = rootfolder + "/stats_agg.csv";
    stats = rootfolder + "/stats.csv";
    agg_cols = 'converged_iters,converged_time,total_iters,total_time,value'.split(',')
    agg_cols_f = dict([ (c,'mean') for c in agg_cols])
    results = []
    dfs = []

    for file in files:
        d = GetResults(rootfolder,file,ids)
        dfs.append(d)
    final = pd.concat(dfs,ignore_index=True)
    data_gpd = final.groupby(['size','algo'], as_index = False).agg(agg_cols_f)
    final.to_csv(stats)
    data_gpd.to_csv(stats_agg)

def Plot(rootfolder, cols_to_plot_dict):
    data = pd.read_csv(rootfolder + "/stats_agg.csv");
    sizes = data['size'].unique()
    algos = ['rhc','sa','mimic','ga']
    algo_decoration = {'mimic':('r','o','mimic'),'ga':('g','s','genetic algo'),'sa':('b','+','sim annealing'),'rhc':('k','*','rhc')}
    for col in cols_to_plot_dict.keys():
        y_ser = []
        for algo in algos:
            x = data[data['algo'] == algo].loc[:,'size']
            y = data[data['algo'] == algo].loc[:,col]
            legend_label = algo_decoration[algo][2]
            marker = algo_decoration[algo][1]
            color = algo_decoration[algo][0]
            yseries = u.YSeries(y,points_marker = marker,line_color=color,xvalues=x,plot_legend_label = legend_label)
            y_ser.append(yseries)
        y_axis_name = cols_to_plot_dict[col]
        x_axis_name = 'size'
        savepath = u.PreparePath(rootfolder + "/plots/" + col + ".png")
        u.SaveDataPlotWithLegends(y_ser,filename = savepath,y1_axis_name = y_axis_name,x_axis_name = x_axis_name)

def GetResults(rootfolder,file,ids):
    dfs = []
    for id in ids:
        path = rootfolder + "/" + str(id) + "/" + file
        times_path = rootfolder + "/" + str(id) + "/times.csv";
        fnevals_path = rootfolder + "/" + str(id) + "/fnevals.csv";
        times = pd.read_csv(times_path).set_index('size');
        fnevals = pd.read_csv(fnevals_path).set_index('size');
        algo = file.split('.')[0]

        d = GetAggNums(path,times,fnevals,algo)
        d['id'] = id
        d['algo'] = algo
        dfs.append(d)
    final = pd.concat(dfs,ignore_index=True)
    return final

def GetAggNums(file,times,fnevals,algo):
    data = pd.read_csv(file)
    c_values = []
    c_iters = []
    iters = []
    sizes = []
    c_fnevals = []
    c_times = []
    total_times = []
    total_evals = []

    for size in data['size'].unique():
        filtered_data = data[data['size'] == size]
        max_value = np.max(filtered_data['fn_value'])
        close_match_indxs = filtered_data[filtered_data['fn_value'] == max_value].index
        converged_index = np.min(np.array(close_match_indxs))
        assert(cmath.isclose(max_value,filtered_data.loc[converged_index]['fn_value']))
        converged_iter = filtered_data.loc[converged_index]['iters']
        total_iters = filtered_data['iters'].max()
        total_eval = fnevals.loc[size][algo]
        total_time = times.loc[size][algo]

        c_time = converged_iter * total_time / total_iters
        c_evals = converged_iter * total_eval / total_iters

        sizes.append(size)
        iters.append(total_iters)
        c_iters.append(converged_iter)
        c_values.append(max_value)
        total_times.append(total_time)
        total_evals.append(total_eval)
        c_fnevals.append(c_evals)
        c_times.append(c_time)

    return pd.DataFrame({'size':sizes,"total_iters":iters,'converged_iters':c_iters,'value':c_values,'converged_evals':c_fnevals,'converged_time':c_times,'total_evals':total_evals,'total_time':total_times})

def NeuralNetworkResults(rootfolder):
    data,ga = ReadNNetResultsFile10k(rootfolder)
    ga = ga.set_index('size')
    #data.loc[:,data.columns != 'loss'].to_csv(r'c:\temp\nnets10knew.csv')
    #ga.loc[:,data.columns != 'loss'].to_csv(r'c:\temp\nnets_ga.csv')
    algos = ['ga','rhc','sa','bp']
    algo_decoration = {'bp':('r','o','backprop'),'ga':('g','s','genetic algo'),'sa':('b','+','sim annealing'),'rhc':('k','*','rhc')}
    y_ser = []
    time_y_ser = []
    loss_y_ser = []

    size_for_loss_curves = {20:[],90:[],100:[]}
    size_for_loss_ga = {20:[],50:[],100:[]}

    for algo in algos:
        filtered_data = data[data['algo'] == algo].set_index('size')
        train_ser = []
        valid_ser = []
        x = []
        time = []
        for size in [20,30,40,50,60,70,80,90,100]:
            x.append(size)
            train_ser.append(filtered_data.loc[size]['train_f1'])
            valid_ser.append(filtered_data.loc[size]['valid_f1'])
            time.append(filtered_data.loc[size]['time'])

            if(size in size_for_loss_curves):
                y_vals = np.array(filtered_data.loc[size]['loss'].split(';'),dtype=float)
                x_vals = np.arange(y_vals.size) + 1
                _ser = u.YSeries(y_vals,xvalues=x_vals,line_color=algo_decoration[algo][0],points_marker='.',legend_marker='o',plot_legend_label=algo_decoration[algo][2])
                size_for_loss_curves[size].append(_ser)

            if(algo == "ga" and size in size_for_loss_ga):
                ga_y_vals = np.array(filtered_data.loc[size]['loss'].split(';'),dtype=float)
                bad_ga_y_vals = np.array(ga.loc[size]['loss'].split(';'),dtype=float)[0:10000]
                _ser = u.YSeries(ga_y_vals,xvalues=np.arange(ga_y_vals.size)+1,line_color='b',points_marker='.',legend_marker='o',plot_legend_label='tournament selection')
                size_for_loss_ga[size].append(_ser)
                _ser = u.YSeries(bad_ga_y_vals,xvalues=np.arange(ga_y_vals.size)+1,line_color='r',points_marker='.',legend_marker='o',plot_legend_label='roulette wheel')
                size_for_loss_ga[size].append(_ser)

        y_ser.append(u.YSeries(train_ser,xvalues=x,line_color=algo_decoration[algo][0],points_marker='x',plot_legend_label=algo_decoration[algo][2]+"-train"))
        y_ser.append(u.YSeries(valid_ser,xvalues=x,line_color=algo_decoration[algo][0],points_marker='o',plot_legend_label=algo_decoration[algo][2]+"-valid"))
        time_y_ser.append(u.YSeries(time,xvalues=x,line_color = algo_decoration[algo][0],points_marker=algo_decoration[algo][1],plot_legend_label=algo_decoration[algo][2]))
        x_axis_name = 'trainset size %'

    y_axis_name = 'f-measure'
    plot_file = u.PreparePath(rootfolder+"/plot10k/learning_curves.png")
    time_plot_file = u.PreparePath(rootfolder+"/plot10k/time.png")
    u.SaveDataPlotWithLegends(y_ser,filename=plot_file,x_axis_name=x_axis_name,y1_axis_name=y_axis_name)
    u.SaveDataPlotWithLegends(time_y_ser,filename=time_plot_file,x_axis_name=x_axis_name,y1_axis_name="Time (MilliSec)")
    for key in size_for_loss_curves.keys():
        loss_plot_file = u.PreparePath(rootfolder+"/plot10k/loss_curves_{0}.png".format(str(key)))
        u.SaveDataPlotWithLegends(size_for_loss_curves[key],filename=loss_plot_file,title="Size % : " + str(key),x_axis_name = 'iters',y1_axis_name="Loss")

    for key in size_for_loss_ga.keys():
        loss_plot_file = u.PreparePath(rootfolder+"/plot10k/loss_curves_ga_{0}.png".format(str(key)))
        u.SaveDataPlotWithLegends(size_for_loss_ga[key],filename=loss_plot_file,title="Size % : " + str(key),x_axis_name = 'iters',y1_axis_name="Loss")

def ReadNNetResultsFile10k(rootpath):
    file = rootpath + "/nnet.output.csv";
    ga_noprogressfile = rootpath + "/output_test_lr_10k-iters_30-hiddenlayers_crowding.csv";
    data = pd.read_csv(file)
    ga_noprogress = pd.read_csv(ga_noprogressfile)
    return data, ga_noprogress

    ga_noprogress = u.FilterRows(data, lambda x : x['algo'] == 'ga')
    data = u.FilterRows(data, lambda x : x['algo'] != 'ga')
    ga_progress = pd.read_csv(rootpath + "/output_test_lr_10k-iters_30-hiddenlayers_ga.csv")
    data = pd.concat([data, ga_progress], ignore_index = True)
    return data, ga_noprogress

def ReadNNetResultsFile20k(rootpath):
    file = rootpath + "/output_test_lr_20k-iters_30-hiddenlayers_all.csv"
    data = pd.read_csv(file)
    return data,None

def PlotPerIterationCurves(rootFolder, outputfolder):
    mimic = pd.read_csv(rootFolder+"/mimic.csv");
    sa = pd.read_csv(rootFolder+"/sa.csv");
    rhc = pd.read_csv(rootFolder+"/rhc.csv");
    ga = pd.read_csv(rootFolder+"/ga.csv");
    sizes = np.array(mimic['size'].unique())
    algo_decoration = {'mimic':('r','o','mimic',mimic),'ga':('g','s','genetic algo',ga),'sa':('b','+','sim annealing',sa),'rhc':('k','*','rhc',rhc)}
    def f(data, name):
        x = data['iters']
        y = data['fn_value']
        deco = algo_decoration[name]
        return u.YSeries(y,xvalues=x,points_marker='.',plot_legend_label=deco[2],legend_marker='o',line_color=deco[0])

    for size in sizes:
        size_root = u.PreparePath(outputfolder + "/itercurves_"+str(size)+".png")
        y_ser = []
        for key in algo_decoration.keys():
            d = u.FilterRows(algo_decoration[key][3],lambda x : x['size'] == size).head(10000)
            y_ser.append(f(d,key))
        u.SaveDataPlotWithLegends(y_ser,x_axis_name="iters",x=None,y1_axis_name="fn value",filename=size_root)

def PlotTempVariationCurvesForSa(rootfolder, algoname, temperatures):
    """

    """
    rhcdata = u.FilterRows(pd.read_csv(rootfolder + "/" + algoname + "/stats_agg.csv"),lambda x : x['algo'] == 'rhc')
    y_Ser = []
    y_Ser.append(u.YSeries(rhcdata['converged_iters'],xvalues = rhcdata['size'],points_marker="*",line_color="k",plot_legend_label="rhc"))
    data_dict = {}
    deco = {'0':("r","x"),'90':("b","o"),'95':("g","+"),'99':("orange",">")}
    for t in temperatures:
        path = rootfolder + "/" + algoname + "_" + t
        CompteStats(path,["sa.csv"])
        data_dict[t] = pd.read_csv(path +"/stats_agg.csv")
        y_Ser.append(u.YSeries(data_dict[t]['converged_iters'],xvalues = data_dict[t]['size'],points_marker=deco[t][1],line_color=deco[t][0],plot_legend_label="sa_"+t))
    outputfile = rootfolder + "/" + algoname + "/plots/sa_temperatures.png"
    u.SaveDataPlotWithLegends(y_Ser,y1_axis_name = "iterations to converge", x_axis_name="size",filename=outputfile)

def PlotMimicProbabilities(inputfile, outputfile,topk,size):
    data = pd.read_csv(inputfile)
    data = data[data['size'] == size].head(topk)
    y_ser = []
    ser = u.YSeries(data['n2'],line_color='k',xvalues = data['iters'],points_marker='*',plot_legend_label="P(X = 1 | parent = 0)")
    y_ser.append(ser)
    ser = u.YSeries(data['root_node_prob'],line_color='r',xvalues = data['iters'],points_marker='x',plot_legend_label="P(X_root = 1)")
    y_ser.append(ser)
    ser = u.YSeries(data['n1'],line_color='g',xvalues = data['iters'],points_marker='o',plot_legend_label="P(X = 1 | parent = 1)")
    y_ser.append(ser)
    
    u.SaveDataPlotWithLegends(y_ser,filename = outputfile,y1_axis_name="probabilities",x_axis_name="iterations")

def Main(rootfolder = r'C:\Users\shkhandu\OneDrive\Gatech\Courses\ML\Assignment2\skhanduja7\data'):
    problems = ["knapsack","twocolors","countones"]
    for p in problems:
        CompteStats(rootfolder + '/'+p)
        Plot(rootfolder + '/'+p,{'converged_iters':'Iterations','value':'Value','converged_time': 'Time (sec)'})

    PlotTempVariationCurvesForSa(rootfolder,"knapsack",['0','90','95','99'])
    PlotMimicProbabilities(
        rootfolder + "/countones/4/mimic.csv",
        rootfolder + "/countones/plots/prob_4_50.png",
        200, ## first k iterations to plot
        50)  ## size of data for which to plot

    NeuralNetworkResults(rootfolder + '/nnets')

if __name__ == "__main__":
    Main(r'C:\Users\shkhandu\OneDrive\Gatech\Courses\ML\Assignment2\skhanduja7\data')