import pandas as pd
import numpy as np
import utils as u

def SwapColValues(x):
    if(x['solver'] == 'pi'):
        temp = x['totalVIters']
        x['totalVIters'] = x['totalPIters']
        x['totalPIters'] = temp
    return x

def RenameCols(x):
    if(x == 'totalVIters'):
        return 'totalIters'
    elif(x == 'totalPIters'):
        return 'totalSweeps'
    elif(x == 'totalTime'):
        return 'totalTime(Millisecs)'
    elif(x == 'actualValIters'):
        return 'sweepsPerIteration'
    elif(x == 'maxInnerVi'):
        return 'maxSweepsPerIteration'
    else:
        return x

def ConcatenateOutputFiles(rootfolder):
    mdp_outputs = ["LargeMdpRwTraps50/output.csv","LargeMdpRwTrapsNeg50/output.csv","SmallMdpRwTraps/output.csv","SmallMdpRwTrapsNeg/output.csv"]
    outputs = []
    for out in mdp_outputs:
        outputs.append(pd.read_csv(rootfolder+"/"+out))
    all_data = pd.concat(outputs).apply(SwapColValues,axis = 1)
    all_data.columns = [c for c in map(RenameCols, all_data.columns)]
    all_data.loc[:,(all_data.columns != 'avg_rewards') & (all_data.columns != 'cum_rewards') & (all_data.columns != 'ran_to_completion')].to_csv(rootfolder+"/all_outputs_min.csv")
    all_data.to_csv(rootfolder+"/all_outputs.csv")

def ComparePolicies(p1,p2, adjustForGoalState = True):
    vi = pd.read_csv(p1,header = None).sort_values([0,1])[[3,4,5,6]].as_matrix()
    pi = pd.read_csv(p2,header = None).sort_values([0,1])[[3,4,5,6]].as_matrix()
    if(adjustForGoalState):
        vi = vi[0:-1,:]
        pi = pi[0:-1,:]
    #diff = np.where(vi>0,1,0)-np.where(pi>0,1,0)
    diff = np.max(np.abs(vi-pi),axis=1)
    non_zero = np.count_nonzero(diff)
    return non_zero

def ComputeDiffInOptimalPolicyForMdp(mdp_folder, gammas, vi_file,pi_file, adjustForGoalState ,outputFile):
    diffs = {}
    for gamma in gammas:
        v_file = mdp_folder +"/"+str(gamma)+"/"+vi_file
        p_file = mdp_folder +"/"+str(gamma)+"/"+pi_file
        vi = pd.read_csv(v_file,header = None).sort_values([0,1])[[3,4,5,6]].as_matrix()
        pi = pd.read_csv(p_file,header = None).sort_values([0,1])[[3,4,5,6]].as_matrix()
        if(adjustForGoalState):
            vi = vi[0:-1,:]
            pi = pi[0:-1,:]
        #diff = np.where(vi>0,1,0)-np.where(pi>0,1,0)
        diff = np.max(np.abs(vi-pi),axis=1)
        non_zero = np.count_nonzero(diff)
        diffs[gamma] = non_zero
    lines = []
    if(outputFile):
        for key in gammas:
            line = "{0},{1}".format(key, diffs[key])
            lines.append(line)
        u.WriteTextArrayToFile(mdp_folder+"/policy_diff.csv",lines)
    return diffs

def ComputeDiffInOptimalPolicyForMdp(mdp_folder, gammas, vi_file,pi_file, adjustForGoalState ,outputFile):
    diffs = {}
    for gamma in gammas:
        v_file = mdp_folder +"/"+str(gamma)+"/"+vi_file
        p_file = mdp_folder +"/"+str(gamma)+"/"+pi_file
        vi = pd.read_csv(v_file,header = None).sort_values([0,1])[[3,4,5,6]].as_matrix()
        pi = pd.read_csv(p_file,header = None).sort_values([0,1])[[3,4,5,6]].as_matrix()
        if(adjustForGoalState):
            vi = vi[0:-1,:]
            pi = pi[0:-1,:]
        #diff = np.where(vi>0,1,0)-np.where(pi>0,1,0)
        diff = np.max(np.abs(vi-pi),axis=1)
        non_zero = np.count_nonzero(diff)
        diffs[gamma] = non_zero
    lines = []
    if(outputFile):
        for key in gammas:
            line = "{0},{1}".format(key, diffs[key])
            lines.append(line)
        u.WriteTextArrayToFile(mdp_folder+"/policy_diff.csv",lines)
    return diffs

def ComputeDiffInOptimalValueForMdp(mdp_folder, gammas, vi_file,pi_file, adjustForGoalState ,outputFile):
    diffs = {}
    for gamma in gammas:
        v_file = mdp_folder +"/"+str(gamma)+"/"+vi_file
        p_file = mdp_folder +"/"+str(gamma)+"/"+pi_file
        vi = pd.read_csv(v_file,header = None).sort_values([0,1])[2].values
        pi = pd.read_csv(p_file,header = None).sort_values([0,1])[2].values
        vi_sum = vi.sum()
        pi_sum = pi.sum()
        diff = vi_sum - pi_sum
        diff = vi-pi
        diff[np.abs(diff) < 0.01] = 0
        pos_ind = np.where(diff > 0)
        neg_ind = np.where(diff < 0)
        if((pos_ind[0].size > 0) & (neg_ind[0].size > 0) ):
            diffs[gamma] = np.NaN
            continue
        elif(pos_ind[0].size > 0):
            diffs[gamma] = 1
        elif(neg_ind[0].size > 0):
            diffs[gamma] = -1
        else:
            diffs[gamma] = 0
    lines = []
    if(outputFile):
        for key in gammas:
            line = "{0},{1}".format(key, diffs[key])
            lines.append(line)
        u.WriteTextArrayToFile(mdp_folder+"/value_diff.csv",lines)
    return diffs

def ComputeDiffInVI_PI_Q(rootFolder):
    output = rootFolder+"/pi_q_comparison.csv"
    pi_file = rootFolder+"/LargeMdpRwTraps50/0.99/pi_10000.policy.csv"
    q_file = rootFolder+"/LargeMdpRwTraps50/0.99/ql_10000_alpha=1.0_po=boltzmann_p=100.0.policy.csv"
    diff = ComparePolicies(pi_file,q_file)
    lines = []
    lines.append("LargeMdp,"+str(diff))
    pi_file = rootFolder+"/SmallMdpRwTraps/0.99/pi_10000.policy.csv"
    q_file = rootFolder+"/SmallMdpRwTraps/0.99/ql_1000_alpha=1.0_po=greedyepsilon_p=0.1.policy.csv"
    diff = ComparePolicies(pi_file,q_file)
    lines.append("SmallMdp,"+str(diff))
    u.WriteTextArrayToFile(output,lines)

def ComputeDiffInVI_PI_Q_1(rootFolder):
    output = rootFolder+"/pi_q_comparison.csv"
    rootFolder1 = r"C:/Users/shkhandu/OneDrive/Gatech/Courses/ML/Assignment4/OutputNew1"
    pi_file = rootFolder1+"/LargeMdpRwTraps50/0.99/pi_10000.policy.csv"
    q_file = rootFolder+"/LargeMdpRwTraps50/0.99/ql_10000_alpha=1.0_po=boltzmann_p=100.0.policy.csv"
    diff = ComparePolicies(pi_file,q_file)
    lines = []
    lines.append("LargeMdp,"+str(diff))
    pi_file = rootFolder1+"/SmallMdpRwTraps/0.99/pi_10000.policy.csv"
    q_file = rootFolder+"/SmallMdpRwTraps/0.99/ql_5000_alpha=0.1_po=greedyepsilon_p=0.05.policy.csv"
    diff = ComparePolicies(pi_file,q_file)
    lines.append("SmallMdp,"+str(diff))
    u.WriteTextArrayToFile(output,lines)

def PlotPiViConvergenceForSmallAndLargeMdp(outputfolder,datafile,gamma):
    data = pd.read_csv(datafile)
    decorations = {1:'g',10:'k',10000:'r'}
    pi_sweeps = [1,10,10000]
    ser = []
    ser1 = []
    vi_added = False
    for sweep in pi_sweeps:
        data_vi = u.FilterRows(data,lambda x : (x['mdp'] == 'LargeMdpRwTraps50') & (x['solver'] == 'vi') & (x['gamma'] == gamma))
        data_pi = u.FilterRows(data,lambda x : (x['mdp'] == 'LargeMdpRwTraps50') & (x['solver'] == 'pi') & (x['gamma'] == gamma) & (x['maxSweepsPerIteration'] == sweep))
        assert(len(data_vi) == 1)
        assert(len(data_pi) == 1)

        data_vi_qchange = np.array([float(s) for s in data_vi.iloc[0]['cum_rewards'].split(';')])
        data_vi_value = np.array([float(s) for s in data_vi.iloc[0]['ran_to_completion'].split(';')])
        data_pi_qchange = np.array([float(s) for s in data_pi.iloc[0]['cum_rewards'].split(';')])
        data_pi_value = np.array([float(s) for s in data_pi.iloc[0]['ran_to_completion'].split(';')])
        if(vi_added == False):
            s_vi = u.YSeries(data_vi_qchange,xvalues=np.arange(len(data_vi_qchange)) + 1,line_color='b',plot_legend_label='VI')
            ser.append(s_vi)
        s_pi = u.YSeries(data_pi_qchange,xvalues=np.arange(len(data_pi_qchange)) + 1,line_color=decorations[sweep],plot_legend_label='PI_'+str(sweep))
        ser.append(s_pi)
        
        if(vi_added == False):
            s_vi = u.YSeries(data_vi_value,xvalues=np.arange(len(data_vi_value)) + 1,line_color='b',plot_legend_label='VI')
            ser1.append(s_vi)
        s_pi = u.YSeries(data_pi_value,xvalues=np.arange(len(data_pi_value)) + 1,line_color=decorations[sweep],plot_legend_label='PI_'+str(sweep))
        ser1.append(s_pi)
        vi_added = True

    outputfile = u.PreparePath(outputfolder+"/plots/large_qchange_gamma="+str(gamma)+".png")
    u.SaveDataPlotWithLegends(ser,filename=outputfile,x_axis_name="iterations",y1_axis_name="Max change in state value")
    outputfile = u.PreparePath(outputfolder+"/plots/large_value_gamma="+str(gamma)+".png")
    u.SaveDataPlotWithLegends(ser1,filename=outputfile,x_axis_name="iterations",y1_axis_name="Total value accross states")


    ser = []
    ser1 = []
    vi_added = False
    for sweep in pi_sweeps:
        data_vi = u.FilterRows(data,lambda x : (x['mdp'] == 'SmallMdpRwTraps') & (x['solver'] == 'vi') & (x['gamma'] == gamma))
        data_pi = u.FilterRows(data,lambda x : (x['mdp'] == 'SmallMdpRwTraps') & (x['solver'] == 'pi') & (x['gamma'] == gamma) & (x['maxSweepsPerIteration'] == sweep))
        assert(len(data_vi) == 1)
        assert(len(data_pi) == 1)

        data_vi_qchange = np.array([float(s) for s in data_vi.iloc[0]['cum_rewards'].split(';')])
        data_vi_value = np.array([float(s) for s in data_vi.iloc[0]['ran_to_completion'].split(';')])
        data_pi_qchange = np.array([float(s) for s in data_pi.iloc[0]['cum_rewards'].split(';')])
        data_pi_value = np.array([float(s) for s in data_pi.iloc[0]['ran_to_completion'].split(';')])
        if(vi_added == False):
            s_vi = u.YSeries(data_vi_qchange,xvalues=np.arange(len(data_vi_qchange)) + 1,line_color='b',plot_legend_label='VI')
            ser.append(s_vi)
        s_pi = u.YSeries(data_pi_qchange,xvalues=np.arange(len(data_pi_qchange)) + 1,line_color=decorations[sweep],plot_legend_label='PI_'+str(sweep))
        ser.append(s_pi)
        
        if(vi_added == False):
            s_vi = u.YSeries(data_vi_value,xvalues=np.arange(len(data_vi_value)) + 1,line_color='b',plot_legend_label='VI')
            ser1.append(s_vi)
        s_pi = u.YSeries(data_pi_value,xvalues=np.arange(len(data_pi_value)) + 1,line_color=decorations[sweep],plot_legend_label='PI_'+str(sweep))
        ser1.append(s_pi)
        vi_added = True

    outputfile = u.PreparePath(outputfolder+"/plots/small_qchange_gamma="+str(gamma)+".png")
    u.SaveDataPlotWithLegends(ser,filename=outputfile,x_axis_name="iterations",y1_axis_name="Max change in state value")
    outputfile = u.PreparePath(outputfolder+"/plots/small_value_gamma="+str(gamma)+".png")
    u.SaveDataPlotWithLegends(ser1,filename=outputfile,x_axis_name="iterations",y1_axis_name="Total value accross states")

def Evaluate(rootFolder):
    ConcatenateOutputFiles(rootFolder)
    ComputeDiffInOptimalPolicyForMdp(rootFolder+"/"+"LargeMdpRwTraps50",[0.99,0.95,0.90,0.80],"vi.policy.csv","pi_10000.policy.csv",True,True)
    ComputeDiffInOptimalPolicyForMdp(rootFolder+"/"+"SmallMdpRwTraps",[0.99,0.95,0.90,0.80],"vi.policy.csv","pi_10000.policy.csv",True,True)

    PlotPiViConvergenceForSmallAndLargeMdp(rootFolder,rootFolder+"/all_outputs.csv",0.99)
    PlotPiViConvergenceForSmallAndLargeMdp(rootFolder,rootFolder+"/all_outputs.csv",0.95)
    PlotPiViConvergenceForSmallAndLargeMdp(rootFolder,rootFolder+"/all_outputs.csv",0.9)
    PlotPiViConvergenceForSmallAndLargeMdp(rootFolder,rootFolder+"/all_outputs.csv",0.8)

    #per iteration metrics for large mdp
    mdpFolder = rootFolder+"/"+"LargeMdpRwTraps50"
    outputfile = mdpFolder + "/output.csv"
    data = pd.read_csv(outputfile)
    PlotAvgRewardsPerEpisode(data,10000,50,mdpFolder+"/avg_reward.png","Avg Reward","ar")
    PlotAvgRewardsPerEpisode(data,10000,50,mdpFolder+"/completion.png","Reached Goal (1/0)","goal")

    #per iteration metrics for small mdp
    mdpFolder = rootFolder+"/"+"SmallMdpRwTraps"
    outputfile = mdpFolder + "/output.csv"
    data = pd.read_csv(outputfile)
    PlotAvgRewardsPerEpisode(data,10000,20,mdpFolder+"/avg_reward.png","Avg Reward","ar",3000)
    PlotAvgRewardsPerEpisode(data,10000,20,mdpFolder+"/completion.png","Reached Goal (1/0)","goal",3000)

    ComputeDiffInVI_PI_Q(rootFolder)


def main(rootFolder = r"C:/Users/shkhandu/OneDrive/Gatech/Courses/ML/Assignment4/OutputNew4"):
    Evaluate(rootFolder)

def PlotAvgRewardsPerEpisode(data, totalpoints,points_to_sample, outputfile,y_axis_name,key_to_plot, max_points = 100000):
    """
    cr : cum_rewards
    ar : avg_rewards
    len : episode_len
    goal : reached_goal
    """
    data_to_plot = u.FilterRows(data,lambda x : (x['solver'] == 'q') & (x['gamma'] == 0.99) & (x['alpha'] == 1) & (x['maxInnerVi'] == totalpoints))
    x_to_take = np.arange(totalpoints)
    x_to_take = x_to_take[x_to_take % points_to_sample == 0]
    x_to_take = x_to_take[x_to_take < max_points]
    ser = data_to_plot.apply(lambda x : GetQRewardSeriesToPlot(x,x_to_take,key_to_plot),axis=1)
    u.SaveDataPlotWithLegends(ser.values,filename = outputfile,x_axis_name="episodes",y1_axis_name=y_axis_name)
   

def GetQRewardSeriesToPlot(x, xvalues,key_to_plot):
    markers = u.GetAllMarkers()
    colors = u.GetColorCombinations()
    if(x['exp_strategy'] == 'boltzmann'):
        line_color = 'b'
        if(x['param'] == 0.1):
            marker = markers[0]
        elif(x['param'] == 1):
            marker = markers[1]
        elif(x['param'] == 10):
            marker = markers[2]
        elif(x['param'] == 100):
            marker = markers[3]
    else:
        line_color = 'r'
        if(x['param'] == 0.01):
            marker = markers[0]
        elif(x['param'] == 0.05):
            marker = markers[1]
        elif(x['param'] == 0.1):
            marker = markers[2]
        elif(x['param'] == 0.2):
            marker = markers[3]

    cum_rewards = np.array([float(a) for a in x['cum_rewards'].split(';')])
    avg_rewards = np.array([float(a) for a in x['avg_rewards'].split(';')])
    completion = np.array([float(a) for a in x['ran_to_completion'].split(';')])
    episode_len = cum_rewards/avg_rewards
    xvalues = xvalues[xvalues < cum_rewards.size]
    if(key_to_plot == "ar"):
        y = avg_rewards[xvalues]
    if(key_to_plot == "cr"):
        y = cum_rewards[xvalues]
    if(key_to_plot == "goal"):
        y = completion[xvalues]
    if(key_to_plot == "len"):
        y = episode_len[xvalues]

    xdata = np.arange(len(y)) + 1
    ser = u.YSeries(y,xvalues=xvalues,points_marker = marker,plot_legend_label=x['exp_strategy'] + "-" + str(x['param']),line_color = line_color)
    return ser

if __name__ == "__main__":
    main()