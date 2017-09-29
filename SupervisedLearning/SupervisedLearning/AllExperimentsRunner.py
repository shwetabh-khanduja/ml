import Svm
import NeuralNetwork as nn
import KnnClassifier as knn
import Adaboost as ada
import DecisionTreesWithCV as dt
import ExperimentsAnalysis as ea

def GenerateResultsForDecisionTrees(root, weka_jar_path):
    dt.RunDecisionTreesOnCreditScreeningDataset(root+r'/CreditScreeningDataset', weka_jar_path)
    dt.RunDecisionTreesOnVowelRecognitionDataset(root+r'/LetterRecognition', weka_jar_path)

    ea.DecisionTreeAnalysis(
        root + r'/CreditScreeningDataset/Plots/dt', 
        r'dt.creditscreening.dt',
        root + r"/CreditScreeningDataset/eval_agg.credit.dt_3_0.csv",
        plt_title = "Decision Tree Performance",
        y_axis_name = 'Accuracy')
    ea.DecisionTreeAnalysis(
        root + r'/LetterRecognition/Plots/dt', 
        r'dt.vowelrecognition.dt',
        root + r"/LetterRecognition/eval_agg.vowel.dt_3_0.csv",
        plt_title = "Decision Tree Performance")

    dataset_files = []
    prune_vals = [False]
    for prune in prune_vals:
        ea.PlotCrossValidationCurvesForWeka(
            root + '/CreditScreeningDataset/eval_agg.credit.dt_3_cv.csv',
            'inst',
            'm',
            root + r'/CreditScreeningDataset/Plots/dt/cv.dt.i-0_t-80_T-20.prune-{0}.png'.format(prune),
            "Decision Tree Model Complexity Curve (Pruning : {0})".format(prune),
            "Min instances per leaf",
            'Accuracy',
            lambda x : (x['prune']==prune) and (x['istrain'] == 1))
        ea.PlotCrossValidationCurvesForWeka(
            root + '/LetterRecognition/eval_agg.vowel.dt_3_cv.csv',
            'inst',
            'm',
            root + r'/LetterRecognition/Plots/dt/cv.dt.i-0_t-80_T-20.prune-{0}.png'.format(prune),
            "Decision Tree Model Complexity Curve (Pruning : {0})".format(prune),
            "Min instances per leaf",
            'F-Measure',
            lambda x : (x['prune']==prune) and (x['istrain'] == 1))

def GenerateResultsForAdaboost(root, weka_jar_path):
    ada.RunAdaBoostOnCreditScreeningDataset(root+r'/CreditScreeningDataset', weka_jar_path)
    ea.DecisionTreeAnalysis(
    root + r'/CreditScreeningDataset/Plots/ada', 
    r'dt.creditscreening.ada',
    root + r"/CreditScreeningDataset/eval_agg.credit.ada_3_0.csv",
    plt_title = "Adaboost Performance",
    y_axis_name = "Accuracy")

    ada.RunAdaBoostOnVowelRecognitionDataset(root+r'/LetterRecognition',weka_jar_path)
    ea.DecisionTreeAnalysis(
        root + r'/LetterRecognition/Plots/ada', 
        r'dt.vowelrecognition.ada',
        root + r"/LetterRecognition/eval_agg.vowel.ada_3_0.csv",
        plt_title = "Adaboost Performance")

    dataset_files = []
    prune_vals = [False]
    for prune in prune_vals:
        ea.PlotCrossValidationCurvesForWeka(
            root + '/CreditScreeningDataset/eval_agg.credit.ada_3_cv.csv',
            'iter',
            'm',
            root + r'/CreditScreeningDataset/Plots/ada/cv.ada.i-0_t-80_T-20.prune-{0}.png'.format(prune),
            "Adaboost Model Complexity Curve (Pruning : {0})".format(prune),
            "Num of weak learners",
            'Accuracy',
            lambda x : (x['prune']==prune) and (x['istrain'] == 1))
        ea.PlotCrossValidationCurvesForWeka(
            root + '/LetterRecognition/eval_agg.vowel.ada_3_cv.csv',
            'iter',
            'm',
            root + r'/LetterRecognition/Plots/ada/cv.ada.i-0_t-80_T-20.prune-{0}.png'.format(prune),
            "Adaboost Model Complexity Curve (Pruning : {0})".format(prune),
            "Num of weak learners",
            'F-Measure',
            lambda x : (x['prune']==prune) and (x['istrain'] == 1))

    iters = [2,5,10,15,20]
    ts = [20,30,40,50,60,70,80,90,100]
    for _ts in ts:
        ada.GetPerIterationMetricsForCreditScreeningDataset(root + r"/CreditScreeningDataset",weka_jar_path,"ts-{0}".format(_ts),iters)
    ea.PlotAdaboostPerIterationCurves(root + '/CreditScreeningDataset/eval_agg.credit.ada_3_ts-{0}.csv',lambda x : x['prune'] == False,root + '/CreditScreeningDataset/Plots/ada/credit.noprune.itercurves.png',[2,5,10,15,20],y_axis_name="Accuracy");
    ea.PlotAdaboostPerIterationCurves(root + '/CreditScreeningDataset/eval_agg.credit.ada_3_ts-{0}.csv',lambda x : x['prune'] == True,root + '/CreditScreeningDataset/Plots/ada/credit.prune.itercurves.png',[2,5,10,15,20],y_axis_name="Accuracy");
    iters = [2,10,20,30,50]
    ts = [20,30,40,50,60,70,80,90,100]
    for _ts in ts:
        ada.GetPerIterationMetricsForVowelRecognitionDataset(root + r"/LetterRecognition",weka_jar_path,"ts-{0}".format(_ts),iters)
    ea.PlotAdaboostPerIterationCurves(root + '/LetterRecognition/eval_agg.vowel.ada_3_ts-{0}.csv',lambda x : x['prune'] == False,root + '/LetterRecognition/Plots/ada/vowel.noprune.itercurves.png',[2,10,20,30,50]);
    ea.PlotAdaboostPerIterationCurves(root + '/LetterRecognition/eval_agg.vowel.ada_3_ts-{0}.csv',lambda x : x['prune'] == True,root + '/LetterRecognition/Plots/ada/vowel.prune.itercurves.png',[2,10,20,30,50]);

def GenerateResultsForSvm(root):
    Svm.RunSvmClassifierOnCreditScreeningDataset(root + r"/CreditScreeningDataset")
    ea.PlotCrossValidationCurvesForSvm(root,y_axis_name='Accuracy',roots=['CreditScreeningDataset'])
    Svm.RunSvmClassifierOnVowelRecognitionDataset(root + r"/LetterRecognition")
    ea.PlotCrossValidationCurvesForSvm(root,roots=['LetterRecognition'])

    ea.SvmAnalysis(
        root + r'/LetterRecognition/Plots/svm', 
        r'dt.vowelrecognition.svm',
        root + r"/LetterRecognition/eval_agg.vowel.svm_3_0.csv",
        None)

    ea.SvmAnalysis(
        root + r'/CreditScreeningDataset/Plots/svm', 
        r'dt.creditscreening.svm',
        root + r"/CreditScreeningDataset/eval_agg.credit.svm_3_0.csv",
        None,y_axis_name='Accuracy')
    ea.PlotSupportVectorsOverlap(root + "/LetterRecognition",r"Plots/svm/vowel.support_overlap.png","i-0_t-80_T-20/vowel.support_overlap.csv")
    ea.PlotSupportVectorsOverlap(root + "/CreditScreeningDataset",r"Plots/svm/credit.support_overlap.png","i-0_t-80_T-20/credit.support_overlap.csv")

def GenerateResultsForNNets(root):
    nn.RunNeuralNetsOnCreditScreeningDataset(root + r"/CreditScreeningDataset")
    nn.RunNeuralNetsOnVowelRecognitionDataset(root + r"/LetterRecognition")
    ea.PlotCrossValidationCurvesForNNets(root,y_axis_name='Accuracy')
    ea.NNetAnalysis(
        root + r'/CreditScreeningDataset/Plots/nnets',
        'dt.creditscreening',
        root + r'/CreditScreeningDataset/eval_agg.credit.nnet_3_0.csv',
        0,y_axis_name='Accuracy')

    ea.NNetAnalysis(
        root + r'/LetterRecognition/Plots/nnets',
        'dt.vowelrecognition',
        root + r'/LetterRecognition/eval_agg.vowel.nnet_3_0.csv',
        0)

    ea.PlotLossCurvesForNeuralNets(root + r'/CreditScreeningDataset/eval_agg.credit.nnet_3_0.csv', root + '/CreditScreeningDataset/Plots/nnets/credit.losscurve.earlystop-{0}.png')
    ea.PlotLossCurvesForNeuralNets(root + r'/LetterRecognition/eval_agg.vowel.nnet_3_0.csv', root + '/LetterRecognition/Plots/nnets/vowel.losscurve.earlystop-{0}.png')

def GenerateResultsForKnn(root):
    knn.RunKnnClassifierOnCreditScreeningDataset(root+r'/CreditScreeningDataset')
    knn.RunKnnClassifierOnVowelRecognitionDataset(root+r'/LetterRecognition')
    ea.PlotCrossValidationCurvesForKnn(root,y_axis_name='Accuracy')
    ea.KnnAnalysisOptK(root + r'/CreditScreeningDataset/Plots/knn', r'dt.creditscreening',
                root + r"/CreditScreeningDataset/eval_agg.credit.knn_3_0.csv",y_axis_name='Accuracy')
    ea.KnnAnalysisOptK(root + r'/LetterRecognition/Plots/knn', r'dt.vowelrecognition',
                root + r"/LetterRecognition/eval_agg.vowel.knn_3_0.csv")

def EvaluateBestModelsOnRealTestSet(root, weka):
    credit_model_info = {'svm':(r'i-0_t-80_ts-60/svm/cvresults/cvresults.model','60'),
                  'ada':r'i-0_t-80_ts-80/ada/prune-False_optiter-20/prune-False_optiter-20.model',
                  'dt':r'i-0_t-80_ts-60/dt/prune-True_optinst-8/prune-True_optinst-8.model',
                  'nnets':(r'i-0_t-80_ts-100/nnets/earlystop-False/earlystop-False.model','100'),
                  'knn':(r'i-0_t-80_ts-80/knn/weights-uniform_neighbors--1/weights-uniform_neighbors--1.model','80')}

    vowel_model_info = {'svm':(r'i-0_t-80_ts-100/svm/cvresults/cvresults.model','100'),
                  'ada':r'i-0_t-80_ts-90/ada/prune-False_optiter-50/prune-False_optiter-50.model',
                  'dt':r'i-0_t-80_ts-90/dt/prune-False_optinst-2/prune-False_optinst-2.model',
                  'nnets':(r'i-0_t-80_ts-90/nnets/earlystop-False/earlystop-False.model','90'),
                  'knn':(r'i-0_t-80_ts-100/knn/weights-uniform_neighbors--1/weights-uniform_neighbors--1.model','100')}

    ea.ComputePerformanceOnRealTestSet(credit_model_info,root+'/CreditScreeningDataset/i-0_t-80_T-20','credit.realtest.scores.csv',weka,'+',compute_accuracy = True)
    ea.ComputePerformanceOnRealTestSet(vowel_model_info,root+'/LetterRecognition/i-0_t-80_T-20','vowel.realtest.scores.csv',weka,'v')

def main():
    root = r'C:/Users/shkhandu/OneDrive/Gatech/Courses/ML/DataSets'
    weka_jar_path = "C:/Program Files/Weka-3-8/weka.jar"
    GenerateResultsForSvm(root)
    GenerateResultsForDecisionTrees(root, weka_jar_path)
    GenerateResultsForAdaboost(root, weka_jar_path)
    GenerateResultsForNNets(root)
    GenerateResultsForKnn(root)
    EvaluateBestModelsOnRealTestSet(root,weka_jar_path)

if __name__ == '__main__':
    main()