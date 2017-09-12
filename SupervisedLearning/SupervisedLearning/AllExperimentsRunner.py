import Svm
import NeuralNetwork as nn
import KnnClassifier as knn
import Adaboost as ada
import DecisionTreesWithCV as dt
import ExperimentsAnalysis as ea

def GenerateResultsForDecisionTrees(root):
    dt.RunDecisionTreesOnCreditScreeningDataset(root+r'/CreditScreeningDataset')
    dt.RunDecisionTreesOnVowelRecognitionDataset(root+r'/LetterRecognition')

    ea.DecisionTreeAnalysis(
        root + r'/CreditScreeningDataset/Plots/dt', 
        r'dt.creditscreening.dt',
        root + r"/CreditScreeningDataset/eval_agg.credit.dt_3_0.csv",
        plt_title = "Decision Tree Performance")
    ea.DecisionTreeAnalysis(
        root + r'/LetterRecognition/Plots/dt', 
        r'dt.vowelrecognition.dt',
        root + r"/LetterRecognition/eval_agg.vowel.dt_3_0.csv",
        plt_title = "Decision Tree Performance")

    dataset_files = []
    prune_vals = [True,False]
    for prune in prune_vals:
        ea.PlotCrossValidationCurvesForWeka(
            root + '/CreditScreeningDataset/eval_agg.credit.dt_3_cv.csv',
            'inst',
            'f',
            root + r'/CreditScreeningDataset/Plots/dt/cv.dt.i-0_t-80_T-20.prune-{0}.png'.format(prune),
            "Decision Tree Model Complexity Curve (Pruning : {0})".format(prune),
            "Min instances per leaf",
            'F-Measure',
            lambda x : (x['prune']==prune) and (x['istrain'] == 0))
        ea.PlotCrossValidationCurvesForWeka(
            root + '/LetterRecognition/eval_agg.vowel.dt_3_cv.csv',
            'inst',
            'f',
            root + r'/LetterRecognition/Plots/dt/cv.dt.i-0_t-80_T-20.prune-{0}.png'.format(prune),
            "Decision Tree Model Complexity Curve (Pruning : {0})".format(prune),
            "Min instances per leaf",
            'F-Measure',
            lambda x : (x['prune']==prune) and (x['istrain'] == 0))

def GenerateResultsForAdaboost(root):
    ada.RunAdaBoostOnCreditScreeningDataset(root+r'/CreditScreeningDataset')
    ada.RunAdaBoostOnVowelRecognitionDataset(root+r'/LetterRecognition')

    ea.DecisionTreeAnalysis(
        root + r'/CreditScreeningDataset/Plots/ada', 
        r'dt.creditscreening.ada',
        root + r"/CreditScreeningDataset/eval_agg.credit.ada_3_0.csv",
        plt_title = "Adaboost Performance")
    ea.DecisionTreeAnalysis(
        root + r'/LetterRecognition/Plots/ada', 
        r'dt.vowelrecognition.ada',
        root + r"/LetterRecognition/eval_agg.vowel.ada_3_0.csv",
        plt_title = "Adaboost Performance")

    dataset_files = []
    prune_vals = [True,False]
    for prune in prune_vals:
        ea.PlotCrossValidationCurvesForWeka(
            root + '/CreditScreeningDataset/eval_agg.credit.ada_3_cv.csv',
            'iter',
            'f',
            root + r'/CreditScreeningDataset/Plots/ada/cv.ada.i-0_t-80_T-20.prune-{0}.png'.format(prune),
            "Adaboost Model Complexity Curve (Pruning : {0})".format(prune),
            "Num of weak learners",
            'F-Measure',
            lambda x : (x['prune']==prune) and (x['istrain'] == 0))
        ea.PlotCrossValidationCurvesForWeka(
            root + '/LetterRecognition/eval_agg.vowel.ada_3_cv.csv',
            'iter',
            'f',
            root + r'/LetterRecognition/Plots/ada/cv.ada.i-0_t-80_T-20.prune-{0}.png'.format(prune),
            "Adaboost Model Complexity Curve (Pruning : {0})".format(prune),
            "Num of weak learners",
            'F-Measure',
            lambda x : (x['prune']==prune) and (x['istrain'] == 0))

def GenerateResultsForSvm(root):
    Svm.RunSvmClassifierOnCreditScreeningDataset(root + r"/CreditScreeningDataset")
    Svm.RunSvmClassifierOnVowelRecognitionDataset(root + r"/LetterRecognition")
    ea.PlotCrossValidationCurvesForSvm(root)
    ea.SvmAnalysis(
        root + r'/LetterRecognition/Plots/svm', 
        r'dt.vowelrecognition.svm',
        root + r"/LetterRecognition/eval_agg.vowel.svm_3_0.csv",
        None)

    ea.SvmAnalysis(
        root + r'/CreditScreeningDataset/Plots/svm', 
        r'dt.creditscreening.svm',
        root + r"/CreditScreeningDataset/eval_agg.credit.svm_3_0.csv",
        None)

def GenerateResultsForNNets(root):
    nn.RunNeuralNetsOnCreditScreeningDataset(root + r"/CreditScreeningDataset")
    nn.RunNeuralNetsOnVowelRecognitionDataset(root + r"/LetterRecognition")
    ea.PlotCrossValidationCurvesForNNets(root)
    ea.NNetAnalysis(
        root + r'/CreditScreeningDataset/Plots/nnets',
        'dt.creditscreening',
        root + r'/CreditScreeningDataset/eval_agg.credit.nnet_3_0.csv',
        0)

    ea.NNetAnalysis(
        root + r'/LetterRecognition/Plots/nnets',
        'dt.vowelrecognition',
        root + r'/LetterRecognition/eval_agg.vowel.nnet_3_0.csv',
        0)

def GenerateResultsForKnn(root):
    knn.RunKnnClassifierOnCreditScreeningDataset(root+r'/CreditScreeningDataset')
    knn.RunKnnClassifierOnVowelRecognitionDataset(root+r'/LetterRecognition')
    ea.PlotCrossValidationCurvesForKnn(root)
    ea.KnnAnalysisOptK(root + r'/CreditScreeningDataset/Plots/knn', r'dt.creditscreening',
                root + r"/CreditScreeningDataset/eval_agg.credit.knn_3_0.csv")
    ea.KnnAnalysisOptK(root + r'/LetterRecognition/Plots/knn', r'dt.vowelrecognition',
                root + r"/LetterRecognition/eval_agg.vowel.knn_3_0.csv")


def main():
    root = r'C:/Users/shwet/OneDrive/Gatech/Courses/ML/DataSets'
    GenerateResultsForDecisionTrees(root)
    # GenerateResultsForAdaboost(root)
    # GenerateResultsForKnn(root)
    # GenerateResultsForNNets(root)
    # GenerateResultsForSvm(root)

if __name__ == '__main__':
    main()