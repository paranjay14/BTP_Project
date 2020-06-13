import argparse
import sys
import os


def check_positive_int(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("%s is an invalid positive int value" % value)
    return ivalue

def check_non_negative_int(value):
    ivalue = int(value)
    if ivalue < 0:
        raise argparse.ArgumentTypeError("%s is an invalid non negative int value" % value)
    return ivalue

def getOptions(args=sys.argv[1:]):
    parser = argparse.ArgumentParser(description="Parses command.")
    parser.add_argument("--Train", dest='trainFlg', action='store_true', help="Set Train Flag", default=False)
    parser.add_argument("--noTrain", dest='trainFlg', action='store_false', help="Unset Train Flag")
    parser.add_argument("--Test", dest='testFlg', action='store_true', help="Set Test Flag", default=True)
    parser.add_argument("--noTest", dest='testFlg', action='store_false', help="Unset Test Flag")
    parser.add_argument("-md", "--maxDepth", type=check_non_negative_int, help="Max Depth (0-indexed)", default=6)
    parser.add_argument("-cDir", "--ckptDir", type=str, help="Name of Ckpt Directory to load", default="ckpt")
    parser.add_argument("-cOut", "--cnnOut", type=check_positive_int, help="No. of Filters/Out Channels in CNN", default=32)
    parser.add_argument("-mFC1", "--mlpFC1", type=check_positive_int, help="No. of Features in MLP FC Layer 1", default=512)    # try changing it, e.g. to 516
    parser.add_argument("-mFC2", "--mlpFC2", type=check_positive_int, help="No. of Features in MLP FC Layer 2", default=32)
    parser.add_argument("-cLR", "--cnnLR", type=float, help="CNN Learning Rate", default=0.001)
    parser.add_argument("-mLR", "--mlpLR", type=float, help="MLP Learning Rate", default=0.001)
    parser.add_argument("-cE", "--cnnEpochs", type=check_positive_int, help="No. of Epochs in CNN", default=71)
    parser.add_argument("-mE", "--mlpEpochs", type=check_positive_int, help="No. of Epochs in MLP", default=60)
    parser.add_argument("-csE", "--cnnSchEpochs", type=check_positive_int, help="CNN Scheduler Epochs", default=10)
    parser.add_argument("-msE", "--mlpSchEpochs", type=check_positive_int, help="MLP Scheduler Epochs", default=10)
    parser.add_argument("-csF", "--cnnSchFactor", type=float, help="CNN Scheduler Factor", default=0.37)
    parser.add_argument("-msF", "--mlpSchFactor", type=float, help="MLP Scheduler Factor", default=0.4)
    parser.add_argument("-cB", "--cnnBatches", type=check_positive_int, help="No. of Batches in CNN", default=100)
    parser.add_argument("-mB", "--mlpBatches", type=check_positive_int, help="No. of Batches in MLP", default=100)
    parser.add_argument("-case", "--caseNum", type=int, help="ID of case no. to select splitting decision among 1(EXP), 2(FD), 3(KM)", choices=[1,2,3], default=1)
    parser.add_argument("-opt", "--optionNum", type=int, help="ID of option no. to select modelToTrain among 1(MLP), 2(DT), 3(RF)", choices=[1,2,3], default=1)
    parser.add_argument("-ens", "--ensemble", type=check_positive_int, help="no of Trees to keep in ensemble", default=1)
    parser.add_argument("--Prob", dest='probabilistic', action='store_true', help="Setting Boolean value for using Probabilistic method", default=False)
    parser.add_argument("--noProb", dest='probabilistic', action='store_false', help="Unsetting Boolean value for not using Probabilistic method")
    parser.add_argument("-v", "--verbose", type=int, help="Verbose mode", default=2)
    options = parser.parse_args(args)
    return options

options = getOptions(sys.argv[1:])

if not os.path.isdir(options.ckptDir+'/'):
        os.mkdir(options.ckptDir+'/')

if options.trainFlg == True:
    with open(options.ckptDir+r"/options.txt","w") as f:
        L = [ "options.testFlg:\t" + str(options.testFlg) + "\n","options.maxDepth:\t" + str(options.maxDepth) + "\n",
        "options.ckptDir:\t" + options.ckptDir + "\n","options.cnnOut:\t" + str(options.cnnOut) + "\n",
        "options.mlpFC1:\t" + str(options.mlpFC1) + "\n","options.mlpFC2:\t" + str(options.mlpFC2) + "\n",
        "options.cnnLR:\t" + str(options.cnnLR) + "\n", "options.mlpLR:\t" + str(options.mlpLR) + "\n",
        "options.cnnEpochs:\t" + str(options.cnnEpochs) + "\n", "options.mlpEpochs:\t" + str(options.mlpEpochs) + "\n",
        "options.cnnSchEpochs:\t" + str(options.cnnSchEpochs) + "\n", "options.mlpSchEpochs:\t" + str(options.mlpSchEpochs) + "\n",
        "options.cnnSchFactor:\t" + str(options.cnnSchFactor) + "\n", "options.mlpSchFactor:\t" + str(options.mlpSchFactor) + "\n",
        "options.cnnBatches:\t" + str(options.cnnBatches) + "\n", "options.mlpBatches:\t" + str(options.mlpBatches) + "\n",
        "options.caseNum:\t" + str(options.caseNum) + "\n", "options.optionNum:\t" + str(options.optionNum) + "\n",
        "options.ensemble:\t" + str(options.ensemble) + "\n", "options.probabilistic:\t" + str(options.probabilistic) + "\n",
        "options.verbose:\t" + str(options.verbose) + "\n" ]

        f.writelines(L)
