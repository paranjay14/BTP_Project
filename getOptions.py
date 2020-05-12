import argparse
import sys
import os

def getOptions(args=sys.argv[1:]):
    parser = argparse.ArgumentParser(description="Parses command.")
    parser.add_argument("-tf", "--trainFlg", type=bool, help="Train Flag", default=False)
    parser.add_argument("-md", "--maxDepth", type=int, help="Max Depth (0-indexed)", default=6)
    parser.add_argument("-cLR", "--cnnLR", type=float, help="CNN Learning Rate", default=0.001)
    parser.add_argument("-mLR", "--mlpLR", type=float, help="MLP Learning Rate", default=0.001)
    parser.add_argument("-cE", "--cnnEpochs", type=int, help="No. of Epochs in CNN", default=71)
    parser.add_argument("-mE", "--mlpEpochs", type=int, help="No. of Epochs in MLP", default=60)
    parser.add_argument("-csE", "--cnnSchEpochs", type=int, help="CNN Scheduler Epochs", default=10)
    parser.add_argument("-csF", "--cnnSchFactor", type=float, help="CNN Scheduler Factor", default=0.37)
    parser.add_argument("-msE", "--mlpSchEpochs", type=int, help="MLP Scheduler Epochs", default=10)
    parser.add_argument("-msF", "--mlpSchFactor", type=float, help="MLP Scheduler Factor", default=0.4)
    parser.add_argument("-cB", "--cnnBatches", type=int, help="No. of Batches in CNN", default=200)
    parser.add_argument("-mB", "--mlpBatches", type=int, help="No. of Batches in MLP", default=200)
    parser.add_argument("-cOut", "--cnnOut", type=int, help="No. of Filters/Out Channels in CNN", default=32)
    parser.add_argument("-mFC1", "--mlpFC1", type=int, help="No. of Features in MLP FC Layer 1", default=516)
    parser.add_argument("-mFC2", "--mlpFC2", type=int, help="No. of Features in MLP FC Layer 2", default=32)
    parser.add_argument("-cDir", "--ckptDir", type=str, help="Name of Ckpt Directory to load", default="ckpt")
    parser.add_argument("-v", "--verbose",dest='verbose',action='store_true', help="Verbose mode.")
    options = parser.parse_args(args)
    return options

options = getOptions(sys.argv[1:])

if not os.path.isdir(options.ckptDir+'/'):
		os.mkdir(options.ckptDir+'/')

if options.trainFlg == True:
	with open(options.ckptDir+r"/options.txt","w") as f:
		L = ["options.ckptDir:\t" + options.ckptDir + "\n","options.maxDepth:\t" + str(options.maxDepth) + "\n","options.cnnLR:\t" + str(options.cnnLR) + "\n","options.mlpLR:\t" + str(options.mlpLR) + "\n","options.cnnEpochs:\t" + str(options.cnnEpochs) + "\n","options.mlpEpochs:\t" + str(options.mlpEpochs) + "\n","options.cnnOut:\t" + str(options.cnnOut) + "\n","options.mlpFC1:\t" + str(options.mlpFC1) + "\n","options.mlpFC2:\t" + str(options.mlpFC2) + "\n"]
		f.writelines(L)