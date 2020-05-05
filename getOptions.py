import argparse
import sys

def getOptions(args=sys.argv[1:]):
    parser = argparse.ArgumentParser(description="Parses command.")
    parser.add_argument("-md", "--maxDepth", type=int, help="Max Depth (0-indexed)", default=0)
    parser.add_argument("-cLR", "--cnnLR", type=float, help="CNN Learning Rate", default=0.001)
    parser.add_argument("-mLR", "--mlpLR", type=float, help="MLP Learning Rate", default=0.001)
    parser.add_argument("-cE", "--cnnEpochs", type=int, help="No. of Epochs in CNN", default=100)
    parser.add_argument("-mE", "--mlpEpochs", type=int, help="No. of Epochs in MLP", default=60)
    parser.add_argument("-v", "--verbose",dest='verbose',action='store_true', help="Verbose mode.")
    options = parser.parse_args(args)
    return options

options = getOptions(sys.argv[1:])
