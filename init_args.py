import argparse

parser = argparse.ArgumentParser(description = 'Neuron demo')
parser.add_argument('-n', type=int, default = 1, help = 'number of neurons to create', dest = 'neuron_count')
parser.add_argument('-f', type=str, default ='sigmoid', help = 'activation function to use', dest = 'activation_function', choices = ['sigmoid', 'binarystep'])
parser.add_argument('-t', type=float, default = 0.5, help = 'threshold for classification', dest = 'threshold')
parser.add_argument('-r', type=float, default = [-1, 1], help = 'range for random weights', dest = 'range', nargs = 2, metavar = ('MIN', 'MAX'))
parser.add_argument('-p', type=int, default = None, help = 'precision for random weights', dest = 'precision')

args = parser.parse_args()