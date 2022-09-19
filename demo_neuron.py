import pandas as pd
from neuron import *
from init_args import args

# Load data from csv
data = pd.read_csv("demo_data.csv")
labels = data["t"].tolist()
data = data.drop("t", axis=1)
data = data.values.tolist()
weight_count = len(data[0])

neuron_count = args.neuron_count
activation_function = sigmoid if args.activation_function == 'sigmoid' else binary_step
threshold = args.threshold
weight_generator = RandomWeightGenerator(args.range[0], args.range[1], args.precision)

neuron = Neuron(weight_count, activation_function, weight_generator)
total_neuron_count = 0

for i in range(neuron_count):
    while True:
        total_neuron_count += 1

        neuron.generate_weights()
        raw_outputs = [neuron.activate(obj) for obj in data]
        processed_outputs = round_outputs(raw_outputs, threshold) 

        if processed_outputs == labels:
            print(f"{i+1}. {neuron}")
            print(f"Output: {processed_outputs}")
            print(f"Raw output: {raw_outputs}")
            break

print(f"Went through {total_neuron_count} neurons")