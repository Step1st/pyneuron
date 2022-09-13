import pandas as pd
from neuron import *

# Load data from csv
data = pd.read_csv("demo_data.csv")
labels = data["t"].tolist()
data = data.drop("t", axis=1)
data = data.values.tolist()
weight_count = len(data[0])

neuron_count = int(input("Enter the number of neurons to generate: "))

random_weight_generator = RandomWeightGenerator(-12, 12, 5)

neuron = Neuron(weight_count, sigmoid, random_weight_generator)

total_neuron_count = 0

for i in range(neuron_count):
    while True:
        total_neuron_count += 1

        neuron.randomize()
        raw_outputs = [neuron.activate(obj) for obj in data]
        processed_outputs = round_outputs(raw_outputs, threshold=0.95) 

        if processed_outputs == labels:
            print(f"{i+1}. {neuron}")
            print(f"Output: {processed_outputs}")
            print(f"Raw output: {raw_outputs}")
            break

print(f"Went through {total_neuron_count} neurons")