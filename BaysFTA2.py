import numpy as np
import pandas as pd
from pomegranate import *

# Define failure probabilities
failure_probabilities = {
    'F1': 0.3248,
    'F2': 0.0500,
    'F3': 0.03,
    'F4': 0.0700,
    'F5': 0.10,
    'F9': 0.08,
    'F11': 0.09,
    'F6': 0.04,
    'F12': 0.01,
    'F13': 0.03,
    'F7': 0.02,
    'F8': 0.06,
    'F10': 0.05,
}

# Define the basic events as DiscreteDistributions
F5 = DiscreteDistribution({'up': 1 - failure_probabilities['F5'], 'down': failure_probabilities['F5']})
F9 = DiscreteDistribution({'up': 1 - failure_probabilities['F9'], 'down': failure_probabilities['F9']})
F11 = DiscreteDistribution({'up': 1 - failure_probabilities['F11'], 'down': failure_probabilities['F11']})
F6 = DiscreteDistribution({'up': 1 - failure_probabilities['F6'], 'down': failure_probabilities['F6']})
F12 = DiscreteDistribution({'up': 1 - failure_probabilities['F12'], 'down': failure_probabilities['F12']})
F13 = DiscreteDistribution({'up': 1 - failure_probabilities['F13'], 'down': failure_probabilities['F13']})
F7 = DiscreteDistribution({'up': 1 - failure_probabilities['F7'], 'down': failure_probabilities['F7']})
F8 = DiscreteDistribution({'up': 1 - failure_probabilities['F8'], 'down': failure_probabilities['F8']})
F10 = DiscreteDistribution({'up': 1 - failure_probabilities['F10'], 'down': failure_probabilities['F10']})
F4 = DiscreteDistribution({'up': 1 - failure_probabilities['F4'], 'down': failure_probabilities['F4']})  # Define F4

# Define Conditional Probability Tables for intermediate events
F2 = ConditionalProbabilityTable([
    ['up', 'up', 'up', 'up', 0.99],
    ['up', 'up', 'up', 'down', 0.01],
    ['up', 'up', 'down', 'up', 0.05],
    ['up', 'up', 'down', 'down', 0.95],
    ['up', 'down', 'up', 'up', 0.06],
    ['up', 'down', 'up', 'down', 0.94],
    ['up', 'down', 'down', 'up', 0.10],
    ['up', 'down', 'down', 'down', 0.90],
    ['down', 'up', 'up', 'up', 0.04],
    ['down', 'up', 'up', 'down', 0.96],
    ['down', 'up', 'down', 'up', 0.08],
    ['down', 'up', 'down', 'down', 0.92],
    ['down', 'down', 'up', 'up', 0.09],
    ['down', 'down', 'up', 'down', 0.91],
    ['down', 'down', 'down', 'up', 0.15],
    ['down', 'down', 'down', 'down', 0.85]
], [F5, F9, F11])

F3 = ConditionalProbabilityTable([
    ['up', 'up', 'up', 'up', 0.98],
    ['up', 'up', 'up', 'down', 0.02],
    ['up', 'up', 'down', 'up', 0.03],
    ['up', 'up', 'down', 'down', 0.97],
    ['up', 'down', 'up', 'up', 0.04],
    ['up', 'down', 'up', 'down', 0.96],
    ['up', 'down', 'down', 'up', 0.06],
    ['up', 'down', 'down', 'down', 0.94],
    ['down', 'up', 'up', 'up', 0.05],
    ['down', 'up', 'up', 'down', 0.95],
    ['down', 'up', 'down', 'up', 0.07],
    ['down', 'up', 'down', 'down', 0.93],
    ['down', 'down', 'up', 'up', 0.09],
    ['down', 'down', 'up', 'down', 0.91],
    ['down', 'down', 'down', 'up', 0.10],
    ['down', 'down', 'down', 'down', 0.90]
], [F6, F12, F13])

F1 = ConditionalProbabilityTable([
    ['up', 'up', 'up', 'up', 0.99],
    ['up', 'up', 'up', 'down', 0.01],
    ['up', 'up', 'down', 'up', 0.05],
    ['up', 'up', 'down', 'down', 0.95],
    ['up', 'down', 'up', 'up', 0.06],
    ['up', 'down', 'up', 'down', 0.94],
    ['up', 'down', 'down', 'up', 0.10],
    ['up', 'down', 'down', 'down', 0.90],
    ['down', 'up', 'up', 'up', 0.04],
    ['down', 'up', 'up', 'down', 0.96],
    ['down', 'up', 'down', 'up', 0.08],
    ['down', 'up', 'down', 'down', 0.92],
    ['down', 'down', 'up', 'up', 0.09],
    ['down', 'down', 'up', 'down', 0.91],
    ['down', 'down', 'down', 'up', 0.15],
    ['down', 'down', 'down', 'down', 0.85]
], [F2, F3, F4])

# Build the Bayesian Network
s1 = State(F1, name="F1")
s2 = State(F2, name="F2")
s3 = State(F3, name="F3")
s4 = State(F4, name="F4")
s5 = State(F5, name="F5")
s6 = State(F6, name="F6")
s7 = State(F7, name="F7")
s8 = State(F8, name="F8")
s9 = State(F9, name="F9")
s10 = State(F10, name="F10")
s11 = State(F11, name="F11")
s12 = State(F12, name="F12")
s13 = State(F13, name="F13")

model = BayesianNetwork("Fault Tree Analysis")
model.add_states(s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13)
model.add_edge(s5, s2)
model.add_edge(s9, s2)
model.add_edge(s11, s2)
model.add_edge(s6, s3)
model.add_edge(s12, s3)
model.add_edge(s13, s3)
model.add_edge(s2, s1)
model.add_edge(s3, s1)
model.add_edge(s4, s1)

model.bake()

# Define functions to calculate reliability and failure probability over time
def reliability_over_time(failure_probability, years):
    return (1 - failure_probability) ** years

def failure_probability_over_time(failure_probability, years):
    return 1 - reliability_over_time(failure_probability, years)

# Calculate reliability and failure probability over 5 years for each event

events = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'F11', 'F12', 'F13']
reliabilities = {event: reliability_over_time(failure_probabilities[event], 5) for event in events}
failure_probs = {event: failure_probability_over_time(failure_probabilities[event], 5) for event in events}

# Print results
print("Reliabilities over 1 years:")
print(reliabilities)
print("\nFailure probabilities over 5 years:")
print(failure_probs)

# Visualization
import matplotlib.pyplot as plt

# Reliability plot
plt.figure(figsize=(12, 6))
plt.bar(reliabilities.keys(), reliabilities.values(), color='green')
plt.xlabel('Events')
plt.ylabel('Reliability over 5 years')
plt.title('Reliability of Each Event over 5 Years')
plt.xticks(rotation=45)
plt.grid(True)
plt.savefig("Reliability_for_5years.pdf")

# Failure probability plot
plt.figure(figsize=(12, 6))
plt.bar(failure_probs.keys(), failure_probs.values(), color='red')
plt.xlabel('Events')
plt.ylabel('Failure Probability over 5 years')
plt.title('Failure Probability of Each Event over 5 Years')
plt.xticks(rotation=45)
plt.grid(True)
plt.savefig("Availability_for_5years.pdf")

