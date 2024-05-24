import matplotlib.pyplot as plt
from pomegranate import *

# Given failure probabilities for events F1 to F13
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

# Create the nodes for the intermediate events
F2 = Node(NaiveBayes([F5, F9, F11]), name="F2")
F3 = Node(NaiveBayes([F6, F12, F13]), name="F3")
F6 = Node(NaiveBayes([F7, F8]), name="F6")
F9 = Node(NaiveBayes([F9, F10]), name="F9") #add F9 distribution

# Define F1, F4, and other intermediate events (assuming OR gates for simplicity)
F1 = Node(NaiveBayes([F2, F3, F4]), name="F1")
F4 = DiscreteDistribution({'up': 1 - failure_probabilities['F4'], 'down': failure_probabilities['F4']})

# Create the Bayesian network
model = BayesianNetwork("Reliability Model")
model.add_states(F1, F2, F3, F4, F5, F6, F7, F8, F9, F10, F11, F12, F13)
model.add_edge(F2, F1)
model.add_edge(F3, F1)
model.add_edge(F4, F1)
model.add_edge(F5, F2)
model.add_edge(F9, F2)
model.add_edge(F11, F2)
model.add_edge(F6, F3)
model.add_edge(F12, F3)
model.add_edge(F13, F3)
model.add_edge(F7, F6)
model.add_edge(F8, F6)
model.add_edge(F10, F9)

model.bake()

# Perform inference to calculate probabilities
predictions = model.predict_proba({})
results = {state.name: state.parameters[0]['down'] for state in model.states}

# Print the results
print("Predicted failure probabilities after 5 years:")
for event, prob in results.items():
    print(f"{event}: {prob:.4f}")

# Create a DataFrame for visualization
import pandas as pd

data = {
    'Event': ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'F11', 'F12', 'F13'],
    'Failure Probability': [results['F1'], results['F2'], results['F3'], results['F4'], results['F5'], results['F6'],
                            results['F7'], results['F8'], results['F9'], results['F10'], results['F11'],
                            results['F12'], results['F13']],
    'Reliability': [1 - results['F1'], 1 - results['F2'], 1 - results['F3'], 1 - results['F4'], 1 - results['F5'],
                    1 - results['F6'], 1 - results['F7'], 1 - results['F8'], 1 - results['F9'], 1 - results['F10'],
                    1 - results['F11'], 1 - results['F12'], 1 - results['F13']],
    'Unavailability': [results['F1'], results['F2'], results['F3'], results['F4'], results['F5'], results['F6'],
                       results['F7'], results['F8'], results['F9'], results['F10'], results['F11'], results['F12'],
                       results['F13']],
    'Availability': [1 - results['F1'], 1 - results['F2'], 1 - results['F3'], 1 - results['F4'], 1 - results['F5'],
                     1 - results['F6'], 1 - results['F7'], 1 - results['F8'], 1 - results['F9'], 1 - results['F10'],
                     1 - results['F11'], 1 - results['F12'], 1 - results['F13']]
}

df = pd.DataFrame(data)

# Plot the results
fig, axes = plt.subplots(3, 1, figsize=(10, 18))

# Plot Reliability
axes[0].bar(df['Event'], df['Reliability'], color='green')
axes[0].set_xlabel('Events')
axes[0].set_ylabel('Reliability')
axes[0].set_title('Reliability of Each Event')
axes[0].grid(True)

# Plot Unavailability
axes[1].bar(df['Event'], df['Unavailability'], color='red')
axes[1].set_xlabel('Events')
axes[1].set_ylabel('Unavailability')
axes[1].set_title('Unavailability of Each Event')
axes[1].grid(True)

# Plot Availability
axes[2].bar(df['Event'], df['Availability'], color='blue')
axes[2].set_xlabel('Events')
axes[2].set_ylabel('Availability')
axes[2].set_title('Availability of Each Event')
axes[2].grid(True)

plt.tight_layout()
plt.show()
