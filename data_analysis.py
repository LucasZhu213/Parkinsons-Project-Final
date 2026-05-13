import pandas as pd, matplotlib.pyplot as plt
lines = open("fastsurfer_9568.out").read().splitlines()
accuracies, recall_0, recall_1, recall_2 = [], [], [], []
for line in lines:
    if "Acc" in line:
        accuracies.append(float(line[line.index("Acc=")+4:line.index("Acc=")+9]))
    #if "Recall" in line:
    #    recall_0.append(float(line[line.index("Recall_0")+10:line.index("Recall_0")+15]))
    #    recall_1.append(float(line[line.index("Recall_1")+10:line.index("Recall_1")+15]))
    #    recall_2.append(float(line[line.index("Recall_2")+10:line.index("Recall_2")+15]))
        
data = {
    "Epoch": [i for i in range(1,101)],
    "Accuracy": accuracies,
}

data = pd.DataFrame(data, )
plt.plot("Epoch", "Accuracy", data=data)
#plt.plot("Epoch", "Recall_0", data=data)
#plt.plot("Epoch", "Recall_1", data=data)
#plt.plot("Epoch", "Recall_2", data=data)
plt.xlabel("Epoch")
plt.ylabel("Testing Accuracy")
plt.plot([i for i in range(1,101)], [0.5510 for i in range(100)])
plt.title("Testing accuracy over time for Graph NN")
plt.legend()
plt.show()
