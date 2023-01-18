import matplotlib.pyplot as plt
import pandas as pd
from sys import argv

if len(argv) != 2:
    print("Usage: {} .log".format(argv[0]))
    exit()

log_file = argv[1]
data = pd.read_csv(log_file)
print("data size ", data.shape)

fig = plt.figure(figsize=(12, 4))

plt_labels = [("accuracy", "val_accuracy", 1), ('loss', 'val_loss', 6)]

for idx, (train, val, ylim) in enumerate(plt_labels):
    plt.subplot(1, 2, idx + 1)
    plt.plot(data[train])
    plt.plot(data[val])
    plt.title(train)
    plt.ylabel(train, fontsize="large")
    plt.xlabel("epoch", fontsize="large")
    plt.legend(["train", "val"], loc="best")
    plt.ylim(0, ylim)

plt.show()
