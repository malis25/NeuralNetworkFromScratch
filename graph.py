import matplotlib.pyplot as plt

epochs = []
losses = []

with open("loss.csv") as f:
    for line in f:
        epoch, loss = line.strip().split(",")
        epochs.append(int(epoch))
        losses.append(float(loss))

plt.plot(epochs, losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.show()