import matplotlib.pyplot as plt


def acc(history, name):
    # summarize history for accuracy
    plt.plot(history["accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper left")
    plt.show()
    plt.savefig(f"./plots/{name}_accuracy.png")


def loss(history, name):
    # summarize history for loss
    plt.plot(history["loss"])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper left")
    plt.show()
    plt.savefig(f"./plots/{name}_loss.png")
