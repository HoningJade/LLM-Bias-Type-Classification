import matplotlib.pyplot as plt

def plot_loss():
    # Sample loss values for training, validation, and testing
    train_loss = [3.1757403120161993, 2.703520080399892, 2.2256485689254033]
    validation_loss = [0.649499, 0.644928, 0.621463]
    # test_loss = [0.5, 0.45, 0.4]

    epochs = [1, 2, 3]  # Replace with your actual epoch values

    # Plot the training loss
    plt.plot(epochs, train_loss, label='Training Loss', marker='o', linestyle='-')

    # Plot the validation loss
    plt.plot(epochs, validation_loss, label='Validation Loss', marker='o', linestyle='-')
    for i, loss in enumerate(validation_loss):
        plt.annotate(f'{loss:.3f}', (epochs[i], loss), textcoords="offset points", xytext=(0, 10), ha='center', va='bottom')

    # Plot the testing loss
    # plt.plot(epochs, test_loss, label='Testing Loss', marker='o', linestyle='-')

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Loss Function Over Three Epochs')
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.savefig("loss")
    plt.show()


def plot_acc():
    # Sample loss values for training, validation, and testing
    train_acc = [0.26666666666666666, 0.2857142857142857, 0.3333333333333333]
    validation_acc = [0.2303944615929734, 0.325915100135219, 0.39885409575675057]
    # test_loss = [0.5, 0.45, 0.4]

    epochs = [1, 2, 3]  # Replace with your actual epoch values

    # Plot the training loss
    plt.plot(epochs, train_acc, label='Training F1', marker='o', linestyle='-')

    # Plot the validation loss
    plt.plot(epochs, validation_acc, label='Validation F1', marker='o', linestyle='-')
    #for i, loss in enumerate(validation_acc):
    #    plt.annotate(f'{loss:.3f}', (epochs[i], loss), textcoords="offset points", xytext=(0, 10), ha='center',
    #                 va='bottom')

    # Plot the testing loss
    # plt.plot(epochs, test_loss, label='Testing Loss', marker='o', linestyle='-')

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Macro F1', fontsize=12)
    plt.title('Macro F1 Over Three Epochs')
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.savefig("F1")
    plt.show()


if __name__ == '__main__':
    plot_acc()