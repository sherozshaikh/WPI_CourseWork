import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
from sklearn.decomposition import PCA
from scipy.interpolate import griddata
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import regularizers

NUM_HIDDEN_LAYERS = 3
NUM_INPUT = 784
NUM_HIDDEN = 10
NUM_OUTPUT = 10

def initWeightsAndBiases ():
    Ws = []
    bs = []
    np.random.seed(0)
    W = 2*(np.random.random(size=(NUM_HIDDEN, NUM_INPUT))/NUM_INPUT**0.5) - 1./NUM_INPUT**0.5
    Ws.append(W)
    b = 0.01 * np.ones(NUM_HIDDEN)
    bs.append(b)
    for i in range(NUM_HIDDEN_LAYERS - 1):
        W = 2*(np.random.random(size=(NUM_HIDDEN, NUM_HIDDEN))/NUM_HIDDEN**0.5) - 1./NUM_HIDDEN**0.5
        Ws.append(W)
        b = 0.01 * np.ones(NUM_HIDDEN)
        bs.append(b)
    W = 2*(np.random.random(size=(NUM_OUTPUT, NUM_HIDDEN))/NUM_HIDDEN**0.5) - 1./NUM_HIDDEN**0.5
    Ws.append(W)
    b = 0.01 * np.ones(NUM_OUTPUT)
    bs.append(b)
    return np.hstack([ W.flatten() for W in Ws ] + [ b.flatten() for b in bs ])

def unpack (weightsAndBiases):
    Ws = []
    start = 0
    end = NUM_INPUT*NUM_HIDDEN
    W = weightsAndBiases[start:end]
    Ws.append(W)
    for i in range(NUM_HIDDEN_LAYERS - 1):
        start = end
        end = end + NUM_HIDDEN*NUM_HIDDEN
        W = weightsAndBiases[start:end]
        Ws.append(W)
    start = end
    end = end + NUM_HIDDEN*NUM_OUTPUT
    W = weightsAndBiases[start:end]
    Ws.append(W)
    Ws[0] = Ws[0].reshape(NUM_HIDDEN, NUM_INPUT)
    for i in range(1, NUM_HIDDEN_LAYERS):
        Ws[i] = Ws[i].reshape(NUM_HIDDEN, NUM_HIDDEN)
    Ws[-1] = Ws[-1].reshape(NUM_OUTPUT, NUM_HIDDEN)
    bs = []
    start = end
    end = end + NUM_HIDDEN
    b = weightsAndBiases[start:end]
    bs.append(b)
    for i in range(NUM_HIDDEN_LAYERS - 1):
        start = end
        end = end + NUM_HIDDEN
        b = weightsAndBiases[start:end]
        bs.append(b)
    start = end
    end = end + NUM_OUTPUT
    b = weightsAndBiases[start:end]
    bs.append(b)
    return Ws, bs

def forward_prop(x, y, weightsAndBiases):
    Ws, bs = unpack(weightsAndBiases)
    num_examples = x.shape[0]
    zs = []
    hs = []
    h = x.T
    hs.append(h)
    for i in range(NUM_HIDDEN_LAYERS):
        z = np.dot(Ws[i], h) + bs[i].reshape(-1, 1)
        zs.append(z)
        h = np.maximum(0, z)
        hs.append(h)
    z = np.dot(Ws[-1], h) + bs[-1].reshape(-1, 1)
    zs.append(z)
    yhat = np.exp(z) / np.sum(np.exp(z), axis=0)
    hs.append(yhat)
    loss = -np.sum(y * np.log(yhat.T)) / num_examples
    return loss, zs, hs, yhat

def back_prop(x, y, weightsAndBiases):
    loss, zs, hs, yhat = forward_prop(x, y, weightsAndBiases)
    num_examples = x.shape[0]
    Ws, bs = unpack(weightsAndBiases)
    dJdWs = []
    dJdbs = []
    dJdz_output = yhat - y.T
    dJdWs_output = np.dot(dJdz_output, hs[-2].T) / num_examples
    dJdbs_output = np.sum(dJdz_output, axis=1) / num_examples
    dJdWs.append(dJdWs_output)
    dJdbs.append(dJdbs_output)
    for i in range(NUM_HIDDEN_LAYERS - 1, -1, -1):
        dJdh = np.dot(Ws[i + 1].T, dJdz_output)
        dJdz = dJdh * (zs[i] > 0)
        dJdWs_hidden = np.dot(dJdz, hs[i].T) / num_examples
        dJdbs_hidden = np.sum(dJdz, axis=1) / num_examples
        dJdWs.insert(0, dJdWs_hidden)
        dJdbs.insert(0, dJdbs_hidden)
        dJdz_output = dJdz
    return np.hstack([ dJdW.flatten() for dJdW in dJdWs ] + [ dJdb.flatten() for dJdb in dJdbs ])

def train(trainX, trainY, weightsAndBiases, testX, testY):
    NUM_EPOCHS = 100
    BATCH_SIZE = 128
    trajectory = []
    for epoch in range(NUM_EPOCHS):
        for i in range(0, trainX.shape[0]):
            x = np.atleast_2d(trainX[i])
            y = np.atleast_2d(trainY[i])
            gradients = back_prop(x, y, weightsAndBiases)
            weightsAndBiases -= gradients
            trajectory.append(weightsAndBiases.copy())
    return weightsAndBiases, trajectory

def problem_2_a():
    NUM_HIDDEN_LAYERS = 3
    LEARNING_RATE = 0.001
    MINIBATCH_SIZE = 128
    NUM_EPOCHS = 30
    L2_REGULARIZATION_STRENGTH = 0.1
    NUM_INPUT = 784
    NUM_HIDDEN = 10
    NUM_OUTPUT = 10
    model = Sequential()
    model.add(Dense(NUM_HIDDEN, input_shape=(NUM_INPUT,), activation='relu', kernel_regularizer=regularizers.l2(L2_REGULARIZATION_STRENGTH)))
    for i in range(NUM_HIDDEN_LAYERS - 1):
        model.add(Dense(NUM_HIDDEN, activation='relu', kernel_regularizer=regularizers.l2(L2_REGULARIZATION_STRENGTH)))
    model.add(Dense(NUM_OUTPUT, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE), metrics=['accuracy'])
    history = model.fit(trainX, trainY, batch_size=MINIBATCH_SIZE, epochs=NUM_EPOCHS, verbose=2, validation_data=(valX, valY))
    test_loss, test_accuracy = model.evaluate(testX, testY)
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2%}')
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

def findBestHyperparameters(trainX, trainY, testX, testY, valX, valY): # problem_2_b
    grid_search_num_epochs_values : list = [60, 80, 100, 120, 140, 160, 180, 200]
    grid_search_minibatch_size_values : list = [16, 32, 64, 80, 100, 128, 160, 180, 200, 256,]
    grid_search_learning_rate_values : list = [5e-2, 3e-1, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
    grid_search_l2_regularization_strength_values : list = [0.001, 0.01, 0.1, 1.0]
    grid_search_num_hidden_layers_values : list = [3,4,5,6]
    grid_search_num_units_per_layer_values : list = [10,20,30,40,50]
    best_accuracy = 0.0
    best_hyperparameters = {}
    for MINIBATCH_SIZE in grid_search_minibatch_size_values:
        for LEARNING_RATE in grid_search_learning_rate_values:
            for L2_REGULARIZATION_STRENGTH in grid_search_l2_regularization_strength_values:
                for NUM_HIDDEN_LAYERS in grid_search_num_hidden_layers_values:
                    for NUM_HIDDEN in grid_search_num_units_per_layer_values:
                        for NUM_EPOCHS in grid_search_num_epochs_values:
                            model = Sequential()
                            model.add(Dense(NUM_HIDDEN, input_shape=(784,), activation='relu', kernel_regularizer=regularizers.l2(L2_REGULARIZATION_STRENGTH)))
                            for _ in range(NUM_HIDDEN_LAYERS - 1):
                                model.add(Dense(NUM_HIDDEN, activation='relu', kernel_regularizer=regularizers.l2(L2_REGULARIZATION_STRENGTH)))
                            model.add(Dense(10, activation='softmax'))
                            model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE), metrics=['accuracy'])
                            history = model.fit(trainX, trainY, batch_size=MINIBATCH_SIZE, epochs=NUM_EPOCHS, verbose=0, validation_data=(valX, valY))
                            test_loss, test_accuracy = model.evaluate(testX, testY)
                            print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2%}")
                            if history.history['val_accuracy'][-1] > best_accuracy:
                                best_accuracy = history.history['val_accuracy'][-1]
                                best_hyperparameters = {
                                    'EPOCH': NUM_EPOCHS,
                                    'MINIBATCH_SIZE': MINIBATCH_SIZE,
                                    'LEARNING_RATE': LEARNING_RATE,
                                    'L2_REGULARIZATION_STRENGTH': L2_REGULARIZATION_STRENGTH,
                                    'NUM_HIDDEN_LAYERS': NUM_HIDDEN_LAYERS,
                                    'NUM_HIDDEN': NUM_HIDDEN,
                                    'TEST_LOSS': test_loss,
                                    'VAL_LOSS': best_accuracy,
                                }
    print(f'Best Hyperparameters : {best_hyperparameters}, Best Validation Accuracy : {best_accuracy * 100:.2f}%')
    return None

def plotSGDPath():
    initial_weights_and_biases = initWeightsAndBiases()
    trained_weights_and_biases, trajectory, losses = train(trainX, trainY, initial_weights_and_biases, testX, testY)
    pca = PCA(n_components=2)
    pca.fit(trajectory)
    reduced_trajectory = pca.transform(trajectory)
    my_span = 100
    x_min, x_max = reduced_trajectory[:, 0].min(), reduced_trajectory[:, 0].max()
    y_min, y_max = reduced_trajectory[:, 1].min(), reduced_trajectory[:, 1].max()
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, my_span), np.linspace(y_min, y_max, my_span))
    grid_losses = griddata(reduced_trajectory, losses, (xx, yy), method='cubic')
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(projection = '3d')
    ax.plot_surface(xx, yy, grid_losses, cmap='viridis', rstride=1, cstride=1, alpha=0.8, linewidth=0)
    ax.scatter(reduced_trajectory[:, 0], reduced_trajectory[:, 1], losses, c='r', marker='o', s=10)
    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    ax.set_zlabel('Cross-Entropy Loss')
    ax.set_title('Loss Landscape and SGD Trajectories')
    plt.show()
    return None

if __name__ == "__main__":
    X_tr: np.ndarray = np.load("./fashion_mnist_train_images.npy") / 255. - 0.5
    y_tr: np.ndarray = np.load("./fashion_mnist_train_labels.npy")
    testX: np.ndarray = np.load("./fashion_mnist_test_images.npy") / 255. - 0.5
    testY: np.ndarray = np.load("./fashion_mnist_test_labels.npy")
    y_tr = np.eye(np.max(y_tr) + 1)[y_tr.flatten()]
    testY = np.eye(np.max(testY) + 1)[testY.flatten()]
    split_80_20: int = int(X_tr.shape[0]*0.8)
    trainX, trainY = X_tr[:split_80_20,:], y_tr[:split_80_20]
    valX, valY = X_tr[split_80_20:,:], y_tr[split_80_20:]
    del X_tr,y_tr

    weightsAndBiases = initWeightsAndBiases()
    print(scipy.optimize.check_grad(lambda wab: forward_prop(np.atleast_2d(trainX[0:5,:]), np.atleast_2d(trainY[0:5,:]), wab)[0], \
                                lambda wab: back_prop(np.atleast_2d(trainX[0:5,:]), np.atleast_2d(trainY[0:5,:]), wab), \
                                weightsAndBiases))
    trainX, trainY, testX, testY = trainX[0:2500,:], trainY[0:2500], testX[0:2500,:], testY[0:2500]
    plotSGDPath()
