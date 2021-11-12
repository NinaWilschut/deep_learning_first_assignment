import math
import data
import random
import matplotlib.pyplot as plt


def cal_layer(old, weights, bias):
    # n is the length of the new list
    new = [0] * len(weights[0])
    for j, r in enumerate(new):
        for i, r2 in enumerate(old):
            new[j] += weights[i][j] * old[i]
        # print(bias)
        new[j] += bias[0][j]
    return new


def cal_back_layer(dldy, h, v):
    ly = [0] * len(dldy)
    dweights = []
    for i in range(len(v)):
        dweights.append(ly)
    dvalues = [0] * len(h)
    for j in range(len(dldy)):
        for i in range(len(h)):
            dweights[i][j] = dldy[j] * h[i]
            dvalues[i] += dldy[j] * v[i][j]

    bias = dldy
    return dweights, bias, dvalues


def sigmoid(old):
    new = [0] * len(old)
    for i, j in enumerate(new):
        new[i] = 1 / (1 + math.exp(-old[i]))
    return new


def back_sigmoid(h_hat, h):
    k_hat = [0] * len(h_hat)
    for i in range(len(h_hat)):
        k_hat[i] = h_hat[i] * h[i] * (1 - h[i])
    return k_hat


def softmax(old):
    new = [0] * len(old)
    exps = []
    for i, j in enumerate(old):
        exponential = math.exp(old[i])
        exps.append(exponential)
        for i, j in enumerate(exps):
            new[i] = exps[i] / (sum(exps))
    return new


def back_softmax(dldy, true_class, y):
    y_hat = []
    for i in range(len(dldy)):
        for j in range(len(true_class)):
            if i == j:
                y_h = dldy[i] * (y[i] * (1 - y[i]))
            else:
                y_h = dldy[i] * (-y[i] * y[j])
            y_hat.append(y_h)
    return y_hat


def forward(x, w, b, c, v, true_class):

    """
    Returns k, h, o, y, and loss
    """

    k = cal_layer(x, w, b)
    h = sigmoid(k)
    o = cal_layer(h, v, c)
    y = softmax(o)
    loss = 0
    for i in range(len(y)):
        if true_class[i]:
            loss = -math.log(y[i])
        else:
            loss += 0
    return k, h, o, y, loss


def backward(x, w, v, true_class, h, y):

    """
    Returns dl_dv, dl_dc, dl_dw, dl_db
    """

    dldy = []
    for i in range(len(y)):
        if true_class[i]:
            ly = -1 / y[i]
            dldy.append(ly)
        else:
            pass
    y_hat = back_softmax(dldy, true_class, y)
    dl_dv, dl_dc, h_hat = cal_back_layer(y_hat, h, v)
    # print(f"Derivative of v and c ={dl_dv, dl_dc}")
    k_hat = back_sigmoid(h_hat, h)
    dl_dw, dl_db, x_hat = cal_back_layer(k_hat, x, w)
    # print(f"Derivative of w and b={dl_dw, dl_db}")

    return dl_dv, dl_dc, dl_dw, dl_db


def initiate_list(
    shape, mode="num"
):  # num for random numbers normally distributed, zero for list with all zeros
    if mode == "num":
        w = []
        for i in range(shape[0]):
            numbers = []
            for j in range(shape[1]):
                numbers.append(0.01 * random.gauss(0, 1))
            w.append(numbers)
    elif mode == "zero":
        w = []
        for i in range(shape[0]):
            numbers = []
            for j in range(shape[1]):
                numbers.append(0)
            w.append(numbers)
    return w


def update_weight(w, dw, lr, mode=False):
    if mode == "first_w_layer":
        w_up = initiate_list([2, 3], "zero")
    elif mode == "second_w_layer":
        w_up = initiate_list([3, 2], "zero")
    else:
        dw = [dw]
        if mode == "bias":
            w_up = initiate_list([1, 3], "zero")
        if mode == "second_bias":
            w_up = initiate_list([1, 2], "zero")

    for i, value in enumerate(zip(w, dw)):
        for j, value2 in enumerate(zip(value[0], value[1])):
            w_up[i][j] = value2[0] - lr * value2[1]
    return w_up


"""x = [1, -1]  # start nodes
w = [[1, 1, 1], [-1, -1, -1]]  # first weights
b = [[0, 0, 0]]  # first bias
c = [[0, 0]]  # second bias
v = [[1, 1], [-1, -1], [-1, -1]]  # second weights
true_class = [1, 0]

result_forward = forward(x, w, b, c, v, true_class)
k = result_forward[0]
h = result_forward[1]
o = result_forward[2]
y = result_forward[3]
loss = result_forward[4]

result_backward = backward(x, w, v, true_class, h, y)"""


(xtrain, ytrain), (xval, yval), num_cls = data.load_synth()


def train(xtrain, ytrain, xval, yval):
    w = initiate_list([2, 3], "num")
    b = initiate_list([1, 3], "zero")
    v = initiate_list([3, 2], "num")
    c = initiate_list([1, 2], "zero")
    weights = [w, v]
    lr = 0.0001
    epochs = 10
    loss_list = []
    for epoch in range(epochs):
        error_sum = 0
        # print(loss_list)
        for x, yt in zip(xtrain, ytrain):
            t = [0, 0]
            t[yt] = 1
            k, h, o, y, loss = forward(x, w, b, c, v, t)
            # loss_list.append(loss)
            dl_dv, dl_dc, dl_dw, dl_db = backward(x, w, v, t, h, y)
            error_sum += loss
            # print("first update weights")
            w_up = update_weight(weights[0], dl_dw, lr, "first_w_layer")
            # print("second update weights")
            v_up = update_weight(weights[1], dl_dv, lr, "second_w_layer")
            # print("first update bias")
            b = update_weight(b, dl_db, lr, "bias")
            # print("second update bias")
            c = update_weight(c, dl_dc, lr, "second_bias")
            weights = [w_up, v_up]
        avg_loss = error_sum / len(xtrain)
        print(f"Epoch = {epoch}, learning rate = {lr}, error = {avg_loss}")
        # loss_list.append(avg_loss)
        # print(loss_list)

    plt.plot(loss_list)
    plt.xlabel("Epochs")
    plt.ylabel("Average loss")
    plt.title("Evaluation loss per epoch")
    plt.savefig("q4_plot")
    plt.show()
    return loss_list


train(xtrain, ytrain, xval, yval)
