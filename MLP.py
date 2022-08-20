from inspect import Parameter
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def ReLU(x):
    if(x < 0):
        return 0
    else:
        return x

#step function is the derivative of ReLU
def step(x):
    if(x <= 0):
        return 0
    else:
        return 1

#element-wise functions for ndarrays
ReLU_vec = np.vectorize(ReLU, otypes=[np.float64])
step_vec = np.vectorize(step, otypes=[np.float64])

#single hidden layer MLP

class MLP():

    def __init__(self, nx, nh, no):

        self.nx = nx
        self.nh = nh
        self.no = no

        #Creates and initializes parameters
        self.parameters = dict();
        self.init_params("w0", (nh,nx))
        self.init_params("b0", (nh, 1))
        self.init_params("w1", (no,nh))
        self.init_params("b1", (no, 1))

    def init_params(self, name, shape):

        par_mean = 0
        par_std = 1/10

        #initializes parameter name
        #parameters are sampled from normal distribution with mean = par_mean and std = par_std
        self.parameters[name] = np.random.normal(par_mean, par_std, size = shape )

        #initializes the gradient of parameter name
        #gradients are initialized with zeros
        self.parameters[name + "_grad"] = np.zeros( shape )

    #------Checked-------# 
    #forward propagation
    def __call__(self,x):

        #stores some intermediate values
        self.aux_values = dict()

        # x has shape (features, batch-dim)
        self.aux_values["x"] = x

        self.aux_values["a0"] = self.parameters["w0"]@x + self.parameters["b0"]

        self.aux_values["h0"] = ReLU_vec(self.aux_values["a0"])
        self.aux_values["o"] = self.parameters["w1"]@self.aux_values["h0"] + self.parameters["b1"]

        return self.aux_values["o"]

    #Loss using outputs from previous call
    def calc_loss(self, y):
        return ((self.aux_values["o"] - y)**2).mean(axis = 1)

    #Currently using MSE loss
    def back_prop(self,y):
        
        self.aux_values["o_grad"] = -2*(y - self.aux_values["o"])
        self.aux_values["h0_grad"] = self.aux_values["o_grad"]*(self.parameters["w1"].T)
        self.aux_values["a0_grad"] = self.aux_values["h0_grad"]*step_vec(self.aux_values["h0"])

        self.parameters["b1_grad"] = (self.aux_values["o_grad"]*np.ones(self.parameters["b1"].shape)).mean(axis=1,keepdims = True)
        self.parameters["b0_grad"] = (self.aux_values["a0_grad"]*np.ones(self.parameters["b0"].shape)).mean(axis=1, keepdims = True)

        self.parameters["w1_grad"] = ((self.aux_values["o_grad"]*self.aux_values["h0"]).mean(axis = 1, keepdims = True)).T
        self.parameters["w0_grad"] = (np.expand_dims( self.aux_values["a0_grad"],1)*np.expand_dims( self.aux_values["x"],0)).mean(axis = 2)


    def back_prop2(self,y):

        self.aux_values["o_grad"] = -2*(y - self.aux_values["o"]).reshape(-1,1,1)

        self.aux_values["h0_grad"] = self.aux_values["o_grad"]@np.expand_dims(self.parameters["w1"],axis = 0)
        self.aux_values["a0_grad"] = self.aux_values["h0_grad"]*step_vec(np.expand_dims((self.aux_values["h0"].T),axis = 1))

        self.parameters["b0_grad"] = (self.aux_values["a0_grad"].mean(axis = 0)).T
        self.parameters["b1_grad"] = (self.aux_values["o_grad"].mean(axis = 0)).T

        self.parameters["w1_grad"] = (np.swapaxes(self.aux_values["o_grad"], 1,2)@np.expand_dims((self.aux_values["h0"].T), axis = 1)).mean(axis = 0)
        self.parameters["w0_grad"] = (np.swapaxes(self.aux_values["a0_grad"], 1,2)@np.expand_dims((self.aux_values["x"].T), axis = 1)).mean(axis = 0)


    def gradient_descent(self, learning_rate):
        self.parameters["b0"] -= self.parameters["b0_grad"]*learning_rate
        self.parameters["b1"] -= self.parameters["b1_grad"]*learning_rate


        self.parameters["w0"] -= self.parameters["w0_grad"]*learning_rate
        self.parameters["w1"] -= self.parameters["w1_grad"]*learning_rate

#test code

def generate_data(x):
    return np.sin(2*x)

if __name__ == "__main__":

    markersize = 3

    nx = 1
    nh = 100
    no = 1
    nb = 100
    lr = 0.1

    epochs = 200

    x_train = np.sort(np.random.uniform(-2,2,(nx,nb)))
    x_test = np.linspace(-2,2,(10*nx)).reshape(nx,10)
    y_train = generate_data(x_train)
    y_test = generate_data(x_test)

    fig, ax = plt.subplots(1,2)
    ax[0].set_title("Train data")
    ax[1].set_title("Test data")
    ax[0].grid()
    ax[1].grid()

    ax[0].plot(x_train.flatten(),y_train.flatten(), "b", markersize = markersize)
    ax[1].plot(x_test.flatten(),y_test.flatten(), "b", markersize = markersize)

    mlp = MLP(nx,nh,no)

    for epoch in range(epochs):

        mlp(x_train)
        loss = mlp.calc_loss(y_train)

        mlp.back_prop2(y_train)
        mlp.gradient_descent(lr)

    print("done training")

    y_train_hat = mlp(x_train)
    y_test_hat = mlp(x_test)

    ax[0].plot(x_train.flatten(),y_train_hat.flatten(), "r", markersize = markersize )
    ax[1].plot(x_test.flatten(),y_test_hat.flatten(), "r", markersize = markersize)

    plt.show()