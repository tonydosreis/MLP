import numpy as np
import matplotlib.pyplot as plt

def check_is_nonzero_integer(x):
    assert ((type(x) == int) and (x > 0))

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

#MLP
class MLP():

    def __init__(self, nx, no, hidden_layers):

        #Makes sure values are valid
        check_is_nonzero_integer(nx)
        check_is_nonzero_integer(no)
        for h in hidden_layers:
            check_is_nonzero_integer(h)

        assert len(hidden_layers) > 0

        self.nx = nx
        self.no = no
        self.hidden_layers = hidden_layers
        self.depth = len(self.hidden_layers)

        #Creates and initializes parameters
        self.parameters = dict();

        #parameters between input and first hidden layer
        self.init_params("w0", (self.hidden_layers[0],nx))
        self.init_params("b0", (self.hidden_layers[0], 1))

        #parameters between last hidden layer and output
        self.init_params(f"w{self.depth}", (no,self.hidden_layers[-1]))
        self.init_params(f"b{self.depth}", (no, 1))

        #parameters between hidden layers
        for i in range(1,self.depth):
            self.init_params(f"w{i}", (self.hidden_layers[i], self.hidden_layers[i-1]))
            self.init_params(f"b{i}", (self.hidden_layers[i], 1))

    def init_params(self, name, shape):

        par_mean = 0
        par_std = 1/10

        #initializes parameter name
        #parameters are sampled from normal distribution with mean = par_mean and std = par_std
        self.parameters[name] = np.random.normal(par_mean, par_std, size = shape )

    #forward propagation
    def __call__(self,x):

        #stores some intermediate values
        self.aux_values = dict()

        # h0 = x and has shape (features, batch-dim)
        self.aux_values["h0"] = x

        for i in range(self.depth):
            self.aux_values[f"a{i}"] = self.parameters[f"w{i}"]@self.aux_values[f"h{i}"] + self.parameters[f"b{i}"]
            self.aux_values[f"h{i + 1}"] = ReLU_vec(self.aux_values[f"a{i}"])

        #special treatment for output, no nonlinearity
        i += 1
        self.aux_values["o"] = self.parameters[f"w{i}"]@self.aux_values[f"h{i}"] + self.parameters[f"b{i}"]

        return self.aux_values["o"]

    #Loss using outputs from previous call
    def calc_loss(self, y):
        return ((self.aux_values["o"] - y)**2).mean()

    #Currently using MSE loss
    def back_prop(self,y):

        self.aux_values["o_grad"] = -2*(y - self.aux_values["o"])

        self.aux_values[f"h{self.depth}_grad"] = np.expand_dims(self.parameters[f"w{self.depth}"].T,axis = 0)@self.aux_values["o_grad"]
        self.aux_values[f"w{self.depth}_grad"] = (self.aux_values["o_grad"]@np.swapaxes(self.aux_values[f"h{self.depth}"],1,2)).mean(axis = 0)
        self.aux_values[f"b{self.depth}_grad"] = (self.aux_values["o_grad"]).mean(axis = 0)

        for i in range(self.depth - 1, -1,-1):

            self.aux_values[f"a{i}_grad"] = self.aux_values[f"h{i + 1}_grad"]*step_vec(self.aux_values[f"a{i}"])

            self.aux_values[f"h{i}_grad"] = np.expand_dims(self.parameters[f"w{i}"].T,axis = 0)@self.aux_values[f"a{i}_grad"]        
            self.aux_values[f"w{i}_grad"] = (self.aux_values[f"a{i}_grad"]@np.swapaxes(self.aux_values[f"h{i}"],1,2)).mean(axis = 0)
            self.aux_values[f"b{i}_grad"] = (self.aux_values[f"a{i}_grad"]).mean(axis = 0)

    def gradient_descent(self, learning_rate):
        for i in range(self.depth + 1):
            self.parameters[f"b{i}"] -= self.aux_values[f"b{i}_grad"]*learning_rate
            self.parameters[f"w{i}"] -= self.aux_values[f"w{i}_grad"]*learning_rate

#test code

#Linear function
def generate_data(x,M):
    return M@x

def generate_data_sin(x,f):
    return np.sin(2*np.pi*f*x)

if __name__ == "__main__":

    markersize = 3

    f = 0.4
    nx = 1
    H = [10]
    no = 1
    nb_train = 100
    nb_test = 10
    lr = .01

    epochs = 10000

    x_train = np.sort(np.random.uniform(-2,2,(nb_train*nx))).reshape(nb_train,nx,1)

    x_test = np.linspace(-2,2,(nb_test*nx)).reshape(nb_test,nx,1)

    M = np.random.uniform(size = (1,no,nx))

    y_train = generate_data(x_train,M)
    y_test = generate_data(x_test,M)

    y_train = generate_data_sin(x_train,f)
    y_test = generate_data_sin(x_test,f)

    mlp = MLP(nx,no, H)

    for epoch in range(epochs):

        mlp(x_train)
        loss = mlp.calc_loss(y_train)
        print(f"Loss: {loss:.5f}")

        mlp.back_prop(y_train)
        mlp.gradient_descent(lr)

    print("done training")

    y_train_hat = mlp(x_train)
    y_test_hat = mlp(x_test)

    fig, ax = plt.subplots(1,2)
    ax[0].set_title("Train data")
    ax[1].set_title("Test data")
    ax[0].grid()
    ax[1].grid()

    ax[0].plot(x_train.flatten(),y_train.flatten(), "b", markersize = markersize)
    ax[0].plot(x_train.flatten(),y_train_hat.flatten(), "r", markersize = markersize )
    ax[1].plot(x_test.flatten(),y_test.flatten(), "b", markersize = markersize)
    ax[1].plot(x_test.flatten(),y_test_hat.flatten(), "r", markersize = markersize)

    plt.show()