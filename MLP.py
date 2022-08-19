import numpy as np

def ReLU(x):
    if(x < 0):
        return 0
    else:
        return x

#step function is the derivative of ReLU
def step(x):
    if(x < 0):
        return 0
    else:
        return 1

#element-wise functions for ndarrays
ReLU_vec = np.vectorize(ReLU)
step_vec = np.vectorize(step)


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

        #stores some intermediate values
        self.values = dict()

    def init_params(self, name, shape):

        par_mean = 0
        par_std = 1

        #initializes parameter name
        #parameters are sampled from normal distribution with mean = par_mean and std = par_std
        self.parameters[name] = np.random.normal(par_mean, par_std, size = shape )

        #initializes the gradient of parameter name
        #gradients are initialized with zeros
        self.parameters[name + "_grad"] = np.zeros( shape )


    #forward propagation
    def __call__(self,x = 0):

        # x has shape (features, batch-dim)
        self.values["x"] = x

        self.values["a0"] = self.parameters["w0"]@x + self.parameters["b0"]
        self.values["h0"] = ReLU_vec(self.values["a0"])
        self.values["o"] = self.parameters["w1"]@self.values["h0"] + self.parameters["b1"]

        return self.values["o"]

#test code
if __name__ == "__main__":
    nx = 2 
    nh = 10 
    no = 1
    nb = 10

    x = np.random.normal(0,1, (nx,nb) )

    mlp = MLP(nx,nh,no)
    print(mlp(x))