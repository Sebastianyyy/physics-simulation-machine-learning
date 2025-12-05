import torch                           # Pytorch
import torch.autograd as autograd      # computation graph
from torch import Tensor
import torch.nn as nn                  # neural networks
import torch.optim as optim            # optimizers
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from tqdm import tqdm                      # progress bar
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
import time, sys
from tqdm.notebook import tqdm_notebook

#Set default dtype to float32
torch.set_default_dtype(torch.float)

#PyTorch random number generator
torch.manual_seed(1234)
from pyDOE import lhs
#We will use Latin Hypercube Sampling from this library

# Random number generators in other libraries
np.random.seed(1234)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("The neural network will be trainied on",device)

if device == 'cuda':
    print(torch.cuda.get_device_name())



#x∈[0,2]
x_min=0
x_max=2

#t∈[0,0.48]
t_min=0
t_max=0.48

viscosity = 0.01/np.pi # Given


#Discretization points for x and t

total_points_x=1001
total_points_t=1000

dx = (x_max-x_min)/(total_points_x-1)

dt = (t_max-t_min)/(total_points_t)


dx,dt, dt/dt

#Implementing Finite Difference Method to solve the 1D Diffusion Equation

def u_fem(x,t):
    un = torch.ones(total_points_x)
    rec = torch.zeros([total_points_x, total_points_t])

    for j in tqdm(range(total_points_t)):

        un = u.clone()

        for i in range(1,total_points_x-1):
            rec[i,j] = u[i]
            u[i] = un[i] - un[i] * dt/dx * (un[i]-un[i-1])  + viscosity * (dt/dx**2) * (un[i+1]- 2*un[i] + un[i-1])
            if np.isnan(u[i]):
              print(i, j, u[i-1])
              break

    return u, rec


x = torch.linspace(x_min, x_max, total_points_x)

u = torch.from_numpy(np.sin(np.pi*x.numpy()))


x = torch.linspace(x_min, x_max, total_points_x).view(-1,1)
t = torch.linspace(t_min, t_max, total_points_t).view(-1,1)
x.shape, t.shape


print("Running Finite Difference Method...")
u_final, u_fem_2D = u_fem(x,t)

assert u_fem_2D.shape == torch.Size([total_points_x, total_points_t]),f"Expected [{total_points_x},{total_points_t}], got {u_fem_2D.shape}"
print("Completed successfully!")

X, T = torch.meshgrid(x.squeeze(1),t.squeeze(1), indexing='ij')


# Creating same amount of grid lattice as FDM
x = torch.linspace(x_min, x_max, total_points_x).view(-1,1)
t = torch.linspace(t_min, t_max, total_points_t).view(-1,1)
x.shape, t.shape

X, T = torch.meshgrid(x.squeeze(1),t.squeeze(1), indexing='ij') #same as FDM
X.shape, T.shape


left_X = torch.hstack((X[:,0][:,None], T[:,0][:,None])) #horizontal stacking to create X, T dataset


left_U = torch.sin(np.pi*left_X[:,0]).unsqueeze(1) #initial condition is a sine wave
left_U.shape

# BC at x_min
bottom_X = torch.hstack((X[0,:][:,None],T[0,:][:,None]))
top_X = torch.hstack((X[-1,:][:,None],T[-1,:][:,None]))

bottom_U = torch.zeros(bottom_X.shape[0],1)
top_U = torch.zeros(top_X.shape[0],1)

bottom_X.shape


X_bc = torch.vstack([bottom_X, top_X])
U_bc = torch.vstack([bottom_U, top_U])

X_bc.shape


N_ic = 1000
N_bc = 1000 #Number of points on IC and BC
N_pde = 30000 #Number of points on PDE domain (Collocation Points)

#Now we will sample N_bc points at random
#from the X_train, U_train dataset

idx = np.random.choice(X_bc.shape[0],N_bc, replace=False)
X_bc_samples = X_bc[idx,:]
U_bc_samples = U_bc[idx,:]

idx = np.random.choice(left_X.shape[0],N_ic, replace=False)
X_ic_samples = left_X[idx,:]
U_ic_samples = left_U[idx,:]

#The boundary conditions will not change.
#Hence, these U values can be used as supervised labels during training

#For PDE collocation points, we will generate new X_train_pde dataset
#We do not know U(X,T) for these points

#Lets get the entire X,T dataset in a format suitable for Neural Network
#We will later use this for testing NN as well. So, lets call this x_test for convenience

x_test = torch.hstack((X.transpose(1,0).flatten()[:,None],
                       T.transpose(1,0).flatten()[:,None]))

#We need column major flattening to simlulte time-marching. Hence the transpose(1,0) or simply use .T

#we will use U generated from FEM as our u_test
#We will use u_test later in the process for calculating NN performance

u_test = u_fem_2D.transpose(1,0).flatten()[:,None]
x_test.shape


#lower and upper bounds of x_test
lb = x_test[0]
ub = x_test[-1]
lb,ub


#Sampling (X,T) domain using LHS
lhs_samples = lhs(2,N_pde)
#2 since there are 2 variables in X_train, [x,t]
lhs_samples.shape


X_train_lhs = lb + (ub-lb)*lhs_samples
X_train_lhs.shape


X_train_final = torch.vstack((X_train_lhs, X_ic_samples, X_bc_samples))
X_train_final.shape



#Lets define a u_NN

class u_NN(nn.Module):

    def __init__(self, layers_list):

        super().__init__()

        self.depth = len(layers_list)

        self.loss_function = nn.MSELoss(reduction="mean")

        self.activation = nn.Tanh() #This is important, ReLU wont work

        self.linears = nn.ModuleList([nn.Linear(layers_list[i],layers_list[i+1]) for i in range(self.depth-1)])

        for i in range(self.depth-1):

          nn.init.xavier_normal_(self.linears[i].weight.data, gain=1.0) #xavier normalization of weights

          nn.init.zeros_(self.linears[i].bias.data) #all biases set to zero

    def Convert(self, x): #helper function

        if torch.is_tensor(x) !=True:
            x = torch.from_numpy(x)
        return x.float().to(device)

    def forward(self, x):

        a = self.Convert(x)

        for i in range(self.depth-2):
            z = self.linears[i](a)
            a = self.activation(z)

        a = self.linears[-1](a)

        return a

    def loss_bc(self, x_bc, u_bc):
        #This is similar to a Supervised Learning

        l_bc = self.loss_function(self.forward(self.Convert(x_bc)), self.Convert(u_bc)) #L2 loss

        return l_bc

    def loss_ic(self, x_ic, u_ic):
        #This is similar to a Supervised Learning

        l_ic = self.loss_function(self.forward(self.Convert(x_ic)), self.Convert(u_ic)) #L2 loss

        return l_ic

    def loss_pde(self, x_pde):
        # We will pass x_train_final here.
        # Note that we do not have U_pde (labels) here to calculate loss. This is not Supervised Learning.
        # Here we want to minimize the residues. So, we will first calculate the residue and then minimize it to be close to zero.

        x_pde = self.Convert(x_pde)

        x_pde_clone = x_pde.clone() ##VERY IMPORTANT

        x_pde_clone.requires_grad = True #enable Auto Differentiation

        NN = self.forward(x_pde_clone) #Generates predictions from u_NN

        NNx_NNt = torch.autograd.grad(NN, x_pde_clone,self.Convert(torch.ones([x_pde_clone.shape[0],1])),retain_graph=True, create_graph=True)[0] #Jacobian of dx and dt

        NNxx_NNtt = torch.autograd.grad(NNx_NNt,x_pde_clone, self.Convert(torch.ones(x_pde_clone.shape)), create_graph=True)[0] #Jacobian of dx2, dt2

        NNxx = NNxx_NNtt[:,[0]] #Extract only dx2 terms

        NNt = NNx_NNt[:,[1]] #Extract only dt terms

        NNx = NNx_NNt[:,[0]] #Extract only dx terms

        # {(du/dt) = viscosity * (d2u/dx2)} is the pde and the NN residue will be {du_NN/dt - viscosity*(d2u_NN)/dx2} which is == {NNt - viscosity*NNxx}

        residue = NNt + self.forward(x_pde_clone)*(NNx) - (viscosity)*NNxx

        # The residues need to be zero (or as low as possible). We'll create an arrazy of Zeros and minimize the residue

        zeros = self.Convert(torch.zeros(residue.shape[0],1))

        l_pde = self.loss_function(residue, zeros) #L2 Loss

        return l_pde

    def total_loss(self, x_ic, u_ic, x_bc, u_bc, x_pde): #Combine both loss
        l_bc = self.loss_bc(x_bc, u_bc)
        l_ic = self.loss_ic(x_ic, u_ic)
        l_pde = self.loss_pde(x_pde)
        return l_bc + l_pde + l_ic #this HAS to be a scalar value for auto differentiation to do its thing.



#Parameters for u_NN

EPOCHS = 100000
initial_lr = 0.001
layers_list = [2, 32, 1]
#batch_size = 32

# Instantiate a model

PINN = u_NN(layers_list).to(device)
print(PINN)

optimizer = torch.optim.Adam(PINN.parameters(), lr=initial_lr,amsgrad=False)

scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.985)

history = pd.DataFrame(columns=["Epochs","Learning_Rate", "IC_Loss","BC_Loss","PDE_Loss","Total_Loss","Test_Loss"])



#****** Training ******#

print("Training Physics Informed Neural Network...")

Epoch = []
Learning_Rate = []
IC_Loss = []
BC_Loss = []
PDE_Loss = []
Total_Loss = []
Test_Loss = []

for i in tqdm(range(EPOCHS)):
    if i==0:
        print("Epoch \t Learning_Rate \t IC_Loss \t BC_Loss \t PDE_Loss \t Total_Loss \t Test_Loss")

    l_ic = PINN.loss_ic(X_ic_samples,U_ic_samples)
    l_bc = PINN.loss_bc(X_bc_samples,U_bc_samples)
    l_pde = PINN.loss_pde(X_train_final)
    loss = PINN.total_loss(X_ic_samples,U_ic_samples,X_bc_samples,U_bc_samples, X_train_final)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if i%100 == 0: #print losses and step the exponential learning rate.

        with torch.no_grad():
            test_loss = PINN.loss_bc(x_test,u_test) #Here we are using loss_bc method as a helper function to calculate L2 loss

            Epoch.append(i)
            Learning_Rate.append(scheduler.get_last_lr()[0])
            IC_Loss.append(l_ic.detach().cpu().numpy())
            BC_Loss.append(l_bc.detach().cpu().numpy())
            PDE_Loss.append(l_pde.detach().cpu().numpy())
            Total_Loss.append(loss.detach().cpu().numpy())
            Test_Loss.append(test_loss.detach().cpu().numpy())

            if i%1000 ==0:
               print(i,'\t',format(scheduler.get_last_lr()[0],".4E"),'\t',format(l_ic.detach().cpu().numpy(),".4E"),'\t',format(l_bc.detach().cpu().numpy(),".4E"),'\t',
                  format(l_pde.detach().cpu().numpy(),".4E"),'\t',format(loss.detach().cpu().numpy(),".4E"),'\t',format(test_loss.detach().cpu().numpy(),".4E"))

        scheduler.step()

print("Completed!!")



u_NN_predict = PINN(x_test)


u_NN_2D = u_NN_predict.reshape(shape=[total_points_t,total_points_x]).transpose(1,0).detach().cpu()

assert u_NN_2D.shape == torch.Size([total_points_x, total_points_t]),f"Expected [{total_points_x},{total_points_t}], got {u_NN_2D.shape}"


RMSE = torch.sqrt(torch.mean(torch.square(torch.subtract(u_NN_2D,u_fem_2D))))

print("The RMSE error between FDM and PINN is :",np.around(RMSE.item(),5))


file_name_result = "Result_RMSE_"+str(np.around(RMSE.item(),5))+"_Epochs_"+str(EPOCHS)+"_lr_"+str(initial_lr)+"SHALLOW.png"

last_U_NN = u_NN_2D[:,-1].unsqueeze(1) #Extracting the last U values at t=0.48


file_name_model = "Result_RMSE_"+str(np.around(RMSE.item(),5))+"_Epochs_"+str(EPOCHS)+"_lr_"+str(initial_lr)+"SHALLOW.pth"
torch.save(PINN, "./"+file_name_model)

