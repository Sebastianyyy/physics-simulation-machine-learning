"""
PINN for 1D Heat Equation (Równanie Ciepła)
============================================
Równanie: ∂u/∂t = α * ∂²u/∂x²

Gdzie:
- u(x,t) - temperatura
- α - współczynnik dyfuzji cieplnej (thermal diffusivity)

Warunki brzegowe:
- u(0,t) = 0  (lewy koniec)
- u(L,t) = 0  (prawy koniec)

Warunek początkowy:
- u(x,0) = sin(πx/L)

Rozwiązanie analityczne:
- u(x,t) = sin(πx/L) * exp(-α*(π/L)²*t)
"""

import torch
import torch.autograd as autograd
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from tqdm import tqdm
import numpy as np
import time
import pandas as pd
import time, sys

# Set default dtype to float32
torch.set_default_dtype(torch.float)

# PyTorch random number generator
torch.manual_seed(1234)
from pyDOE import lhs

# Random number generators in other libraries
np.random.seed(1234)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("="*60)
print("PINN dla Równania Ciepła 1D (Heat Equation)")
print("="*60)
print(f"Urządzenie: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name()}")

# =============================================================================
# PARAMETRY FIZYCZNE I DOMENY
# =============================================================================

# Domena przestrzenna x ∈ [0, L]
x_min = 0
x_max = 1.0  # L = 1

# Domena czasowa t ∈ [0, T]
t_min = 0
t_max = 0.5

# Współczynnik dyfuzji cieplnej
alpha = 0.1

print(f"\nParametry domeny:")
print(f"  x ∈ [{x_min}, {x_max}]")
print(f"  t ∈ [{t_min}, {t_max}]")
print(f"  α (dyfuzja) = {alpha}")

# Dyskretyzacja
total_points_x = 101
total_points_t = 500

dx = (x_max - x_min) / (total_points_x - 1)
dt = (t_max - t_min) / total_points_t

print(f"\nDyskretyzacja:")
print(f"  Punkty x: {total_points_x}, dx = {dx:.6f}")
print(f"  Punkty t: {total_points_t}, dt = {dt:.6f}")

# Warunek stabilności CFL dla równania ciepła: dt <= dx²/(2*alpha)
cfl = alpha * dt / (dx**2)
print(f"  CFL = α*dt/dx² = {cfl:.4f} (powinno być < 0.5 dla stabilności FDM)")

# =============================================================================
# ROZWIĄZANIE ANALITYCZNE
# =============================================================================

def u_analytical(x, t, alpha, L=1.0):
    """
    Rozwiązanie analityczne równania ciepła 1D.
    u(x,t) = sin(πx/L) * exp(-α*(π/L)²*t)
    """
    return np.sin(np.pi * x / L) * np.exp(-alpha * (np.pi / L)**2 * t)

# =============================================================================
# ROZWIĄZANIE ANALITYCZNE - PRZYGOTOWANIE DANYCH
# =============================================================================

# Siatka przestrzenna i czasowa
x = torch.linspace(x_min, x_max, total_points_x).view(-1, 1)
t = torch.linspace(t_min, t_max, total_points_t).view(-1, 1)

print("\n" + "="*60)
print("Obliczanie rozwiązania analitycznego...")

# Meshgrid
X, T = torch.meshgrid(x.squeeze(1), t.squeeze(1), indexing='ij')

# Rozwiązanie analityczne na całej siatce
u_analytical_2D = torch.from_numpy(u_analytical(X.numpy(), T.numpy(), alpha)).float()

print("Rozwiązanie analityczne obliczone!")

# =============================================================================
# PRZYGOTOWANIE DANYCH TRENINGOWYCH
# =============================================================================

print("\n" + "="*60)
print("Przygotowanie danych treningowych...")

# Punkty na warunku początkowym (IC)
left_X = torch.hstack((X[:, 0][:, None], T[:, 0][:, None]))
left_U = torch.sin(np.pi * left_X[:, 0]).unsqueeze(1)

# Punkty na warunkach brzegowych (BC)
bottom_X = torch.hstack((X[0, :][:, None], T[0, :][:, None]))  # x = 0
top_X = torch.hstack((X[-1, :][:, None], T[-1, :][:, None]))    # x = L

bottom_U = torch.zeros(bottom_X.shape[0], 1)  # u(0, t) = 0
top_U = torch.zeros(top_X.shape[0], 1)        # u(L, t) = 0

X_bc = torch.vstack([bottom_X, top_X])
U_bc = torch.vstack([bottom_U, top_U])

# Liczba punktów próbkowania (dostosowane do rozmiaru danych)
N_ic = min(500, left_X.shape[0])  # Nie więcej niż dostępne punkty IC
N_bc = min(500, X_bc.shape[0])    # Nie więcej niż dostępne punkty BC
N_pde = 20000

print(f"Dostępne punkty IC: {left_X.shape[0]}, używamy: {N_ic}")
print(f"Dostępne punkty BC: {X_bc.shape[0]}, używamy: {N_bc}")

# Próbkowanie IC
idx = np.random.choice(left_X.shape[0], N_ic, replace=False)
X_ic_samples = left_X[idx, :]
U_ic_samples = left_U[idx, :]

# Próbkowanie BC
idx = np.random.choice(X_bc.shape[0], N_bc, replace=False)
X_bc_samples = X_bc[idx, :]
U_bc_samples = U_bc[idx, :]

# Dane testowe
x_test = torch.hstack((X.transpose(1, 0).flatten()[:, None],
                       T.transpose(1, 0).flatten()[:, None]))
u_test = u_analytical_2D.transpose(1, 0).flatten()[:, None]

# Dolna i górna granica domeny
lb = x_test[0]
ub = x_test[-1]

# Punkty kolokacji PDE (LHS)
lhs_samples = lhs(2, N_pde)
X_train_lhs = lb + (ub - lb) * lhs_samples
X_train_final = torch.vstack((X_train_lhs, X_ic_samples, X_bc_samples))

print(f"Punkty IC: {N_ic}")
print(f"Punkty BC: {N_bc}")
print(f"Punkty PDE (kolokacja): {N_pde}")
print(f"Łącznie punktów treningowych: {X_train_final.shape[0]}")

# =============================================================================
# DEFINICJA SIECI NEURONOWEJ PINN
# =============================================================================

class HeatPINN(nn.Module):
    """
    Physics-Informed Neural Network dla równania ciepła.
    Residuum: ∂u/∂t - α * ∂²u/∂x² = 0
    """
    
    def __init__(self, layers_list, alpha):
        super().__init__()
        
        self.depth = len(layers_list)
        self.alpha = alpha
        self.loss_function = nn.MSELoss(reduction="mean")
        self.activation = nn.Tanh()
        
        self.linears = nn.ModuleList([
            nn.Linear(layers_list[i], layers_list[i+1]) 
            for i in range(self.depth - 1)
        ])
        
        # Inicjalizacja wag
        for i in range(self.depth - 1):
            nn.init.xavier_normal_(self.linears[i].weight.data, gain=1.0)
            nn.init.zeros_(self.linears[i].bias.data)
    
    def Convert(self, x):
        if torch.is_tensor(x) != True:
            x = torch.from_numpy(x)
        return x.float().to(device)
    
    def forward(self, x):
        a = self.Convert(x)
        
        for i in range(self.depth - 2):
            z = self.linears[i](a)
            a = self.activation(z)
        
        a = self.linears[-1](a)
        return a
    
    def loss_bc(self, x_bc, u_bc):
        """Strata na warunkach brzegowych."""
        l_bc = self.loss_function(self.forward(self.Convert(x_bc)), self.Convert(u_bc))
        return l_bc
    
    def loss_ic(self, x_ic, u_ic):
        """Strata na warunku początkowym."""
        l_ic = self.loss_function(self.forward(self.Convert(x_ic)), self.Convert(u_ic))
        return l_ic
    
    def loss_pde(self, x_pde):
        """
        Strata na residuum PDE (równanie ciepła).
        Residuum: ∂u/∂t - α * ∂²u/∂x² = 0
        """
        x_pde = self.Convert(x_pde)
        x_pde_clone = x_pde.clone()
        x_pde_clone.requires_grad = True
        
        NN = self.forward(x_pde_clone)
        
        # Pierwsze pochodne
        NNx_NNt = torch.autograd.grad(
            NN, x_pde_clone, 
            self.Convert(torch.ones([x_pde_clone.shape[0], 1])),
            retain_graph=True, create_graph=True
        )[0]
        
        # Drugie pochodne
        NNxx_NNtt = torch.autograd.grad(
            NNx_NNt, x_pde_clone, 
            self.Convert(torch.ones(x_pde_clone.shape)), 
            create_graph=True
        )[0]
        
        NNx = NNx_NNt[:, [0]]   # ∂u/∂x
        NNt = NNx_NNt[:, [1]]   # ∂u/∂t
        NNxx = NNxx_NNtt[:, [0]] # ∂²u/∂x²
        
        # Residuum równania ciepła: ∂u/∂t - α * ∂²u/∂x² = 0
        residue = NNt - self.alpha * NNxx
        
        zeros = self.Convert(torch.zeros(residue.shape[0], 1))
        l_pde = self.loss_function(residue, zeros)
        
        return l_pde
    
    def total_loss(self, x_ic, u_ic, x_bc, u_bc, x_pde):
        """Łączna strata."""
        l_bc = self.loss_bc(x_bc, u_bc)
        l_ic = self.loss_ic(x_ic, u_ic)
        l_pde = self.loss_pde(x_pde)
        return l_bc + l_pde + l_ic

# =============================================================================
# TRENING SIECI
# =============================================================================

print("\n" + "="*60)
print("Konfiguracja treningu...")

# Hiperparametry
EPOCHS = 50000
initial_lr = 0.001
layers_list = [2, 64, 64, 64, 64, 1]

print(f"Epochs: {EPOCHS}")
print(f"Learning rate: {initial_lr}")
print(f"Architektura sieci: {layers_list}")

# Inicjalizacja modelu
PINN = HeatPINN(layers_list, alpha).to(device)
print(f"\nModel:\n{PINN}")

optimizer = torch.optim.Adam(PINN.parameters(), lr=initial_lr, amsgrad=False)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

# Listy do zapisu historii
Epoch = []
Learning_Rate = []
IC_Loss = []
BC_Loss = []
PDE_Loss = []
Total_Loss = []
Test_Loss = []

print("\n" + "="*60)
print("Rozpoczynam trening PINN dla równania ciepła...")
print("="*60)

for i in tqdm(range(EPOCHS)):
    if i == 0:
        print("Epoch \t LR \t\t IC_Loss \t BC_Loss \t PDE_Loss \t Total_Loss \t Test_Loss")
    
    l_ic = PINN.loss_ic(X_ic_samples, U_ic_samples)
    l_bc = PINN.loss_bc(X_bc_samples, U_bc_samples)
    l_pde = PINN.loss_pde(X_train_final)
    loss = PINN.total_loss(X_ic_samples, U_ic_samples, X_bc_samples, U_bc_samples, X_train_final)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if i % 100 == 0:
        with torch.no_grad():
            test_loss = PINN.loss_bc(x_test, u_test)
            
            Epoch.append(i)
            Learning_Rate.append(scheduler.get_last_lr()[0])
            IC_Loss.append(l_ic.detach().cpu().numpy())
            BC_Loss.append(l_bc.detach().cpu().numpy())
            PDE_Loss.append(l_pde.detach().cpu().numpy())
            Total_Loss.append(loss.detach().cpu().numpy())
            Test_Loss.append(test_loss.detach().cpu().numpy())
            
            if i % 2000 == 0:
                print(f"{i}\t{scheduler.get_last_lr()[0]:.4E}\t{l_ic.detach().cpu().numpy():.4E}\t"
                      f"{l_bc.detach().cpu().numpy():.4E}\t{l_pde.detach().cpu().numpy():.4E}\t"
                      f"{loss.detach().cpu().numpy():.4E}\t{test_loss.detach().cpu().numpy():.4E}")
        
        scheduler.step()

print("\n" + "="*60)
print("Trening zakończony!")
print("="*60)

# =============================================================================
# EWALUACJA WYNIKÓW
# =============================================================================

print("\nEwaluacja wyników...")

# Predykcja PINN
u_NN_predict = PINN(x_test)
u_NN_2D = u_NN_predict.reshape(shape=[total_points_t, total_points_x]).transpose(1, 0).detach().cpu()

# Oblicz RMSE
RMSE_pinn_analytical = torch.sqrt(torch.mean(torch.square(u_NN_2D - u_analytical_2D)))

print("\n" + "="*60)
print("WYNIKI KOŃCOWE - RÓWNANIE CIEPŁA")
print("="*60)
print(f"RMSE (PINN vs Analityczne): {RMSE_pinn_analytical.item():.6f}")
print("="*60)

# =============================================================================
# WIZUALIZACJA
# =============================================================================

print("\nGenerowanie wykresów...")

# Wykres strat
fig, ax1 = plt.subplots(figsize=(12, 6))
ax2 = ax1.twinx()

ax1.semilogy(Epoch, IC_Loss, "b-", label="IC Loss")
ax1.semilogy(Epoch, BC_Loss, "g-", label="BC Loss")
ax1.semilogy(Epoch, PDE_Loss, "c-", label="PDE Loss")
ax1.semilogy(Epoch, Total_Loss, "k-", label="Total Loss")
ax1.semilogy(Epoch, Test_Loss, "m-", label="Test Loss")
ax2.plot(Epoch, Learning_Rate, "ro", markersize=1, label="Learning Rate")

ax1.set_xlabel('Epochs')
ax1.set_ylabel('Losses (log scale)', color='k')
ax2.set_ylabel('Learning Rate', color='r')
ax1.legend(loc='upper right')
ax2.legend(loc='center right')
plt.title(f'Heat Equation PINN - Training Losses\nRMSE vs Analityczne: {RMSE_pinn_analytical.item():.5f}')
plt.savefig("heat_equation_loss_curves.png", dpi=300, bbox_inches='tight')
plt.close()

# Porównanie rozwiązań w różnych momentach czasu
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

time_snapshots = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
x_plot = x.squeeze().numpy()

for idx, t_val in enumerate(time_snapshots):
    ax = axes[idx // 3, idx % 3]
    t_idx = int(t_val / t_max * (total_points_t - 1))
    
    ax.plot(x_plot, u_analytical_2D[:, t_idx].numpy(), 'b-', linewidth=2, label='Analityczne')
    ax.plot(x_plot, u_NN_2D[:, t_idx].numpy(), 'r--', linewidth=2, label='PINN')
    
    ax.set_xlabel('x')
    ax.set_ylabel('u(x,t)')
    ax.set_title(f't = {t_val:.2f}')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.suptitle('Heat Equation: Porównanie rozwiązań\n(Analityczne vs PINN)', fontsize=14)
plt.tight_layout()
plt.savefig("heat_equation_comparison.png", dpi=300, bbox_inches='tight')
plt.close()

# Heatmapy
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Analityczne
im1 = axes[0].imshow(u_analytical_2D.numpy().T, aspect='auto', cmap='hot',
                      extent=[x_min, x_max, t_min, t_max], origin='lower')
axes[0].set_xlabel('x')
axes[0].set_ylabel('t')
axes[0].set_title('Rozwiązanie Analityczne')
plt.colorbar(im1, ax=axes[0])

# PINN
im2 = axes[1].imshow(u_NN_2D.numpy().T, aspect='auto', cmap='hot',
                      extent=[x_min, x_max, t_min, t_max], origin='lower')
axes[1].set_xlabel('x')
axes[1].set_ylabel('t')
axes[1].set_title('PINN')
plt.colorbar(im2, ax=axes[1])

# Błąd
error = torch.abs(u_NN_2D - u_analytical_2D)
im3 = axes[2].imshow(error.numpy().T, aspect='auto', cmap='coolwarm',
                      extent=[x_min, x_max, t_min, t_max], origin='lower')
axes[2].set_xlabel('x')
axes[2].set_ylabel('t')
axes[2].set_title('|PINN - Analityczne|')
plt.colorbar(im3, ax=axes[2])

plt.suptitle('Heat Equation: Heatmapy rozwiązań', fontsize=14)
plt.tight_layout()
plt.savefig("heat_equation_heatmaps.png", dpi=300, bbox_inches='tight')
plt.close()

# =============================================================================
# ZAPIS MODELU I WYNIKÓW
# =============================================================================

model_filename = f"heat_RMSE_{np.around(RMSE_pinn_analytical.item(), 5)}_Epochs_{EPOCHS}_lr_{initial_lr}.pth"
torch.save(PINN, model_filename)
print(f"\nModel zapisano jako: {model_filename}")

# Zapis wyników do pliku
results_filename = "heat_equation_results.txt"
with open(results_filename, 'w') as f:
    f.write("="*60 + "\n")
    f.write("WYNIKI PINN DLA RÓWNANIA CIEPŁA 1D\n")
    f.write("="*60 + "\n\n")
    f.write("Równanie: ∂u/∂t = α * ∂²u/∂x²\n\n")
    f.write(f"Parametry:\n")
    f.write(f"  α (dyfuzja) = {alpha}\n")
    f.write(f"  x ∈ [{x_min}, {x_max}]\n")
    f.write(f"  t ∈ [{t_min}, {t_max}]\n\n")
    f.write(f"Hiperparametry treningu:\n")
    f.write(f"  Epochs: {EPOCHS}\n")
    f.write(f"  Learning rate: {initial_lr}\n")
    f.write(f"  Architektura: {layers_list}\n\n")
    f.write(f"WYNIKI:\n")
    f.write(f"  RMSE (PINN vs Analityczne): {RMSE_pinn_analytical.item():.6f}\n")

print(f"Wyniki zapisano jako: {results_filename}")
print("\nWszystkie pliki wygenerowane:")
print("  - heat_equation_loss_curves.png")
print("  - heat_equation_comparison.png")
print("  - heat_equation_heatmaps.png")
print(f"  - {model_filename}")
print(f"  - {results_filename}")
print("\nGotowe!")
