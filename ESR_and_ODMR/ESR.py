# %% 
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
'''
Algoritmo para simular centros NV, com N-14. Aqui é considerado o centro NV sem campos externos, para verificar 
o ESR causado pelos seguintes Hamiltonianos: 
    (1) Estrutura Hiperfina; 
    (2) Quadripolo Nuclear; 
    (3) Zero Field Splitting.
Para centros com N-14, o spin do sistema eletrônico (6 elétrons) é S = 1 e o spin nuclear é I = 1.
'''
# Função que cria as matrizes de spin, para gerar ms = |1>, ms = |-1> e ms = |0>
def spin_1_matrices():
    # base |+1>, |0>, |-1>
    Sx = (1/np.sqrt(2)) * np.array([[0, 1, 0],
                                   [1, 0, 1],
                                   [0, 1, 0]], dtype=complex)
    Sy = (1/np.sqrt(2)) * np.array([[0, -1j, 0],
                                   [1j, 0, -1j],
                                   [0, 1j, 0]], dtype=complex)
    Sz = np.array([[1, 0, 0],
                   [0, 0, 0],
                   [0, 0, -1]], dtype=complex)
    return Sx, Sy, Sz

# Função que retorna o produto de Kronecker (produto tensorial), que será usado pra criar o espaço
# do Hamiltoniano, que é o produto tensorial entre S e I
def kron(a, b):
    return np.kron(a, b)


def build_H_gs_14N(
    Dgs_MHz=2870.0,
    Egs_MHz=0.0,
    A_par_MHz=-2.14,   # A_parallel 
    A_perp_MHz=-2.7,   # A_perp    
    Pgs_MHz=-4.95,     # quadripolo (somente 14N)
    B_G=(0.0, 0.0, 0.0),
    eps_e_MHz_per_G=2.8,    # Fatores de Zeeman (eletrônico, muito forte)
    eps_I_MHz_per_G=0.00031  # Fatores de Zeeman (nuclear, muito fraco)
):
    # spins: S=1 e I=1, 9 dimensões
    Sx, Sy, Sz = spin_1_matrices()
    Ix, Iy, Iz = spin_1_matrices()  # para I=1 é a mesma álgebra

    # Cria matriz identidade 3x3
    I3 = np.eye(3, dtype=complex)

    # operadores no espaço total (produto tensorial)
    SxT, SyT, SzT = kron(Sx, I3), kron(Sy, I3), kron(Sz, I3)
    IxT, IyT, IzT = kron(I3, Ix), kron(I3, Iy), kron(I3, Iz)

    S = 1
    I = 1

    # Hamiltoniano para o Zero Field Splitting
    H_zfs = Dgs_MHz * (SzT @ SzT - (S*(S+1)/3) * np.eye(9))
    # Termo de strain do Hamiltoniano. "Mistura" os estados ms = |1> e ms = |-1>
    H_E   = Egs_MHz * (SxT @ SxT - SyT @ SyT)

    # Hamiltoniano devido ao efeito Zeeman
    Bx, By, Bz = B_G
    H_ze = eps_e_MHz_per_G * (Bx*SxT + By*SyT + Bz*SzT)         # εe B·S
    H_zn = eps_I_MHz_per_G * (Bx*IxT + By*IyT + Bz*IzT)         # εI B·I (pequeno)

    # Hamiltoniano da estrutura hiperfina
    H_hf = A_par_MHz * (SzT @ IzT) + A_perp_MHz * (SxT @ IxT + SyT @ IyT)  
    # Hamiltoniano do quadripolo
    H_qp = Pgs_MHz * (IzT @ IzT - (I*(I+1)/3) * np.eye(9))                   

    return H_zfs + H_E + H_ze + H_hf + H_zn + H_qp

# Função de construção do Hamiltoniano devido ao MW
def build_H_mw_14N(
    Bmw_G=(1.0, 0.0, 0.0),    # polarização MW (ajuste se quiser)
    eps_e_MHz_per_G=2.8,
    eps_I_MHz_per_G=0.00031
):
    # Constrói as matrizes de spin e a identidade
    Sx, Sy, Sz = spin_1_matrices()
    Ix, Iy, Iz = spin_1_matrices()
    I3 = np.eye(3, dtype=complex)

    # Cria o espaço com produto tensorial
    SxT, SyT, SzT = np.kron(Sx, I3), np.kron(Sy, I3), np.kron(Sz, I3)
    IxT, IyT, IzT = np.kron(I3, Ix), np.kron(I3, Iy), np.kron(I3, Iz)

    # Polarização do campo MW
    Bx, By, Bz = Bmw_G
    # Hamiltoniano é o campo magnético acoplado aos spins
    return eps_e_MHz_per_G * (Bx*SxT + By*SyT + Bz*SzT) + eps_I_MHz_per_G * (Bx*IxT + By*IyT + Bz*IzT)

# Cria o perfil de absorção de Lorentz
def lorentz_absorption(w, w0, fwhm):
    # Re{L} 
    gamma = fwhm / 2.0
    return gamma / ((w - w0)**2 + gamma**2)

# Cria o espectro ESR
def esr_spectrum_14N(
    wmin=2860.0, wmax=2880.0, npts=4000,
    linewidth_MHz=0.8,
    Egs_MHz=0.0
):
    w = np.linspace(wmin, wmax, npts)

    # Constrói o Hamiltoniano do estado fundamental
    Hgs = build_H_gs_14N(Egs_MHz=Egs_MHz)
    evals, evecs = eigh(Hgs)  # autovalores (energia ou frequência) e autoestados

    # Constrói o Hamiltoniano do MW
    Hmw = build_H_mw_14N(Bmw_G=(1.0, 0.0, 0.0))

    # Início do espectro
    ESR = np.zeros_like(w, dtype=float)

    # soma sobre i<f 
    for i in range(len(evals)):
        # Autoestado inicial
        vi = evecs[:, i]
        for f in range(i+1, len(evals)):
            # Autoestado final
            vf = evecs[:, f]
            w0 = abs(evals[f] - evals[i])  # MHz (transição)

            # Calcula a amplitude de transição (momento dipolar magnético), |<f|H_mw|i>|^2
            Tif = abs(np.vdot(vf, Hmw @ vi))**2
            # Calcula o ESR
            ESR += Tif * lorentz_absorption(w, w0, linewidth_MHz)

    # normalização 
    ESR /= ESR.max()
    return w, ESR
# %% 
# Plot do ESR
if __name__ == "__main__":
    w, esr = esr_spectrum_14N(linewidth_MHz=0.8, Egs_MHz=0.0)

    plt.figure(figsize=(6.8, 4.4))
    plt.plot(w, esr, lw=2)
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Transition Strength (a.u.)")
    plt.xlim(2860, 2880)
    plt.tight_layout()
    plt.show()
# %% 
