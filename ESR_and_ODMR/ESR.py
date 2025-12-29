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

D_GS_MHz = 2870.0
GAMMA_E_MHz_per_G = 2.8


def dc_field_nv_frame(B_amp_G: float, polar_angle_rad: float) -> tuple[float, float, float]:
    """B em coordenadas do NV (Bx, By, Bz) com ângulo polar phi."""
    Bz = B_amp_G * np.cos(polar_angle_rad)
    Bx = B_amp_G * np.sin(polar_angle_rad)
    By = 0.0
    return (Bx, By, Bz)


def esr_spectrum_single_nv_14N(
    frequency_axis_MHz: np.ndarray,
    B_amp_G: float,
    polar_angle_rad: float,
    strain_E_MHz: float,
    linewidth_MHz: float,
    mw_field_G_nv: tuple[float, float, float] = (1.0, 0.0, 0.0),
) -> np.ndarray:
    """
    ESR(f) = sum_{i<j} |<j|Hmw|i>|^2 * Lorentz(f; f_ij, linewidth)
    para um único NV com 14N (S=1,I=1 -> 9 níveis).
    """
    Hmw = build_H_mw_14N(Bmw_G=mw_field_G_nv)

    B_nv = dc_field_nv_frame(B_amp_G, polar_angle_rad)
    Hgs = build_H_gs_14N(Egs_MHz=strain_E_MHz, B_G=B_nv)

    evals, evecs = eigh(Hgs)

    esr = np.zeros_like(frequency_axis_MHz, dtype=float)
    dim = len(evals)

    for i in range(dim):
        vi = evecs[:, i]
        for j in range(i + 1, dim):
            vj = evecs[:, j]
            f0 = abs(evals[j] - evals[i])  # MHz
            strength = abs(np.vdot(vj, Hmw @ vi))**2
            if strength > 0:
                esr += strength * lorentz_absorption(frequency_axis_MHz, f0, linewidth_MHz)

    return esr


def esr_map_single_nv_14N(
    B_axis_G: np.ndarray,
    f_axis_MHz: np.ndarray,
    polar_angle_rad: float,
    strain_E_MHz: float,
    linewidth_MHz: float,
    mw_field_G_nv: tuple[float, float, float] = (1.0, 0.0, 0.0),
) -> np.ndarray:
    """
    Retorna matriz ESR_map com shape (N_f, N_B) para usar com imshow:
      x -> B, y -> f
    """
    esr_map = np.zeros((len(f_axis_MHz), len(B_axis_G)), dtype=float)

    for k, B in enumerate(B_axis_G):
        esr_map[:, k] = esr_spectrum_single_nv_14N(
            frequency_axis_MHz=f_axis_MHz,
            B_amp_G=B,
            polar_angle_rad=polar_angle_rad,
            strain_E_MHz=strain_E_MHz,
            linewidth_MHz=linewidth_MHz,
            mw_field_G_nv=mw_field_G_nv,
        )

    return esr_map


def plot_esr_heatmap_with_three_orientations(
    B_max_G: float = 60.0,
    n_B: int = 241,
    f_min_MHz: float = 2680.0,
    f_max_MHz: float = 3050.0,
    n_f: int = 1200,
    strain_E_MHz: float = 5.0,
    linewidth_MHz: float = 1.0,
    mw_field_G_nv: tuple[float, float, float] = (1.0, 0.0, 0.0),
    combine_mode: str = "sum",  # "sum" ou "max"
):
  
      # um heatmap de intensidade (combinando θ=0, π/3, π/2)
      

    B_axis = np.linspace(0.0, B_max_G, n_B)
    f_axis = np.linspace(f_min_MHz, f_max_MHz, n_f)

    angles = [0.0, np.pi/3, np.pi/2]
    labels = [r"$\theta=0$", r"$\theta=\pi/3$", r"$\theta=\pi/2$"]
    line_colors = ["k", "g", "r"]

    # calcula mapa para cada direção
    maps = []
    for theta in angles:
        maps.append(
            esr_map_single_nv_14N(
                B_axis_G=B_axis,
                f_axis_MHz=f_axis,
                polar_angle_rad=theta,
                strain_E_MHz=strain_E_MHz,
                linewidth_MHz=linewidth_MHz,
                mw_field_G_nv=mw_field_G_nv,
            )
        )

    # combina em um único heatmap (como na 1ª figura)
    if combine_mode == "sum":
        esr_combined = np.sum(maps, axis=0)
    elif combine_mode == "max":
        esr_combined = np.max(np.stack(maps, axis=0), axis=0)
    else:
        raise ValueError("combine_mode deve ser 'sum' ou 'max'.")

    # normaliza só para visualização (mantém dinâmica boa)
    esr_combined = esr_combined / np.max(esr_combined)

    plt.figure(figsize=(8.5, 5.5))
    im = plt.imshow(
        esr_combined,
        extent=[B_axis.min(), B_axis.max(), f_axis.min(), f_axis.max()],
        aspect="auto",
        origin="lower",
    )

    # Linhas teóricas: f± = D ± sqrt((γ B cosθ)^2 + E^2)
    for theta, lab, c in zip(angles, labels, line_colors):
        f_plus = D_GS_MHz + np.sqrt((GAMMA_E_MHz_per_G * B_axis * np.cos(theta))**2 + strain_E_MHz**2)
        f_minus = D_GS_MHz - np.sqrt((GAMMA_E_MHz_per_G * B_axis * np.cos(theta))**2 + strain_E_MHz**2)

        plt.plot(B_axis, f_plus, linestyle="--", linewidth=2, color=c, label=lab)
        plt.plot(B_axis, f_minus, linestyle="--", linewidth=2, color=c)

    plt.xlabel("B (G)")
    plt.ylabel("Frequency (MHz)")
    plt.title(r"ESR intensity map (NV-$^{14}$N) with three field orientations overlaid")
    plt.legend(loc="upper left")
    plt.colorbar(im, label="Normalized ESR (a.u.)")
    plt.tight_layout()
    plt.show()


plot_esr_heatmap_with_three_orientations(
    B_max_G=60.0,
    n_B=241,
    f_min_MHz=2680.0,
    f_max_MHz=3050.0,
    n_f=1200,
    strain_E_MHz=5.0,
    linewidth_MHz=1.0,
    mw_field_G_nv=(1.0, 0.0, 0.0),
    combine_mode="sum",   # experimente também "max"
)



# %%
