# ODMR.py
# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh


from ESR import build_H_gs_14N, build_H_mw_14N, lorentz_absorption


def _unit(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError("Vetor nulo não pode ser normalizado.")
    return v / n


def B_to_NV_frame(B_lab, nv_axis):
    """
    Converte um vetor de campo no LAB para componentes (Bx,By,Bz)
    no referencial do NV onde z || nv_axis.
    (Implementa a ideia do Eq. 3.1: BNVi = Bin · TNVi) :contentReference[oaicite:9]{index=9}
    """
    z = _unit(nv_axis)

    ref = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(ref, z)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])

    x = _unit(np.cross(ref, z))
    y = np.cross(z, x)

    B_lab = np.asarray(B_lab, dtype=float)
    return (float(np.dot(B_lab, x)), float(np.dot(B_lab, y)), float(np.dot(B_lab, z)))


def esr_transition_spectrum_single_nv(
    w, *,
    Bvec_G_nv=(0.0, 0.0, 0.0),
    Egs_MHz=0.2,
    linewidth_MHz=1.6,
    Bmw_G_nv=(1.0, 0.0, 0.0)
):
    """
    Calcula ESR(ϖ) (Eq. 3.3) para UM NV com campo já expresso no frame do NV. :contentReference[oaicite:10]{index=10}
    """
    Hgs = build_H_gs_14N(Egs_MHz=Egs_MHz, B_G=Bvec_G_nv)
    evals, evecs = eigh(Hgs)

    Hmw = build_H_mw_14N(Bmw_G=Bmw_G_nv)

    S = np.zeros_like(w, dtype=float)
    for i in range(len(evals)):
        vi = evecs[:, i]
        for f in range(i + 1, len(evals)):
            vf = evecs[:, f]
            w0 = abs(evals[f] - evals[i])
            Tif = abs(np.vdot(vf, Hmw @ vi)) ** 2
            S += Tif * lorentz_absorption(w, w0, linewidth_MHz)

    return S


def odmr_ensemble_14N(
    *,
    B_G=50.0,               # 50 G = 5 mT
    B_dir_lab=(0.0, 0.0, 1.0),
    Egs_MHz=0.2,
    linewidth_MHz=1.6,
    omega=0.004,            # contraste ω da Eq. 3.4
    R=1.0,                  # taxa de fótons (a.u.)
    wmin=2845.0, wmax=2866.0, npts=6000,
    Bmw_G_nv=(1.0, 0.0, 0.0),
    include_VN=True
):
    """
    - Ensemble: soma sobre orientações NV (e opcionalmente VN), como descrito no texto. :contentReference[oaicite:11]{index=11}
    - ODMR: I(ϖ)=R[1-ω ESR(ϖ)] (Eq. 3.4). :contentReference[oaicite:12]{index=12}
    """
    w = np.linspace(wmin, wmax, npts)

    B_lab = B_G * _unit(B_dir_lab)

    # quatro eixos cristalográficos do ensemble (subseção 2.5) :contentReference[oaicite:13]{index=13}
    nv_axes = [
        _unit(( 1,  1,  1)),
        _unit((-1, -1,  1)),
        _unit(( 1, -1, -1)),
        _unit((-1,  1, -1)),
    ]

    axes = nv_axes + ([-a for a in nv_axes] if include_VN else [])

    ESR_total = np.zeros_like(w, dtype=float)

    for axis in axes:
        Bx, By, Bz = B_to_NV_frame(B_lab, axis)
        ESR_total += esr_transition_spectrum_single_nv(
            w,
            Bvec_G_nv=(Bx, By, Bz),
            Egs_MHz=Egs_MHz,
            linewidth_MHz=linewidth_MHz,
            Bmw_G_nv=Bmw_G_nv
        )

    ESR_total /= len(axes)

    # normaliza o "ESR" para que ω controle a profundidade como no ajuste do TCC
    if ESR_total.max() > 0:
        ESRn = ESR_total / ESR_total.max()
    else:
        ESRn = ESR_total

    # Eq. 3.4 :contentReference[oaicite:14]{index=14}
    I = R * (1.0 - omega * ESRn)
    return w, I

# %% 
if __name__ == "__main__":
    # Parâmetros da Fig. 12 :contentReference[oaicite:15]{index=15}
    w, I = odmr_ensemble_14N(
        B_G=50.0,
        B_dir_lab=(0.0, 0.0, 1.0),
        Egs_MHz=0.2,
        linewidth_MHz=1.6,
        omega=0.004,
        R=1.0,
        wmin=2600.0, wmax=3100.0, npts=8000,
        include_VN=True
    )



    plt.figure(figsize=(7.2, 4.8))
    plt.plot(w, I, lw=2)
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("ODMR Contrast Intensity (a.u.)")
    plt.tight_layout()
    plt.show()

# %%
