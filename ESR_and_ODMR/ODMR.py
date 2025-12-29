# %%
import numpy as np
from scipy.linalg import eigh

from ESR import build_H_gs_14N, build_H_mw_14N, lorentz_absorption


def esr_ensemble_14N(
    B_G=0.0,
    B_dir_lab=(0, 0, 1),
    Egs_MHz=0.2,
    linewidth_MHz=1.6,
    wmin=2600, wmax=3100, npts=3000,
    Bmw_G_nv=(1.0, 0, 0),
    include_VN=True
):
    """
    Retorna ESR_total(w) (sem normalização) somado sobre orientações NV.
    """

    w = np.linspace(wmin, wmax, npts)
    ESR_total = np.zeros_like(w)

    # Orientações NV no cristal (como no seu código original)
    nv_axes = np.array([
        (1, 1, 1),
        (1, -1, -1),
        (-1, 1, -1),
        (-1, -1, 1)
    ], dtype=float)
    nv_axes = np.array([v / np.linalg.norm(v) for v in nv_axes])

    # Caso queira incluir vetor de nitrogênio (VN)
    # (mantive a lógica simples; você pode adaptar se seu ESR.py já faz VN)
    axes = nv_axes.copy()

    # Hamiltoniano MW (depende de Bmw)
    Hmw = build_H_mw_14N(Bmw_G=Bmw_G_nv)

    for nv_axis in axes:
        # Campo DC projetado no eixo desse NV
        # Aqui assumimos B_dir_lab já é um vetor unitário no lab.
        Bvec_lab = B_G * np.array(B_dir_lab, dtype=float)
        # componente paralela ao eixo do NV:
        Bpar = np.dot(Bvec_lab, nv_axis)
        # se você quiser manter componentes transversais, teria que rotacionar o Hamiltoniano,
        # mas no seu modelo atual você só usa B no eixo z do NV (equivalente ao Bpar).
        Bvec_nv = (0.0, 0.0, Bpar)

        Hgs = build_H_gs_14N(Egs_MHz=Egs_MHz, B_G=Bvec_nv)
        evals, evecs = eigh(Hgs)

        # calcular todas as transições permitidas |<f|Hmw|i>|^2
        for i in range(len(evals)):
            vi = evecs[:, i]
            for f in range(i + 1, len(evals)):
                vf = evecs[:, f]
                w0 = abs(evals[f] - evals[i])  # MHz

                # força de transição ~ |<f|Hmw|i>|^2
                Tif = abs(np.vdot(vf, Hmw @ vi))**2

                if Tif > 0:
                    ESR_total += Tif * lorentz_absorption(w, w0, linewidth_MHz)

    # Média por número de orientações (opcional)
    ESR_total /= len(axes)

    return w, ESR_total


def odmr_ensemble_14N(
    B_G=0.0,
    B_dir_lab=(0, 0, 1),
    Egs_MHz=0.2,
    linewidth_MHz=1.6,
    wmin=2840, wmax=2900, npts=3000,
    Bmw_G_nv=(1.0, 0, 0),
    omega=0.004,
    R=1.0,
    normalize_mode="global",   # "off" | "global" | "per_curve"
    ESR_ref_max=None,          # usado quando normalize_mode="global"
    power_broadening=False,    # opcional
    broadening_coeff=0.0       # MHz/G^2  (ex: 0.02)
):
    """
    Retorna ODMR I(w) = R * (1 - omega * ESRn(w))

    normalize_mode:
      - "off"      : não normaliza ESR (mais físico para comparar Bmw)
      - "global"   : normaliza por ESR_ref_max (fixo para todas potências)
      - "per_curve": normaliza pelo máximo de cada curva (NÃO recomendado p/ potência MW)
    """

    # se quiser power broadening: linewidth aumenta com Bmw^2
    if power_broadening:
        Bmw_amp = np.linalg.norm(np.array(Bmw_G_nv))
        linewidth_eff = linewidth_MHz + broadening_coeff * (Bmw_amp**2)
    else:
        linewidth_eff = linewidth_MHz

    w, ESR_total = esr_ensemble_14N(
        B_G=B_G,
        B_dir_lab=B_dir_lab,
        Egs_MHz=Egs_MHz,
        linewidth_MHz=linewidth_eff,
        wmin=wmin, wmax=wmax, npts=npts,
        Bmw_G_nv=Bmw_G_nv
    )

    # ---------- NORMALIZAÇÃO CORRIGIDA ----------
    if normalize_mode == "off":
        ESRn = ESR_total

    elif normalize_mode == "per_curve":
        # Isso apaga dependência de potência MW (use só para comparar forma)
        m = ESR_total.max()
        ESRn = ESR_total / m if m > 0 else ESR_total

    elif normalize_mode == "global":
        # Normalização global correta: usa um máximo FIXO
        if ESR_ref_max is None:
            # se não foi fornecido, assume o máximo dessa curva como referência
            # (mas idealmente você passa o máximo do caso MW mais forte)
            ESR_ref_max = ESR_total.max()
        ESRn = ESR_total / ESR_ref_max if ESR_ref_max > 0 else ESR_total

    else:
        raise ValueError("normalize_mode deve ser: 'off', 'global' ou 'per_curve'")

    # ODMR
    I = R * (1.0 - omega * ESRn)

    return w, I, ESR_total
# %%
import matplotlib.pyplot as plt
import numpy as np

Bfix = 0
Bmw_list = [0.2, 1.0, 3.0]

# Primeiro, calcule um ESR_ref_max para normalização global (caso MW mais forte)
_, _, ESR_strong = odmr_ensemble_14N(
    B_G=Bfix, Bmw_G_nv=(max(Bmw_list),0,0),
    normalize_mode="off"
)
ESR_ref_max = ESR_strong.max()

plt.figure(figsize=(7,4))
for Bmw in Bmw_list:
    w, I, ESR = odmr_ensemble_14N(
        B_G=Bfix,
        Bmw_G_nv=(Bmw,0,0),
        omega=0.01,
        normalize_mode="global",
        ESR_ref_max=ESR_ref_max
    )
    plt.plot(w, I, label=f"Bmw={Bmw} G")

plt.xlabel("Frequency (MHz)")
plt.ylabel("ODMR Intensity (a.u.)")
plt.title(f"ODMR: fixed B={Bfix} G, varying MW amplitude")
plt.legend()
plt.tight_layout()
plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt

def odmr_map_B_vs_freq(B_list, Bmw, omega=0.01):
    wmin, wmax, npts = 2800, 2900, 2000
    w = np.linspace(wmin, wmax, npts)
    I_map = np.zeros((len(B_list), len(w)))

    # referência para normalização global (pega máximo no maior B do range)
    _, _, ESR_ref = odmr_ensemble_14N(
        B_G=max(B_list),
        Bmw_G_nv=(Bmw,0,0),
        normalize_mode="off",
        wmin=wmin, wmax=wmax, npts=npts
    )
    ESR_ref_max = ESR_ref.max()

    for i, B_G in enumerate(B_list):
        _, I, _ = odmr_ensemble_14N(
            B_G=B_G,
            Bmw_G_nv=(Bmw,0,0),
            omega=omega,
            normalize_mode="global",
            ESR_ref_max=ESR_ref_max,
            wmin=wmin, wmax=wmax, npts=npts
        )
        I_map[i,:] = I

    return w, I_map

B_list = np.linspace(0, 150, 120)
Bmw_list = [0.2, 1.0, 3.0]

for Bmw in Bmw_list:
    w, I_map = odmr_map_B_vs_freq(B_list, Bmw)

    plt.figure(figsize=(8,5))
    plt.imshow(
        I_map,
        extent=[w.min(), w.max(), B_list.min(), B_list.max()],
        aspect="auto",
        origin="lower"
    )
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("B (G)")
    plt.title(f"ODMR map (MW amplitude Bmw = {Bmw} G)")
    plt.colorbar(label="ODMR intensity (a.u.)")
    plt.tight_layout()
    plt.show()

# %%
