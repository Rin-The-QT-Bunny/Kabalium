import numpy as np
import matplotlib.pyplot as plt
from scipy.special import sph_harm
from scipy.special import assoc_laguerre


def hydrogen_cloud(n,l,m):
    x = np.linspace(-30, 30, 500)
    y = 0  #### the plane locates at y = 0 
    z = np.linspace(-35, 35, 500)
    X, Z = np.meshgrid(x, z)

    rho = np.linalg.norm((X,y,Z), axis=0) / n
    Lag = assoc_laguerre(2 * rho, n - l - 1, 2 * l + 1)
    Ylm  = sph_harm(m, l, np.arctan2(y,X), np.arctan2(np.linalg.norm((X,y), axis=0), Z))
    Psi = np.exp(-rho) * np.power((2*rho),l) * Lag * Ylm
    
    density = np.conjugate(Psi) * Psi
    #### visualization
    fig, ax = plt.subplots(figsize=(10,10))
    ax.imshow(density.real, extent=[-30, 30, -35, 35])
    plt.show()
    fig.savefig("H_psi420.png", dpi=300)