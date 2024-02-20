# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import seaborn as sns
import tqdm
from cosmic.evolve import Evolve
from schwimmbad import MultiPool
from matplotlib import colors
import legwork as lw

# +
met_grid = [0.0001, 0.00015029, 0.00022588, 0.00033948, 0.00051021, 0.00076681,
            0.00115245, 0.00173205, 0.00260314, 0.00391233, 0.00587993, 0.0088371, 
            0.01328149, 0.01996108, 0.03]
key = 'conv'

met_edges = np.logspace(np.log10(1e-4), np.log10(0.03), 15)
met_edges = np.round(met_edges, 8)
met_edges = np.append(0.0, met_edges)
met_edges = np.append(met_edges, 15.0)

columns = ['bin_num', 'tphys', 'mass_1', 'mass_2', 'kstar_1', 'kstar_2', 'sep',
       'porb', 'ecc', 'RRLO_1', 'RRLO_2', 'evol_type', 'aj_1', 'aj_2', 'tms_1',
       'tms_2', 'massc_1', 'massc_2', 'rad_1', 'rad_2', 'mass0_1', 'mass0_2',
       'lum_1', 'lum_2', 'teff_1', 'teff_2', 'radc_1', 'radc_2', 'menv_1',
       'menv_2', 'renv_1', 'renv_2', 'omega_spin_1', 'omega_spin_2', 'B_1',
       'B_2', 'bacc_1', 'bacc_2', 'tacc_1', 'tacc_2', 'epoch_1', 'epoch_2',
       'bhspin_1', 'bhspin_2', 't_merge', 't_rlo', 't_evol', 'xGx', 'yGx',
       'zGx', 'kern_len']

# +
dat_LISA_tot = []
for ii, m in enumerate(met_grid[:13]):
    dat = np.load(f'ns_dat_store_{ii}.npy')
    dat = pd.DataFrame(dat, columns=columns)
    
    filename = f'data/dat_kstar1_13_kstar2_10_13_SFstart_13700.0_SFduration_0.0_metallicity_{m}.h5'
    NS_binaries = pd.read_hdf(filename, 'conv')
    
    NS_select = NS_binaries.loc[NS_binaries.bin_num.isin(dat.bin_num.unique())]
    #print(np.sort(dat.bin_num.unique()))
    #print(NS_select[['tphys', 'kstar_1', 'kstar_2', 'porb', 'evol_type', 'bin_num']])
    dat_circ = dat.loc[dat.ecc == 0.0].copy()
    dat_ecc = dat.loc[dat.ecc > 0].copy()
    
    f_orb_ecc, ecc = lw.evol.evol_ecc(
        t_evol=dat_ecc.t_evol.values * u.Myr, n_step=2, m_1=dat_ecc.mass_1.values * u.Msun, 
        m_2=dat_ecc.mass_2.values * u.Msun, a_i=dat_ecc.sep.values * u.Rsun, ecc_i=dat_ecc.ecc.values,
        output_vars=["f_orb", "ecc"]
    )
    
    f_orb_circ = lw.evol.evol_circ(
        t_evol=dat_circ.t_evol.values * u.Myr, n_step=2, m_1=dat_circ.mass_1.values * u.Msun, 
        m_2=dat_circ.mass_2.values * u.Msun, a_i=dat_circ.sep.values * u.Rsun, 
        output_vars=["f_orb"]
    )
    
    dat_ecc['forb_today'] = f_orb_ecc[:,1].value
    dat_circ['forb_today'] = f_orb_circ[:,1].value

    dat_ecc['ecc_today'] = ecc[:,1]
    dat_circ['ecc_today'] = np.zeros(len(dat_circ))

    dat_ecc = dat_ecc.loc[dat_ecc.forb_today > 1e-4]
    dat_circ = dat_circ.loc[dat_circ.forb_today > 1e-4]
    
    #print(dat_ecc.loc[(dat_ecc.t_rlo < dat_ecc.tphys + dat_ecc.t_evol) & (dat_ecc.t_rlo > 0)][['porb', 'ecc', 'ecc_today', 'forb_today', 'RRLO_2']])

    
    if (len(dat_ecc) > 0) & (len(dat_circ) > 0):
        dat_LISA = pd.concat([dat_ecc, dat_circ])
        
    elif len(dat_ecc) > 0:
        dat_LISA = dat_ecc
        
    elif len(dat_circ) > 0:
        dat_LISA = dat_ecc
      
    else:
        print(f'no LISA-band sources for metallicity={m}')
        dat_LISA = []
    
    if len(dat_LISA) > 0:
        dat_LISA_tot.append(dat_LISA)
        print(f'the number of sources in the LISA band for metallicity={m} is {len(dat_LISA)}')
        
dat_LISA_tot = pd.concat(dat_LISA_tot) 
print(f'total number of sources in the LISA band is: {len(dat_LISA_tot)}')

dat_LISA_tot['fdot'] = lw.utils.fn_dot(
    m_c=lw.utils.chirp_mass(dat_LISA_tot.mass_1.values * u.Msun, dat_LISA_tot.mass_2.values * u.Msun),
    f_orb=dat_LISA_tot.forb_today.values * u.Hz, e=dat_LISA_tot.ecc.values, n=2,
).to(u.Hz/u.s).value
# -

chirp_limit = ((1 / (4 * u.yr)) * (1 / u.yr)).to(1/u.s**2)
print(chirp_limit)
plt.scatter(dat_LISA_tot.forb_today, dat_LISA_tot.fdot, c=dat_LISA_tot.t_evol, norm=colors.LogNorm())
#plt.plot(np.array([1e-4, 6e-4]), np.array([1e-4, 6e-4]), ls='--', color='gray')
plt.axhline(chirp_limit.value)
plt.colorbar()
plt.xscale('log')
plt.yscale('log')

plt.scatter(dat_LISA_tot.t_merge, dat_LISA_tot.t_evol + dat_LISA_tot.tphys, c=1/(dat_LISA_tot.porb*86400), norm=colors.LogNorm())

dat = np.load('ns_dat_store_6.npy')
dat = pd.DataFrame(dat, columns=columns)

np.shape(dat), len(columns)

dat_circ = dat.loc[dat.ecc == 0.0].copy()
dat_ecc = dat.loc[dat.ecc > 0].copy()

dat.t_merge

# +
f_orb_ecc = lw.evol.evol_ecc(
    t_evol=dat_ecc.t_evol.values * u.Myr, n_step=2, m_1=dat_ecc.mass_1.values * u.Msun, 
    m_2=dat_ecc.mass_2.values * u.Msun, a_i=dat_ecc.sep.values * u.Rsun, ecc_i=dat_ecc.ecc.values,
    output_vars=["f_orb"]
)

f_orb_circ = lw.evol.evol_circ(
    t_evol=dat_circ.t_evol.values * u.Myr, n_step=2, m_1=dat_circ.mass_1.values * u.Msun, 
    m_2=dat_circ.mass_2.values * u.Msun, a_i=dat_circ.sep.values * u.Rsun, 
    output_vars=["f_orb"]
)


# -



dat_ecc['forb_today'] = f_orb_ecc[:,1].value
dat_circ['forb_today'] = f_orb_circ[:,1].value


dat_ecc['LISA'] = np.zeros(len(dat_ecc))
dat_circ['LISA'] = np.zeros(len(dat_circ))


dat_ecc.loc[dat_ecc.forb_today > 1e-4, 'LISA'] = 1.0
dat_circ.loc[dat_circ.forb_today > 1e-4, 'LISA'] = 1.0

print(len(dat_ecc.loc[dat_ecc.LISA > 0]))
print(len(dat_circ.loc[dat_circ.LISA > 0]))

plt.hist(np.log10(dat_circ.forb_today))
plt.hist(np.log10(dat_ecc.forb_today))


