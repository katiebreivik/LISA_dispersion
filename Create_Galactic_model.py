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

# +
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import legwork.utils as utils
import legwork.source as source

from astropy import units as u
from cosmic.sample.initialbinarytable import InitialBinaryTable
from cosmic.evolve import Evolve


# +
def assign_age(n_pop):
    return np.random.uniform(0,10000,n_pop)
    
def data_read(path, m_disk):
    # read in the data
    filename = '{}/dat_kstar1_13_kstar2_10_13_SFstart_13700.0_SFduration_0.0_metallicity_0.02.h5'.format(path)
    key = 'bpp'
    
    NS_binaries = pd.read_hdf(filename, 'bpp')
    initC = pd.read_hdf(filename, 'initCond')
    NS_binaries = NS_binaries.loc[((NS_binaries.kstar_1 == 13) & (NS_binaries.kstar_2.isin([10,11,12,13]))) | 
                              ((NS_binaries.kstar_2 == 13) & (NS_binaries.kstar_1.isin([10,11,12,13])))]
    
    m_samp = np.max(pd.read_hdf(filename, 'mass_stars'))[0]
    
    n_pop = len(NS_binaries)/m_samp * m_disk
    n_int = int(n_pop)
    n_dec = n_pop - n_int
    n_dec_select = np.random.uniform(0, 1)
    if n_dec_select <= n_dec:
        n_pop = n_int + 1
    else:
        n_pop = n_int
    initC = initC.loc[initC.bin_num.isin(NS_binaries.bin_num)]
    return NS_binaries, initC, m_samp, n_pop


# -

# ### I already ran these cells to produce: NS_binaries.h5

# +
#This is the thin disk from McMillan+2011
NS_binaries, initC, m_samp, n_pop = data_read(path='./', m_disk=4.32e10)
    
# resample the population and assign a random age
NS_pop = NS_binaries.sample(n_pop, replace=True)
NS_pop['age'] = assign_age(n_pop)

# filter out anything whose age is less than the time the NS binary is born
NS_pop = NS_pop.loc[NS_pop.age - NS_pop.tphys > 0]

# -

bin_num_counts = NS_pop.bin_num.value_counts().sort_index()

initC = pd.DataFrame((np.repeat(initC.loc[initC.bin_num.isin(bin_num_counts.index)].to_numpy(), bin_num_counts.values, axis=0)), columns=initC.columns)

initC['tphysf'] = NS_pop.sort_values('bin_num').age.values

bpp_rerun, bcm_rerun, initC_rerun, kick_info_rerun = Evolve.evolve(initialbinarytable=initC, 
                                                                   BSEDict={}, nproc=28)


bcm_rerun = bpp_rerun.groupby('bin_num').last().reset_index()

# +
bpp_rerun.to_hdf('NS_binaries.h5', key='bpp')
bcm_rerun.to_hdf('NS_binaries.h5', key='bcm')
initC_rerun.to_hdf('NS_binaries.h5', key='initC')

#bpp_rerun = pd.read_hdf('NS_binaries.h5', key='bpp')
#bcm_rerun = pd.read_hdf('NS_binaries.h5', key='bcm')
#initC_rerun = pd.read_hdf('NS_binaries.h5', key='initC')
# -

import detect_psr_gw as dpg

B_field, spin_period, death_line_mask = 

bcm_rerun['B_field'] = B_field

f_dot = utils.fn_dot(
    utils.chirp_mass(bcm_rerun.mass_1.values * u.Msun, bcm_rerun.mass_1.values * u.Msun), 
    f_orb=1/(bcm_rerun.porb.values * u.day), 
    e = bcm_rerun.ecc.values, n=2)

bcm_rerun['f_dot'] = f_dot.to(u.s**(-2))

bcm_rerun['pulsar_death'] = np.where(
    (bcm_rerun.B_field.values * u.Gauss) / ((bcm_rerun.spin_period.values * u.s)**2) > 0.17e12 * u.Gauss * u.s**(-2), 1, 0
)

bcm_rerun['chirping'] = np.where(
    bcm_rerun.f_dot.values > 1/(4 * 3.155e7)**2, 1, 0
)


def death_line(spin_period):
    return 0.17e12 * spin_period**2


# +
plt.scatter(bcm_rerun.loc[bcm_rerun.chirping == 0].spin_period, 
            bcm_rerun.loc[bcm_rerun.chirping == 0].B_field, 
            label = 'non-measured distance', c='salmon')
plt.scatter(bcm_rerun.loc[bcm_rerun.chirping == 1].spin_period, 
            bcm_rerun.loc[bcm_rerun.chirping == 1].B_field, 
            label = 'measured distance', c='teal')

spin_period = np.linspace(10**-5, 10, 100) 
deathline_B = death_line(spin_period)
plt.plot(spin_period, deathline_B, label = 'Pulsar Death Line', linestyle = '--', color = 'black')

plt.legend()
plt.xscale('log')
plt.yscale('log')

plt.ylim(1e7, 5e13)
plt.xlim(1e-5, 10)
plt.xlabel('Spin Period [s]')
plt.ylabel('Magnetic Field [G]')

plt.tight_layout()
plt.savefig('Pulsar Death Line')


# -

def assign_distances(dat):
    


sources = source.Source(m_1=bcm_rerun.mass_1 * u.Msun, 
                        m_2=bcm_rerun.mass_2 * u.Msun, 
                        ecc=bcm_rerun.ecc, 
                        dist=bcm_rerun.dist * u.kpc, 
                        f_orb=1/(bcm_rerun.porb * u.day))



