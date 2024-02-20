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

filename = 'dat_kstar1_13_kstar2_10_13_SFstart_10000.0_SFduration_10000.0_metallicity_0.02.h5'
key = 'conv'

NS_binaries = pd.read_hdf(filename, key)
m_samp = np.max(pd.read_hdf(filename, 'mass_stars'))[0]
NS_binaries = NS_binaries.loc[((NS_binaries.kstar_1 == 13) & (NS_binaries.kstar_2.isin([10,11,12,13,14]))) | 
                              ((NS_binaries.kstar_2 == 13) & (NS_binaries.kstar_1.isin([10,11,12,13,14])))]

len(NS_binaries), 1e10/m_samp

plt.hist(np.log10(NS_binaries.tphys))


# # B-field vs. Spin Period Graph

def death_line(spin_period):
    return 0.17e12 * spin_period**2


NS_binaries['spin_period_1'] = 1/(NS_binaries.omega_spin_1.values * u.yr**(-1)).to(u.s**(-1))

NS_binaries['pulsar_death'] = np.where(
    (NS_binaries.B_1.values * u.Gauss) / ((NS_binaries.spin_period_1.values * u.s)**2) > 0.17e12 * u.Gauss * u.s**(-2), 1, 0
)

# +
plt.scatter(NS_binaries.loc[NS_binaries.pulsar_death == 1].spin_period_1, 
            NS_binaries.loc[NS_binaries.pulsar_death == 1].B_1, 
            label = 'Radio Detectable')
plt.scatter(NS_binaries.loc[NS_binaries.pulsar_death == 0].spin_period_1, 
            NS_binaries.loc[NS_binaries.pulsar_death == 0].B_1, 
            label = 'Not Radio Detectable')


spin_period = np.linspace(10**-5, 10, 100) 
deathline_B = death_line(spin_period)
plt.plot(spin_period, deathline_B, label = 'Pulsar Death Line', linestyle = '--', color = 'black')

plt.legend()
plt.xscale('log')
plt.yscale('log')

plt.ylim(1e7, 2e13)
plt.xlim(1e-5, 10)
plt.xlabel('Spin Period [s]')
plt.ylabel('Magnetic Field [G]')

plt.tight_layout()
plt.savefig('Pulsar Death Line')
# -

# # Chirp vs. Orbital Frequency Graph

# Chirp Mass

# +
# Variables

m1 = NS_binaries['mass_1'].values
m2 = NS_binaries['mass_2'].values

# mass_1 (primary mass) and mass_2 (secondary mass)

# +
# Equation

def chirp_mass(m1, m2):
    Mc = ((m1*m2)**(3/5))/((m1+m2)**(1/5))
    return Mc #in kg


# -

Mc = chirp_mass(m1, m2) #kg

# Chirp Equation

import legwork.utils as utils
from astropy import units as u

plt.hist(NS_binaries.ecc)

f_dot = utils.fn_dot(
    utils.chirp_mass(m1 * u.Msun, m2 * u.Msun), f_orb=1/(NS_binaries.porb.values * u.day), 
    e = NS_binaries.ecc.values, n=2)

f_dot.to(u.s**(-2))

# Graph

# LISA observation time is 10 years
t_obs = 10 * u.yr

np.where(f_dot.to(u.s**(-2)).value > 1/((t_obs.to(u.s).value)**2))

# +
plt.scatter((1/(NS_binaries.porb.values * u.day)).to(1/u.s), 
            f_dot.to(u.s**(-2)), s=5, 
            c=np.log10(NS_binaries.B_1/NS_binaries.spin_period_1**2), vmin=np.log10(0.177e12))
cbar = plt.colorbar()
cbar.set_label(r'chirp mass [M$_{\odot}$]', size=14)

plt.xlim(1e-7, 1e-1)
plt.ylim(1e-26, 1e-11)

plt.xscale('log')
plt.yscale('log')

plt.xlabel(r'Orbital Frequency [$Hz$]', size = 14)
plt.ylabel(r'Chirp [Hz/yr]', size = 14 )

# Horizontal line of the chirp limit, f_dot = 1/t_obs**2
plt.axhline(y = 1/((t_obs.to(u.s).value)**2), linewidth = 2, linestyle = '--', color = 'black', label = 'Chirp Limit') 

plt.legend()

plt.tight_layout()
plt.savefig('Chirp vs Orbital Period')
# -

# # Subset of population that is above the death line and above the chirp limit
# That's the detectable part of the population where it's above death line, above chirp limit, above LISA limit, below beaming fraction

# Above Death Line

NS_binaries["P"] = spin_period_1 # Add Spin Period to Dataset
NS_binaries["B-field"] = B_1 # Add B-field to Dataset

NS_binaries["above_deathline"] = kiel_equation(NS_binaries["B_1"], NS_binaries["P"]) # Pop that's > death line

# Above Chirp Limit

NS_binaries["chirp"] = f_dot # Added Chirp to Dataset
data_above_chirp_limit = NS_binaries[NS_binaries.chirp > 1/(t_obs**2)] # Pop that's > chirp limit

# +
# NS_binaries[NS_binaries.chirp < t_obs]
# -

# Detectable binary systems

# Binaries above chirp limit and above death line
Chirp_deathline_detectable_binaries = NS_binaries[(NS_binaries.chirp > 1/(t_obs**2)) & (NS_binaries.above_deathline)]

print(Chirp_deathline_detectable_binaries)

# NS_binaries.chirp < t_obs

# + active=""
# Note: Ran this cell to delete the "P_true" column I added by mistake and do not need it anymore
# NS_binaries.drop(columns="P_true",inplace=True)
# -

# # Beaming Fraction vs. Spin Period Graph

# Beaming Fraction Equation

# +
# Variables

P = 2*pi/(NS_binaries['omega_spin_1'].values/3.154e7) #seconds

# omega_spin_1 (spin period)

# +
# Equation

def Beaming_frac(P):
    Beam_frac = 0.09*((np.log10(P/10))**2) + 0.03
    return Beam_frac 


# -

Beam_frac = Beaming_frac(P)

# Graph

P_low = P.min()
P_high = P.max()

# +
plt.scatter(P, Beam_frac)

plt.xlim(P_low, P_high)

plt.xscale('log')
plt.yscale('log')

plt.xlabel(r'Spin Period [s]', size = 14)
plt.ylabel(r'Beaming Fraction', size = 14 )

plt.legend()

plt.tight_layout()
plt.savefig('Beaming Fraction vs Spin Period')
# -

# # Population Size based on Death Line, Chirp, & Beaming Fraction Cuts 

# Calculate Beaming Fraction
Test_Beaming_Fraction = np.random.uniform(0, 1, 1510) # 1510: number of binaries that made the chirp & death line cut

print(Test_Beaming_Fraction) #How can I print the whole table. Does this array not contain all the data?

# For every datapoint, does it make the beaming fraction cut?
Beam_frac = Beaming_frac(Chirp_deathline_detectable_binaries["P"])

print(Beam_frac)

Chirp_deathline_detectable_binaries["Beam_frac"] = Beaming_frac(Chirp_deathline_detectable_binaries["P"])

print(Chirp_deathline_detectable_binaries)

Chirp_deathline_detectable_binaries = Chirp_deathline_detectable_binaries[(Test_Beaming_Fraction <
                                                                           Chirp_deathline_detectable_binaries.Beam_frac)]

Chirp_deathline_detectable_binaries

# xGx and yGx are the x and y axes of the Galaxy
plt.scatter(Chirp_deathline_detectable_binaries.xGx, Chirp_deathline_detectable_binaries.yGx)
plt.scatter(8.3, 0) # 8.3 is the position of the Sun
# This is the line-of-sight drawing from NE2001 but *MORE ACCURATE!!!!*
# x (kpc) and y (kpc) "These are the Cartesian Galactocentric positions"
