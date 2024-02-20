import numpy as np
import astropy.units as u
from legwork import utils, source

def get_psr_properties(dat):
    '''Select the magnetic field, spin period, and pulsar death line masks
    
    Parameters
    ----------
    dat : `pandas.DataFrame`
        dataframe of cosmic outputs, in cosmic units
    
    Returns
    -------
    B_field : `numpy.array`
        present day magentic field of the pulsar according to evolution following Kiel+2008 in units of Gauss
        
    spin_period : `numpy.array`
        present day pulsar spin period in units of s^(-1)
        
    death_line_mask : `numpy.array`
        0 if below death line, 1 if above deathline
        
    lum_psr : numpy.array`
        luminosity of pulsar drawn from log-normal distribution following Szary+2014
    '''
    
    def death_line(spin_period):
        return 0.17e12 * spin_period**2

    B_field = np.where(dat.kstar_1 == 13, dat.B_1.values, dat.B_2.values)
    spin_period = np.where(dat.kstar_1 == 13, 
                           1/(dat.omega_spin_1.values * u.yr**(-1)).to(u.s**(-1)).value,
                           1/(dat.omega_spin_2.values * u.yr**(-1)).to(u.s**(-1)).value)
    
    death_line_mask = np.where(B_field > death_line(spin_period), 1, 0)
    log_lum_psr = np.random.normal(0.5, 1.0, len(B_field))
    
    dat['B_field'] = B_field
    dat['spin_period'] = spin_period
    dat['death_line_mask'] = death_line_mask
    dat['log_lum_psr'] = log_lum_psr
    return dat


                                    