import numpy as np

def uniformSphere(maxRad):
    '''
    Samples 3D random vector uniformly in a sphere of rad maxRad
    '''
    rr = np.random.rand()
    th = 2*np.random.rand() - 1.0
    ph = 2 * np.pi *np.random.rand()
    rr = maxRad * rr**(1.0/3.0)
    th = np.arccos(th)
    result = np.zeros(3)
    result[0] = rr * np.sin(th) * np.cos(ph)
    result[1] = rr * np.sin(th) * np.sin(ph)
    result[2] = rr * np.cos(th)
    return result

def uniformShell(minRad, maxRad):
    '''
    Samples 3D random vector uniformly in a spherical shell between minRad and maxRad
    '''
    rr = np.random.uniform(minRad**3, maxRad**3)
    th = 2 * np.random.rand() - 1.0
    ph = 2 * np.pi * np.random.rand()
    rr = rr ** (1.0 / 3.0)
    th = np.arccos(th)
    result = np.zeros(3)
    result[0] = rr * np.sin(th) * np.cos(ph)
    result[1] = rr * np.sin(th) * np.sin(ph)
    result[2] = rr * np.cos(th)
    return result