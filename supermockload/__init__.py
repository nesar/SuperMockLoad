"""SuperMockLoad -- load & plot SPHEREx SuperMock lightcone catalogs.

    from supermockload import SuperMock, observations, plots
    sm = SuperMock(3)                 # a patch (catalog only; SEDs opt-in)
    plots.gsmf(sm)                    # report-style panels vs observations
"""
from .loader import SuperMock, available_patches, BAND_NAMES
from . import observations
from . import plots

__version__ = '0.1.0'
__all__ = ['SuperMock', 'available_patches', 'BAND_NAMES',
           'observations', 'plots']
