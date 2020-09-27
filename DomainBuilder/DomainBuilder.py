
from DomainBuilder.DiscretisationMethods import FinDiff, FinEl

class DomainBuilder:
    """Builds domain onto grid, for example a diseased tree or freespace
    Attributes
    ----------
        grid: object
            grid dimension parameters and discretisation method
    Methods
    ----------
    BuildSource
        returns domain with desired attributes
    Notes
    ----------  """
    def __init__(self, grid):
        method = grid['discretisationMethod']
        if method == 'FinDiff':
            self.builder = FinDiff(grid)
        if method == 'FinEl':
            self.builder = FinEl(grid)

    def BuildDomain(self, domain, diseasetype=None, healthyeps=None, diseaseeps=None, treerad=None, yshift = 0, xshift = 0):
        return self.builder.BuildDomain(domain, diseasetype, healthyeps, diseaseeps, treerad, yshift, xshift)
