import warnings

warnings.simplefilter('ignore')

from ._model import CellDISECT
from ._module import CellDISECTModule

from importlib.metadata import version

package_name = "celldisect"
__version__ = version(package_name)

__all__ = [
    "CellDISECT",
    "CellDISECTModule",
]