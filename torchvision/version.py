__version__ = '0.4.1'
git_version = 'd94043ad5592ba4eba5cdb61062deee5bb19bcb3'
from torchvision import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
