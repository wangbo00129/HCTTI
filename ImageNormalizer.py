
import sys
import os
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import cv2
import numpy as np
import cupy as cp
import PIL
from Utils import cost_time

from mpi4py import MPI
MPI_COMM_WORLD_RANK = MPI.COMM_WORLD.Get_rank()  # The process ID (integer 0-3 for 4-process run)
MPI_COMM_WORLD_SIZE = MPI.COMM_WORLD.Get_size()

class NullNormalizer():
    '''
    Normalizer but do nothing.
    '''
    @cost_time
    def __init__(self, target_input, **kargs) -> None:
        pass
    
    @cost_time
    def transform(self, array):
        array = np.array(array)
        return array
    
class TiatoolboxMacenkoNormalizer(NullNormalizer):
    '''
    Encapsulates Tiatoolbox get_normaliser.
    '''
    @cost_time
    def __init__(self, target_input, **kargs) -> None:
        import tiatoolbox.tools.stainnorm as sn
        from tiatoolbox import utils
        self.normaliser = sn.get_normalizer(method_name=kargs.get('method_name', 'macenko'), stain_matrix=kargs.get('stain_matrix'))
        self.normaliser.fit(utils.misc.imread(target_input))

    @cost_time
    def transform(self, array):
        if isinstance(array, PIL.Image.Image):
            array = np.array(array)
        elif isinstance(array, cp.ndarray):
            array = array.get()
        return self.normaliser.transform(array)

class TorchstainMacenkoNormalizer(NullNormalizer):
    '''
    By torchstain
    '''
    @cost_time
    def __init__(self, target_input, **kargs) -> None:
        # example from github 
        import torchstain
        import torch
        from torchvision import transforms
        
        rank = MPI_COMM_WORLD_RANK
        print('using {} as normalization target'.format(target_input))
        target = cv2.cvtColor(cv2.imread(target_input), cv2.COLOR_BGR2RGB)

        self.T = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x*255)
        ])
        self.device = 'cuda:{}'.format(rank) # torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('device', self.device, type(self.device))

        self.torch_normalizer = torchstain.normalizers.MacenkoNormalizer(backend='torch')

        self.torch_normalizer.fit(self.T(target).to(self.device))
    
    @cost_time
    def transform(self, array):
        to_transform = array
        if isinstance(to_transform, cp.ndarray):
            # https://blog.csdn.net/l297969586/article/details/102824246
            t_to_transform = self.T(torch.as_tensor(to_transform)).to(self.device)
        elif isinstance(to_transform, np.ndarray):
            t_to_transform = self.T(to_transform).to(self.device)
        else:
            raise Exception('to_transform should be cp.ndarray or np.ndarray, got {}'.format(type(to_transform)))

        print(t_to_transform)
        norm, H, E = self.torch_normalizer.normalize(I=t_to_transform, stains=True)
        norm = norm.to('cpu').numpy().astype('uint8')
        # print('transform returns a ', type(norm), norm.shape)
        return norm
    
class MacenkoNormalizer(NullNormalizer):
    '''
    By ChenYuhang243, https://github.com/ChenYuhang243/stain-normalizer
    '''
    @cost_time
    def __init__(self, target_input, **kargs) -> None:
        with cp.cuda.Device(int(MPI_COMM_WORLD_RANK)):
            from MacenkoNormalizer import Normalizer
            self.normalizer = Normalizer()
            img_tmpl = cv2.cvtColor(cv2.imread(target_input), cv2.COLOR_BGR2RGB)
            img_tmpl = cp.asarray(img_tmpl)
            self.normalizer.fit(img_tmpl)
        
    @cost_time
    def transform(self, array):
        with cp.cuda.Device(int(MPI_COMM_WORLD_RANK)):
            if isinstance(array, cp.ndarray):
                array_with_expanded_dims = cp.expand_dims(array, 0)
            elif isinstance(array, np.ndarray):
                array_with_expanded_dims = np.expand_dims(array, 0)
            else:
                raise Exception('array should be cp.ndarray or np.ndarray, got {}'.format(type(dtype(array))))

            array_with_expanded_dims = cp.asarray(array_with_expanded_dims)

            transformed = self.normalizer.normalize(array_with_expanded_dims)[0].get()
            return transformed

class NumpyMacenkoNormalizer(NullNormalizer):
    '''
    By ChenYuhang243, https://github.com/ChenYuhang243/stain-normalizer
    '''
    @cost_time
    def __init__(self, target_input, **kargs) -> None:
        from NumpyMacenkoNormalizer import Normalizer
        self.normalizer = Normalizer()
        img_tmpl = cv2.cvtColor(cv2.imread(target_input), cv2.COLOR_BGR2RGB)
        self.normalizer.fit(img_tmpl)
        
    @cost_time
    def transform(self, array):
        if isinstance(array, cp.ndarray):
            array = array.get()
        transformed = self.normalizer.normalize(np.expand_dims(array, 0))[0] # .get()
        return transformed
    
class ReinhardNormalizer(NullNormalizer):
    @cost_time
    def __init__(self, target_input, **kargs) -> None:
        with cp.cuda.Device(int(MPI_COMM_WORLD_RANK)):
            from ReinhardNormalizer import CupyReinhardNormalizer
            import cupy as cp
            self.normalizer = CupyReinhardNormalizer()
            img_tmpl = cv2.cvtColor(cv2.imread(target_input), cv2.COLOR_BGR2RGB)
            self.normalizer.fit(cp.array(img_tmpl))

    @cost_time
    def transform(self, array):
        with cp.cuda.Device(int(MPI_COMM_WORLD_RANK)):
            import cupy as cp
            transformed = self.normalizer.normalize(cp.array(array)).get()
            return transformed

class RuifrokNormalizer(NullNormalizer):
    @cost_time
    def __init__(self, target_input, **kargs) -> None:
        with cp.cuda.Device(int(MPI_COMM_WORLD_RANK)):
            from RuifrokNormaliser import RuifrokNormaliser
            self.normalizer = RuifrokNormaliser()
            img_tmpl = cv2.cvtColor(cv2.imread(target_input), cv2.COLOR_BGR2RGB)
            self.normalizer.fit(cp.array(img_tmpl))

    @cost_time
    def transform(self, array):
        with cp.cuda.Device(int(MPI_COMM_WORLD_RANK)):
            transformed = self.normalizer.transform(cp.array(array)).get()
            return transformed
