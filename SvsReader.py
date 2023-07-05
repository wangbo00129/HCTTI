#!/usr/bin/env python
import numpy as np
from Utils import cost_time

from mpi4py import MPI
MPI_COMM_WORLD_RANK = MPI.COMM_WORLD.Get_rank()  # The process ID (integer 0-3 for 4-process run)
MPI_COMM_WORLD_SIZE = MPI.COMM_WORLD.Get_size()

class SvsReader():
    '''
    To make the methods of cucim or openslide the same. 
    '''
    @cost_time
    def __init__(self, path_svs, use_cucim=True, discard_page_cache=False) -> None:
        self.use_cucim = use_cucim
        if self.use_cucim:
            import cucim
            if discard_page_cache:
                assert cucim.clara.filesystem.discard_page_cache(path_svs), 'discard_page_cache for {} failed'.format(path_svs)
            self.image = cucim.CuImage(path_svs)
        else:
            import openslide
            self.image = openslide.open_slide(path_svs)
        print(self.image)
    
    @cost_time
    def read_region(self, xy, patch_size_xy, **kargs):
        if self.use_cucim:
            if 'device' in kargs:
                if kargs['device'] == 'cuda':
                    kargs['device'] = 'cuda:{}'.format(MPI_COMM_WORLD_RANK)
            ret = self.image.read_region(xy, patch_size_xy, 0, **kargs)  #.convert('RGB') # convert is not needed since the CuImage is 3-channel.
            # ret = np.array(ret)
            # if device='cuda', cv2 parts need to be replace by gpu version.
        else:
            ret = self.image.read_region(xy,0,patch_size_xy).convert('RGB')
        # print(type(ret))
        return ret
    
    def get_dimensions(self):
        if self.use_cucim:
            [n, m] = self.image.shape[:2]
        else:
            [m, n] = self.image.dimensions    
        return [m,n]

if __name__ == '__main__':
    svs_reader = SvsReader('/home/wangb/.conda/envs/cucim_env/lib/python3.9/site-packages/tests/data/albers27.tif', True)
    x = svs_reader.read_region((0,0),(15,52))
    print(np.array(x))
