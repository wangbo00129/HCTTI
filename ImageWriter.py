import sys
import os
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import cupy as cp
import PIL
import numpy as np
from largedataloader.H5Dataset import H5Dataset
from largedataloader.ZarrDataset import ZarrDataset
import cv2
import PIL.Image
from Utils import cost_time

class NullWriter():
    '''
    Abstract class for save png to disk. 
    '''
    @cost_time
    def __init__(self, output_file_or_dir, **kargs) -> None:
        self.output_file_or_dir = output_file_or_dir
    @cost_time
    def save(self, array, svs_file_name, ith_roi, x, y, save_name_suffix):
        pass

class WriterInPng(NullWriter):
    '''
    Save in png.
    '''
    @cost_time
    def __init__(self, output_file_or_dir, **kargs) -> None:
        self.output_file_or_dir = output_file_or_dir
        os.makedirs(self.output_file_or_dir, exist_ok=True)
    
    @cost_time
    def save(self, array, svs_file_name, ith_roi, x, y, save_name_suffix):
        tile_name = '_'.join(map(str, \
                [svs_file_name, 'roi'+str(ith_roi), x, y])) + save_name_suffix
        # print([type(array)] * 100)
        if isinstance(array, cp._core.core.ndarray):
            array = array.get()
        elif isinstance(array, PIL.Image.Image):
            array = np.array(array)
        path = os.path.join(self.output_file_or_dir, tile_name)
        cv2.imwrite(path, array)

class WriterInHdf5(NullWriter):
    '''
    Save in hdf5.
    '''
    @cost_time
    def __init__(self, output_file_or_dir, **kargs) -> None:
        self.output_file_or_dir = output_file_or_dir
        os.makedirs(os.path.dirname(self.output_file_or_dir), exist_ok=True)
        self.dataset = H5Dataset(self.output_file_or_dir, **kargs)
    
    @cost_time
    def save(self, array, svs_file_name, ith_roi, x, y, save_name_suffix):
        self.dataset.save_array_and_label(array, ith_roi, x, y) #save_array_and_label_in_batch

class WriterInZarr(NullWriter):
    '''
    Save in zarr.
    '''
    @cost_time
    def __init__(self, output_file_or_dir, **kargs) -> None:
        self.output_file_or_dir = output_file_or_dir
        os.makedirs(os.path.dirname(self.output_file_or_dir), exist_ok=True)
        self.dataset = ZarrDataset(self.output_file_or_dir, **kargs)
    
    @cost_time
    def save(self, array, svs_file_name, ith_roi, x, y, save_name_suffix):
        self.dataset.save_array_and_label(array, ith_roi, x, y) #save_array_and_label_in_batch

class WriterInRaw(NullWriter):
    '''
    Save in raw. Reference: nvidia's Accessing_File_with_GDS.ipynb
    '''
    @cost_time
    def __init__(self, output_file_or_dir, **kargs) -> None:
        self.output_file_or_dir = output_file_or_dir
        os.makedirs(self.output_file_or_dir, exist_ok=True)
    
    @cost_time
    def save(self, array, svs_file_name, ith_roi, x, y, save_name_suffix):
        tile_name = '_'.join(map(str, \
                [svs_file_name, 'roi'+str(ith_roi), x, y])) + save_name_suffix
                
        path = os.path.join(self.output_file_or_dir, tile_name)

        from cucim.clara.filesystem import CuFileDriver
        import cucim.clara.filesystem as fs

        # Create an array with size 10 (in bytes)
        # cp_arr = cp.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=cp.uint8)

        array = array.astype(cp.uint8) # must convert to uint8 or the bits are all 0.
        fno = os.open(path, os.O_RDWR | os.O_CREAT | os.O_TRUNC)
        fd = CuFileDriver(fno)
        fd.pwrite(array, array.size, 0)
        fd.close()
        os.close(fno)
        # print("content: {}".format(list(open(path, "rb").read())))

class WriterInArray(NullWriter):
    '''
    Save by np/cp.save.
    '''
    @cost_time
    def __init__(self, output_file_or_dir, **kargs) -> None:
        self.output_file_or_dir = output_file_or_dir
        os.makedirs(self.output_file_or_dir, exist_ok=True)
    
    @cost_time
    def save(self, array, svs_file_name, ith_roi, x, y, save_name_suffix):
        tile_name = '_'.join(map(str, \
                [svs_file_name, 'roi'+str(ith_roi), x, y])) + save_name_suffix

        path = os.path.join(self.output_file_or_dir, tile_name)
        if isinstance(array, np.ndarray): 
            np.save(path, array)
        elif isinstance(array, cp.ndarray):
            cp.save(path, array)
        else:
            raise Exception('array must be a np. or cp.ndarray, got {}'.format(type(array)))

if __name__ == '__main__':
    writer = WriterInRaw('.')
    writer.save(cp.array([[0,1,3],[4,5,6]]), 'xxx', 0, 1000, 2000, save_name_suffix='.raw')