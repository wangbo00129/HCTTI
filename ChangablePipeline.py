#!/usr/bin/env python
import sys
import os
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse
from pathlib import Path
import numpy as np
import cv2
import pandas as pd
from glob import glob
import time
from mpi4py import MPI
MPI_COMM_WORLD_RANK = MPI.COMM_WORLD.Get_rank()  # The process ID (integer 0-3 for 4-process run)
MPI_COMM_WORLD_SIZE = MPI.COMM_WORLD.Get_size()

from SvsReader import SvsReader
import ImageWriter, ImageNormalizer
from Utils import load_xml, DEFAULT_LOGGER, judge_position, get_area_ratio, cu_image_to_array

class SvsSpliter():
    '''
    Class for svs splitting, normalizing and saving to disk.
    '''
    def __init__(self, path_svs, path_xml, writer_before_normalization, normalizer, writer_after_normalization, \
        count, max_ratio_hole, patch_size, step_size,
        use_cucim=True, logger=DEFAULT_LOGGER, save_name_suffix='.png', folder_save_cords='./', 
        read_region_device='cpu', discard_page_cache=False, read_region_num_workers=0) -> None:
        '''
        save_name_suffix: 
            If use WriterInPng, set to '.png' or other format is necessary. 
            If use large file system such as WriterInHdf5, you should use '' instead.
        '''
        self.path_svs = path_svs
        self.path_xml = path_xml
        self.writer_before_normalization = writer_before_normalization
        self.normalizer = normalizer
        self.writer_after_normalization = writer_after_normalization
        self.svs_reader = SvsReader(path_svs, use_cucim = use_cucim, discard_page_cache=discard_page_cache)
        self.logger = logger
        self.start = time.time()
        self.logger.info('start at: {}'.format(self.start))
        self.count = count
        self.max_ratio_hole = max_ratio_hole
        self.patch_size = patch_size
        self.step_size = step_size
        self.dimension_m, self.dimension_n = self.svs_reader.get_dimensions()
        self.logger.info('self.dimension_m {}, self.dimension_n {}'.format(self.dimension_m, self.dimension_n))
        self.svs_file_name = Path(self.path_svs).stem
        self.save_name_suffix = save_name_suffix
        # To save tile cords meet max_ratio_hole criteria
        self.folder_save_cords = folder_save_cords
        self.df_cords_valid_tiles = pd.DataFrame(columns=['roi', 'x', 'y', 'patch_size_x', 'patch_size_y', 'rank'], index=range(int(1e6)), dtype=int)
        self.row_num_df_cords_valid_tiles = 0
        self.read_region_device = read_region_device
        self.read_region_num_workers = read_region_num_workers
        
        self.logger.info(self.__dict__)

    def parse_xml_and_tile_norm_save(self):
        """
        通过滑窗的方式得到小窗口
        :param image: svs文件
        :param x_begin: 滑窗的左上角坐标x
        :param y_begin: 滑窗的左上角坐标y
        :param w: 外接矩形的宽
        :param h: 外接矩形的高
        :param patch_size: 滑窗大小
        :param stride: 滑窗步长
        :param num_name: 图片索引
        :param luncancer: 癌症区域索引
        :param health: 健康区域索引
        :param i: 癌症类型标志
        :param contours: 轮廓
        :param points_con_thre: 轮廓内点的个数阈值
        :param area_ratio_thre: window内面积比率阈值
        """
        xy_list_all = load_xml(self.path_xml)
        rank = MPI_COMM_WORLD_RANK
        xy_list = xy_list_all # [rank::WORLD_SIZE]
        current_patch_id = 0
        # 遍历多个标注区域
        for i in range(len(xy_list)):
            self.logger.info("Dealing with the {0}th Cancer area of {1}".format(i, self.svs_file_name))
            points = xy_list[i]
            contours = np.array(points) # 轮廓
            # 取当前标注区域的外接矩阵
            x_begin, y_begin, w, h = cv2.boundingRect(contours)

            # 异常控制
            if w < self.patch_size:
                w = self.patch_size+10
            if h < self.patch_size:
                h = self.patch_size+10

            x_end = x_begin+w-self.patch_size
            y_end = y_begin+h-self.patch_size
            self.logger.info([x_begin,y_begin,x_end,y_end])

            for x in range(x_begin, x_end, self.step_size):
                for y in range(y_begin, y_end, self.step_size):
                    if current_patch_id % MPI_COMM_WORLD_SIZE == rank:
                        self.read_region_and_norm_and_save(x, y, i, contours)
                    current_patch_id += 1
        
            self.logger.info('tiles are done.')

    def read_region_and_norm_and_save(self, x, y, ith_roi, contours):
        # 越界控制
        if x+self.patch_size > self.dimension_m or y+self.patch_size > self.dimension_n:
            return
        # 去除轮廓外干扰区域
        point_list = [
            (x+int(self.patch_size/2), y+int(self.patch_size/2)), 
            (x, y), 
            (x+self.patch_size, y),
            (x, y+self.patch_size), 
            (x+self.patch_size, y+self.patch_size)
            ]
        
        if judge_position(contours, point_list) < self.count:
            return

        ret = self.svs_reader.read_region((x, y), (self.patch_size, self.patch_size), \
            device=self.read_region_device, num_workers=self.read_region_num_workers)
        ret = cu_image_to_array(ret)
        # self.logger.info(ret)
        # 去除轮廓内的白色
        ratio = get_area_ratio(ret)
        if ratio > self.max_ratio_hole:
            return
        self.logger.info("get the [ {0}, {1} ] cancer picture".format(x, y))
        
        self.df_cords_valid_tiles.loc[self.row_num_df_cords_valid_tiles, 'roi'] = ith_roi
        self.df_cords_valid_tiles.loc[self.row_num_df_cords_valid_tiles, 'x'] = x
        self.df_cords_valid_tiles.loc[self.row_num_df_cords_valid_tiles, 'y'] = y
        self.df_cords_valid_tiles.loc[self.row_num_df_cords_valid_tiles, 'patch_size_x'] = self.patch_size
        self.df_cords_valid_tiles.loc[self.row_num_df_cords_valid_tiles, 'patch_size_y'] = self.patch_size
        self.df_cords_valid_tiles.loc[self.row_num_df_cords_valid_tiles, 'rank'] = MPI_COMM_WORLD_RANK
        self.row_num_df_cords_valid_tiles += 1

        self.writer_before_normalization.save(ret, self.svs_file_name, ith_roi, x, y, self.save_name_suffix)
        try:
            array_normed = self.normalizer.transform(ret)
        except Exception as e:
            if 'error, len(phi) == 0' in str(e):
                self.logger.error('failed due to error, len(phi) == 0, ignore')
                return
            else:
                raise Exception(e)
            
        self.writer_after_normalization.save(array_normed, self.svs_file_name, ith_roi, x, y, self.save_name_suffix)

    def __del__(self):
        path_cords = '_'.join([self.folder_save_cords, os.path.basename(self.path_svs), os.path.basename(self.path_xml), '.cords'])

        path_cords_tmp = '_'.join([self.folder_save_cords, os.path.basename(self.path_svs), os.path.basename(self.path_xml), '.rank{}.cords.tmp'.format(MPI_COMM_WORLD_RANK)])
        self.df_cords_valid_tiles.dropna(subset=['x'], inplace=True)
        self.df_cords_valid_tiles.to_csv(
             path_cords_tmp, 
             sep='\t', index=False)
        self.end = time.time()
        self.logger.info('end at: {}'.format(self.end))
        MPI.COMM_WORLD.barrier()
        if MPI_COMM_WORLD_RANK == 0:
            tmp_files_from_all_ranks = glob(path_cords_tmp.replace('.rank0.','.rank*.'))
            dfs_tmp = [pd.read_csv(p, sep='\t') for p in tmp_files_from_all_ranks]
            df_all_cords = pd.concat(dfs_tmp)
            df_all_cords.to_csv(path_cords, sep='\t', index=False)
            [os.remove(p) for p in tmp_files_from_all_ranks]
            self.logger.info('elapse: {}'.format(self.end - self.start))

def get_available_member(object, filter_str):
    return list(filter(lambda x:filter_str in x, dir(object)))
    
def parseArgs():
    parser = argparse.ArgumentParser(description='Changable pipeline from svs to normalized tiles')
    
    parser.add_argument('--path_svs', type=str)
    parser.add_argument('--path_xml', type=str)
    parser.add_argument('--str_normalizer', type=str, default="NullNormalizer", help="available: {}".format(', '.join(get_available_member(ImageNormalizer, 'Normalizer'))))
    parser.add_argument('--str_writer_before_norm', type=str, default="NullWriter", help="available: {}".format(', '.join(get_available_member(ImageWriter, 'Writer'))))
    parser.add_argument('--str_writer_after_norm', type=str, default="NullWriter", help="available: {}".format(', '.join(get_available_member(ImageWriter, 'Writer'))))
    parser.add_argument('--path_output_before_norm', type=str, default="/home/wangb/projects/BeforeNorm")
    parser.add_argument('--path_output_after_norm', type=str, default="/home/wangb/projects/AfterNorm")
    parser.add_argument('--use_cucim', type=str, default="true")
    parser.add_argument('--read_region_device', type=str, default="cpu", help="could be cpu or cuda. Only take effect when use_cucim")
    parser.add_argument('--read_region_num_workers', type=int, default=0, help="Only take effect when use_cucim")
    parser.add_argument('--discard_page_cache', type=str, default='false', help="whether to discard_page_cache when use_cucim")
    parser.add_argument('--target_input', type=str, default="/home/wangb/pipelines/svspreprocessor/asset/Template.png")
    parser.add_argument('--save_name_suffix', type=str, default='.png', help='It means file suffix in WriterInPng, \
        but means nothing in large file writers and you should use "" instead. ')
    parser.add_argument("--patch_size", type=int, default=512)
    parser.add_argument("--step_size", type=int, default=512)
    parser.add_argument("--max_ratio_hole", type=float, default=0.3)
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--folder_save_cords", type=str, default=None)
    # parser.add_argument("--backend", type=str, default="nccl", help='WriterInHdf5 requires mpi backend')

    args = parser.parse_args()

    if args.use_cucim.lower() == "true":
        args.use_cucim = True
    else:
        args.use_cucim = False

    if args.discard_page_cache.lower() == "true":
        args.discard_page_cache = True
    else:
        args.discard_page_cache = False
    return args

def main():
    args = parseArgs()

    normalizer = getattr(ImageNormalizer, args.str_normalizer)(target_input=args.target_input)
    writer_before_norm = getattr(ImageWriter, args.str_writer_before_norm)(output_file_or_dir=args.path_output_before_norm, r_or_w="w")
    writer_after_norm = getattr(ImageWriter, args.str_writer_after_norm)(output_file_or_dir=args.path_output_after_norm, r_or_w="w")
    svs_spliter = SvsSpliter(args.path_svs, args.path_xml, writer_before_norm, normalizer, writer_after_norm, \
        args.count, args.max_ratio_hole, args.patch_size, args.step_size, use_cucim=args.use_cucim, folder_save_cords=args.folder_save_cords,
        save_name_suffix=args.save_name_suffix, read_region_device=args.read_region_device, read_region_num_workers=args.read_region_num_workers, 
        discard_page_cache=args.discard_page_cache)
    svs_spliter.parse_xml_and_tile_norm_save()
    
if __name__ == '__main__':
    main()
