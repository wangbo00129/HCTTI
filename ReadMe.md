Before using, clone https://github.com/wangbo00129/largedataloader besides this repo's folder. 

Usage: ChangablePipeline.py [-h] [--path_svs PATH_SVS] [--path_xml PATH_XML] [--str_normalizer STR_NORMALIZER] [--str_writer_before_norm STR_WRITER_BEFORE_NORM] [--str_writer_after_norm STR_WRITER_AFTER_NORM]

                            [--path_output_before_norm PATH_OUTPUT_BEFORE_NORM] [--path_output_after_norm PATH_OUTPUT_AFTER_NORM] [--use_cucim USE_CUCIM] [--read_region_device READ_REGION_DEVICE]

                            [--read_region_num_workers READ_REGION_NUM_WORKERS] [--discard_page_cache DISCARD_PAGE_CACHE] [--target_input TARGET_INPUT] [--save_name_suffix SAVE_NAME_SUFFIX] [--patch_size PATCH_SIZE]

                            [--step_size STEP_SIZE] [--max_ratio_hole MAX_RATIO_HOLE] [--count COUNT] [--folder_save_cords FOLDER_SAVE_CORDS]



Changable pipeline from svs to normalized tiles



optional arguments:

  -h, --help            show this help message and exit

  --path_svs PATH_SVS

  --path_xml PATH_XML

  --str_normalizer STR_NORMALIZER

                        available: MacenkoNormalizer, NullNormalizer, NumpyMacenkoNormalizer, ReinhardNormalizer, RuifrokNormalizer, TiatoolboxMacenkoNormalizer, TorchstainMacenkoNormalizer

  --str_writer_before_norm STR_WRITER_BEFORE_NORM

                        available: NullWriter, WriterInArray, WriterInHdf5, WriterInPng, WriterInRaw, WriterInZarr

  --str_writer_after_norm STR_WRITER_AFTER_NORM

                        available: NullWriter, WriterInArray, WriterInHdf5, WriterInPng, WriterInRaw, WriterInZarr

  --path_output_before_norm PATH_OUTPUT_BEFORE_NORM

  --path_output_after_norm PATH_OUTPUT_AFTER_NORM

  --use_cucim USE_CUCIM

  --read_region_device READ_REGION_DEVICE

                        could be cpu or cuda. Only take effect when use_cucim

  --read_region_num_workers READ_REGION_NUM_WORKERS

                        Only take effect when use_cucim

  --discard_page_cache DISCARD_PAGE_CACHE

                        whether to discard_page_cache when use_cucim

  --target_input TARGET_INPUT

  --save_name_suffix SAVE_NAME_SUFFIX

                        It means file suffix in WriterInPng, but means nothing in large file writers and you should use "" instead.

  --patch_size PATCH_SIZE

  --step_size STEP_SIZE

  --max_ratio_hole MAX_RATIO_HOLE

  --count COUNT

  --folder_save_cords FOLDER_SAVE_CORDS



Use mpirun for acceleration.

