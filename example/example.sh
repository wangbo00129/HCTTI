source activate cucim_env
python ChangablePipeline.py \
--path_svs /home/wangb/projects/20220801_svs/svs/TCGA-EA-A4BA-01Z-00-DX1/TCGA-EA-A4BA-01Z-00-DX1.7EB090B2-E79E-417F-A871-247353679D7B.svs \
--path_xml /home/wangb/projects/20220801_svs/svs/TCGA-EA-A4BA-01Z-00-DX1/TCGA-EA-A4BA-01Z-00-DX1.7EB090B2-E79E-417F-A871-247353679D7B.xml \
--str_normalizer MacenkoNormalizerByChenYuhang243 \
--str_writer_before_norm WriterInPng \
--str_writer_after_norm WriterInPng \
--path_output_before_norm /home/wangb/projects/20220920_test_changablepipelineforsvspreprocessor/before_norm/ \
--path_output_after_norm /home/wangb/projects/20220920_test_changablepipelineforsvspreprocessor/after_norm/ \
--use_cucim True \
--target_input /home/wangb/pipelines/svspreprocessor/asset/Template.png \
--save_name_suffix .png \


# /home/wangb/projects/20220920_test_changablepipelineforsvspreprocessor/


