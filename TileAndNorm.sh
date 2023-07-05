#!/bin/bash
source activate pytorch_env


#!/bin/bash
# @Desc 此脚本用于获取一个指定区间且未被占用的随机端口号
# @Author Hellxz <hellxz001@foxmail.com>

PORT=0
#判断当前端口是否被占用，没被占用返回0，反之1
function Listening {
   TCPListeningnum=`netstat -an | grep ":$1 " | awk '$1 == "tcp" && $NF == "LISTEN" {print $0}' | wc -l`
   UDPListeningnum=`netstat -an | grep ":$1 " | awk '$1 == "udp" && $NF == "0.0.0.0:*" {print $0}' | wc -l`
   (( Listeningnum = TCPListeningnum + UDPListeningnum ))
   if [ $Listeningnum == 0 ]; then
       echo "0"
   else
       echo "1"
   fi
}

#指定区间随机数
function random_range {
   shuf -i $1-$2 -n1
}

#得到随机端口
function get_random_port {
   templ=0
   while [ $PORT == 0 ]; do
       temp1=`random_range $1 $2`
       if [ `Listening $temp1` == 0 ] ; then
              PORT=$temp1
       fi
   done
   echo "$PORT"
}
# get_random_port 50000 65535; #这里指定了1~10000区间，从中任取一个未占用端口号



MASTER_ADDR=ps
MASTER_PORT=`get_random_port 50000 65535`
WORLD_SIZE=4

file_input=$1 # .xml
folder_before_norm=$2
folder_after_norm=$3
folder_valid_cords=$4
target_input=$5

mkdir $folder_valid_cords

set -euxo pipefail  

svs=${file_input/.xml/.svs}

if [[ -f $svs ]];then
svs=$svs
elif [[ -f ${svs/.svs/.tif} ]]; then
svs=${svs/.svs/.tif}
else
svs=${svs/.svs/.tiff}
fi

if [[ ! -f $svs ]]; then
echo "svs file not found"
exit 1
fi

DIR_SCRIPTS=$(cd `dirname $0`; pwd)

# TorchstainNormalizer
# torchrun
python -m torch.distributed.run --nnodes=1 --nproc_per_node=4 \
    --rdzv_id=100 --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    $DIR_SCRIPTS/ChangablePipeline.py \
        --path_svs $svs \
        --path_xml $file_input \
        --str_normalizer MacenkoNormalizerByChenYuhang243 \
        --str_writer_before_norm WriterInPng \
        --str_writer_after_norm WriterInPng \
        --path_output_before_norm $folder_before_norm/`basename $file_input` \
        --path_output_after_norm $folder_after_norm/`basename $file_input` \
        --use_cucim True \
        --target_input $target_input \
        --save_name_suffix .png \
        --folder_save_cords $folder_valid_cords/`basename $file_input` \
        # > ${output_folder_prefix}.log 2>&1