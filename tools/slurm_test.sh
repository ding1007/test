#!/usr/bin/env bash #声明该脚本使用 Bash 解释器来执行。

set -x #执行时显示每条命令。

PARTITION=$1 #指定计算节点的分区。
JOB_NAME=$2 #作业名称。
CONFIG=$3 #MMDetection 的配置文件路径。
CHECKPOINT=$4 #MMDetection 模型检查点文件路径。
GPUS=${GPUS:-8} #用于设置使用的 GPU 数量，默认为 8。
GPUS_PER_NODE=${GPUS_PER_NODE:-8} #每个节点使用的 GPU 数量，默认为 8。
CPUS_PER_TASK=${CPUS_PER_TASK:-5} #每个任务使用的 CPU 核心数量，默认为 5。
PY_ARGS=${@:5} #保存剩余的命令行参数作为 Python 脚本的参数。
SRUN_ARGS=${SRUN_ARGS:-""}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \ 
srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    ${SRUN_ARGS} \
    python -u tools/test.py ${CONFIG} ${CHECKPOINT} --launcher="slurm" ${PY_ARGS}
