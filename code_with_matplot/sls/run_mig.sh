#!/bin/bash

# Este arquivo é executado por um job.
#
# Os módulos devem ser recarregados e os ambiente reativados.

module load anaconda3/2022.10
module load openmpi/4.0.7-gcc-ucx-1.13.1

source ~/.bashrc

# É necessário desativar o ambiente atual primeiro devido a um bug.
conda deactivate
conda activate devito

export DEVITO_LANGUAGE=openmp

# Caminho absoluto do seu programa em python.
PGR=/home/joao.santana/cluster_gas/code_rtm_implemnte/code_with_matplot/sls/cluster_visco_gas_sls.py

python $PGR $WORK_DIR/$1 $WORK_DIR/$2