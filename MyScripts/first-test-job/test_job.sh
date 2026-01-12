#!/bin/bash
#PBS -l select=1:ncpus=1:mem=1gb
#PBS -l walltime=00:01:00
#PBS -N hello_world
#PBS -o /rds/general/user/yl9422/home/files
#PBS -e /rds/general/user/yl9422/home/files


cd $PBS_O_WORKDIR
echo "hi from job echo"


eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate M4R


python test.py
