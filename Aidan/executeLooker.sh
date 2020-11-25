#!/bin/sh
#PBS -N backlook
#PBS -l walltime=24:00:00
#PBS -l select=1:ncpus=16:mem=50gb
#PBS -J 1-20

module load anaconda3/personal
source activate reinforcement

LOOKBACK=$(( ($PBS_ARRAY_INDEX+1)*10 ))
echo "$LOOKBACK"

python $PBS_O_WORKDIR/Lookback.py $LOOKBACK



