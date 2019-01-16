#!/bin/bash
#PBS - S /bin/bash
#PBS - N p_structure_0.17_10nm_7
#PBS - l nodes=1:ppn=8,mem=80gb,walltime=300:00:00,nice=15
#PBS - q long
#PBS - o /home/Clayton/Files/HoletransportAlGaN_0.17_10nm_2/Bias7
#PBS - e /home/Clayton/Files/HoletransportAlGaN_0.17_10nm_2/Bias7
cd /home/Clayton/Files/HoletransportAlGaN_0.17_10nm_2//Bias7
export MKL_NUM _THREADS=8
export OPENMP_NUM _THREADS=8
3D-ddcc-dyna.exe p_structure_0.17_10nm_7.sh > test7.txt

wait

