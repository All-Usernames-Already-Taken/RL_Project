#!/bin/sh
#
# Simple "Hello World" submit script for Slurm.
#
# Replace <ACCOUNT> with your account name before submitting.
#
#SBATCH --account=dsi      # The account name for the job.
#SBATCH --job-name=TestTensorflow    # The job name.
#SBATCH -c 2
#SBATCH --time=30:00:00              # The time the job will take to run.
#SBATCH --mem-per-cpu=16gb        # The memory the job will use per cpu core.

echo "testData"
module load cuda80/toolkit cuda80/blas cudnn/5.1
module load anaconda/3-4.2.0
time
echo "TestParData"
date
python ./do_learning.py TestPar1.txt > TestParData1out.txt
date


# End of script
