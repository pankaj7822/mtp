#!/bin/sh
#SBATCH --nodes=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=pankaj.krjha.cse17.iitbhu
#SBATCH --error=vae_error
#SBATCH --output=vae_output
#SBATCH --partition=cpu
#SBATCH --exclusive


python memd_features.py gi