#!/bin/bash

# Specify the number of times you want to run the Python script
times_to_run=20

# Loop and run the Python script
# for ((i=1; i<=times_to_run; i++))
# do
#    echo "Execution $i"
#    CUDA_VISIBLE_DEVICES=$1 python3 allen_cahn_inverse_spectral_bilevel.py --number_data_points $i --nu_lr 1e-4
# done

# Specify the number of times you want to run the Python script
times_to_run=15

# init_values=(1.11450594e-01 2.30850847e-01 3.63664645e-01 5.15955779e-01 7.05740422e-01 9.21523538e-01 1.20781956e+00 1.61418379e+00 2.29463028e+00 8.98358041)
init_values=(2.18231327e-03 4.57061134e-03 7.26983969e-03 1.03304875e-02 1.39288295e-02 1.84102060e-02 2.43125497e-02 3.21533896e-02 4.62505950e-02 2.02613127e-01)
# Loop and run the Python script
# for ((i=2; i<=times_to_run; i++))
# do
#    echo "Execution $i"
#    CUDA_VISIBLE_DEVICES=$1 python3 allen_cahn_inverse_jax.py --number_data_points $i --nu_lr 1e-3
# done

for value in "${init_values[@]}"
do
   for ((i=1; i<=times_to_run; i++))
   do
      echo "Running with initial value $value and grid size $i x $i"
      # CUDA_VISIBLE_DEVICES=$1 python3 burgers_inverse_jax.py --number_data_points $i --nu_lr 1e-4 --nu_init $value --iterations 300000
      CUDA_VISIBLE_DEVICES=$1 python3 allen_cahn_inverse_spectral_bilevel.py --number_data_points $i --nu_lr 1e-3 --nu_init $value
   done
done