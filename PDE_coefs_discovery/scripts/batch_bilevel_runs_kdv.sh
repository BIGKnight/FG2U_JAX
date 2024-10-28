#!/bin/bash

# Specify the number of times you want to run the Python script
times_to_run=8
init_values=(2.23907267e-03 2.944e-03 3.63436295e-03 4.37e-3 5.12098877e-03 6.85725195e-03 9.17509385e-03)
# Loop and run the Python script
for value in "${init_values[@]}"
do
   for ((i=1; i<=times_to_run; i++))
   do
      echo "Running with initial value $value and grid size $i x $i"
      CUDA_VISIBLE_DEVICES=$1 python3 kdv_inverse_spectral_bilevel.py --number_data_points $i --nu_init $value --nu_lr 1e-3 --iteration 1000 --log_file kdv_bilevel_7_15 --random_or_grid grid
   done
done
   # for ((i=1; i<=times_to_run; i++))
   # do
   #    echo "Execution $i"
   #    CUDA_VISIBLE_DEVICES=$1 python3 kdv_inverse_jax.py --number_data_points $i --nu_lr 1e-3 --nu_init
   # done