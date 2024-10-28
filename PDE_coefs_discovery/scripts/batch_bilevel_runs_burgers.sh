#!/bin/bash

# Specify the number of times you want to run the Python script
# times_to_run=15
data_size=(4 9 16 25 36 49 64 81 100)
# init_values=(1.08557134e-03 2.23907267e-03 3.63436295e-03 5.12098877e-03 6.85725195e-03 9.17509385e-03 1.20251292e-02 1.59978319e-02 2.30263305e-02, 1.08753133e-01)
# init_values=(5.24486311e-03 1.13734470e-02 1.80194242e-02 2.57665720e-02)
# init_values=(1.12253425e-01 4.18104778e-01)
# Loop and run the Python script
# for value in "${init_values[@]}"
# do
#    for data_num in "${data_size[@]}"
#    do
#       echo "Running with initial value $value and grid size $data_num"
#       CUDA_VISIBLE_DEVICES=$1 python3 burgers_inverse_spectral_bilevel.py \
#       --number_data_points $data_num \
#       --nu_lr 5e-4 \
#       --nu_init $value \
#       --iterations 1000 \
#       --noise 0 \
#       --log_file "burgers_bileveldiffnoise_1" \
#       --random_or_grid "grid"
#    done
# done


# init_values=(3.49560715e-02 4.54880225e-02 5.91116957e-02)
# for value in "${init_values[@]}"
# do
#    for data_num in "${data_size[@]}"
#    do
#       echo "Running with initial value $value and grid size $data_num"
#       CUDA_VISIBLE_DEVICES=$1 python3 burgers_inverse_spectral_bilevel.py \
#       --number_data_points $data_num \
#       --nu_lr 5e-3 \
#       --nu_init $value \
#       --iterations 1000 \
#       --noise 0 \
#       --log_file "burgers_bileveldiffnoise_1" \
#       --random_or_grid "grid"
#    done
# done

# init_values=(1.12253425e-01)
# for value in "${init_values[@]}"
# do
#    for data_num in "${data_size[@]}"
#    do
#       echo "Running with initial value $value and grid size $data_num"
#       CUDA_VISIBLE_DEVICES=$1 python3 burgers_inverse_spectral_bilevel.py \
#       --number_data_points $data_num \
#       --nu_lr 1e-2 \
#       --nu_init $value \
#       --iterations 1000 \
#       --noise 0 \
#       --log_file "burgers_bileveldiffnoise_2" \
#       --random_or_grid "grid"
#    done
# done

init_values=(4.18104778e-01)
for value in "${init_values[@]}"
do
   for data_num in "${data_size[@]}"
   do
      echo "Running with initial value $value and grid size $data_num"
      CUDA_VISIBLE_DEVICES=$1 python3 burgers_inverse_spectral_bilevel.py \
      --number_data_points $data_num \
      --nu_lr 1e-2 \
      --nu_init $value \
      --iterations 2000 \
      --noise 0 \
      --log_file "burgers_bileveldiffnoise_1" \
      --random_or_grid "grid"
   done
done