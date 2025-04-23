#!/bin/bash

# Arrays of values
lamb=(0.1 1 10)
path=("results/E3/E3.1" "results/E3/E3.2" "results/E3/E3.3")
cc=("none" "tahoe" "reno" "cubic")
# cc=("reno")


# Iterate over indices
for i in "${!lamb[@]}"; do
  for c in "${cc[@]}"; do
    echo "Running: python3 mm1.py --cc $c --results_path ${path[$i]} --theta ${lamb[$i]}"
    python3 mm1.py --cc "$c" --results_path "${path[$i]}" --theta "${lamb[$i]}"
  done
done
