#!/bin/bash

cd PPMI_SCANS
mapfile -d '' files < <(find . -type f -name "*.nii.gz" -print0)
git config --global user.name LucasZhu213
git config --global user.email awesomeknght@gmail.com
ssh -T git@github.com

batch_size=100
total=${#files[@]}

echo "Total files: $total"

for ((i=0; i<$total; i+=batch_size)); do
  echo "Processing batch $((i/batch_size + 1))..."

  # Add batch to git
  git reset

  for ((j=i; j<i+batch_size && j<total; j++)); do
    git add "${files[j]}"
  done

  git commit -m "Add nii.gz batch $((i/batch_size + 1))"
  git push
done
