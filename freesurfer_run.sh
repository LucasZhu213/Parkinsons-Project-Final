#!/bin/bash

set -e
. /opt/setup.sh
export FS_LICENSE=`pwd`/license.txt
export SUBJECTS_DIR=$PWD 

sub="$1"
sub="${sub%.nii.gz}"
export OMP_NUM_THREADS=4

already_found=(ls *.tar.gz)
if printf '%s\0' "${already_found[@]}" | grep -qw "$element";
then 
    echo "ALREADY FOUND"
else
    recon-all -subject "$sub" -i "$1" -all -openmp 4
    tar czf "${sub}.tar.gz" "$sub"
    rm -rf "$sub"
fi