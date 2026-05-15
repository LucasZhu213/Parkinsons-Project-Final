#!/bin/bash

arr=(ls *.tar.gz)
for i in {1..587}
do
    mv "${arr[i]}" CORT_SURFACES2
done
