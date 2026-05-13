arr=(*.tar.gz)
for i in {0..586}
do
    file="${arr[i]}"
    tar -xzf "$file"
    folder="${file%.tar.gz}"
    cd $folder
    for dir in *
    do
        if [ "$dir" != "surf" ] && [ -d "$dir" ]; then
            rm -rf "$dir"
        fi
    done
    cd ..
    tar -czf "$file" "$folder"
    rm -rf "$folder"
done