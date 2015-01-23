# Convert images and pad intead of stretching the image
new_name="data/256_padded"
mkdir $new_name
mkdir $new_name/train_all
mkdir $new_name/test

# Train images
cp -R data/raw/train $new_name/train_all # Quick and dirty way to copy folders structure
for d in $new_name/train_all/*; do
    echo "Working on... $d"
    for name in $d/*.jpg; do
        convert -gravity center -extent 256x256 $name $name
    done
done

# Test images
cp -R data/raw/test $new_name/test
for name in $new_name/test *.jpg; do
    convert -gravity center -extent 256x256 $name $name
done

# Resize without keeping aspect ration
# convert -resize 256x256\! $name $name

# Resize and keep aspect ratio
# convert -resize 256x256 -gravity center -extent 256x256 $name $new_name 

# Preserve size but pad the images
# convert -gravity center -extent 256x256 $name $new_name

