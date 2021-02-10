#!/bin/bash -eux

# Run this script from the repo's root folder:
#
# $ ./docker/build-and-push.sh

# 1. Build Docker images for CPU and GPU

image="us-docker.pkg.dev/replicate/raoumer/isrrescnet"
gpu_tag="$image:gpu"

docker build -f docker/Dockerfile.gpu --tag "$gpu_tag" .

# 2. Test the images on sample data

test_input_folder=/tmp/test-isrrescnet/input
mkdir -p $test_input_folder
cp isrrescnet_code_demo/samples/bird.png $test_input_folder/
test_output_folder=/tmp/test-isrrescnet/output

docker run -it --rm --gpus all \
    -v $test_input_folder:/code/LR \
    -v $test_output_folder/gpu:/code/sr_results \
    $gpu_tag

[ -f $test_output_folder/gpu/bird.png ] || exit 1

sudo rm -rf "$test_input_folder"
sudo rm -rf "$test_output_folder"

# 3. Push images to Replicate's Docker registry

docker push "$gpu_tag"
