# build according to Dockerfile
FINAL_TAG=openxrlab/xrmocap_runtime:ubuntu1804_x64_cu114_py38_torch1120_mmcv161
TAG=${FINAL_TAG}_not_compatible
eval $(ssh-agent)
ssh-add ~/.ssh/id_rsa
docker build --ssh default=${SSH_AUTH_SOCK} -t $TAG -f Dockerfile .
# remove 2 files to be compatible with newer docker
CONTAINER_ID=$(docker run -ti -d $TAG)
docker exec -ti $CONTAINER_ID sh -c "
    rm /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1
    rm /usr/lib/x86_64-linux-gnu/libcuda.so.1"
docker commit $CONTAINER_ID $FINAL_TAG
docker rm -f $CONTAINER_ID
echo "Successfully tagged $FINAL_TAG"
