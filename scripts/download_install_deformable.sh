mkdir xrmocap_download
cd xrmocap_download
wget -q https://openxrlab-share-mainland.oss-cn-hangzhou.aliyuncs.com/xrmocap/Deformable.tar
tar -xvf Deformable.tar
cd Deformable
sh make.sh
cd ../..
rm -rf xrmocap_download
