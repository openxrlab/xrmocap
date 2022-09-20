mkdir xrmocap_download
cd xrmocap_download
wget -q https://openxrlab-share.oss-cn-hongkong.aliyuncs.com/xrmocap/Deformable.tar
tar -xvf Deformable.tar
cd Deformable
sh make.sh
cd ../..
rm -rf xrmocap_download
