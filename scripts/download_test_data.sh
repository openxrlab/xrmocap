mkdir xrmocap_download
cd xrmocap_download
wget -q https://openxrlab-share-mainland.oss-cn-hangzhou.aliyuncs.com/xrmocap/tests.tar.gz
tar -zxvf tests.tar.gz
wget -q https://openxrlab-share-mainland.oss-cn-hangzhou.aliyuncs.com/xrmocap/xrmocap_data.tar.gz
tar -zxvf xrmocap_data.tar.gz
cd ..
cp -r xrmocap_download/tests ./
cp -r xrmocap_download/xrmocap_data ./
rm -rf xrmocap_download
