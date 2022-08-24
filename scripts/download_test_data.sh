mkdir xrmocap_download
cd xrmocap_download
wget -q http://10.4.11.59:18080/resources/XRlab/xrmocap/tests.tar.gz
tar -zxvf tests.tar.gz
wget -q http://10.4.11.59:18080/resources/XRlab/xrmocap/xrmocap_data.tar.gz
tar -zxvf xrmocap_data.tar.gz
cd ..
cp -r xrmocap_download/tests ./
cp -r xrmocap_download/xrmocap_data ./
rm -rf xrmocap_download
