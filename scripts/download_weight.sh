mkdir xrmocap_download
cd xrmocap_download
wget -q http://10.4.11.59:18080/resources/XRlab/xrmocap/weight.tar.gz
tar -zxvf weight.tar.gz
cd ..
cp -r xrmocap_download/weight ./
rm -rf xrmocap_download
