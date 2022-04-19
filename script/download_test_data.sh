mkdir xrmocap_download
cd xrmocap_download
wget -q http://10.4.11.59:18080/resources/XRlab/xrmocap.tar.gz
tar -zxvf xrmocap.tar.gz
cd ..
cp -r xrmocap_download/xrmocap/* ./
rm -rf xrmocap_download
