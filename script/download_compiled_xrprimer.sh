mkdir xrprimer_download
cd xrprimer_download
wget -q http://10.4.11.59:18080/resources/XRlab/xrprimer_compiled/xrprimer_4308689b_mocapimage.tar.gz
tar -zxvf xrprimer_4308689b_mocapimage.tar.gz
cd ..
cp -r xrprimer_download/xrprimer/build ./xrprimer/
rm -rf xrprimer_download
