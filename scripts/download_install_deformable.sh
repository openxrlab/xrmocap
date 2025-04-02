mkdir xrmocap_download
cd xrmocap_download
wget -q --no-check-certificate 'https://docs.google.com/uc?export=download&id=1t92uAuJWyoKI0uuiMq_VkBzC6HJ75Bz_' -O Deformable.tar
tar -xvf Deformable.tar
cd Deformable
sh make.sh
cd ../..
rm -rf xrmocap_download
