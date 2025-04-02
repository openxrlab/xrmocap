mkdir xrmocap_download
cd xrmocap_download
gdown https://docs.google.com/uc?id=1Mt2oq5Ghf4SY5cqn1ak5fQ5UGC3p8DDz
tar -zxvf tests.tar.gz
gdown https://docs.google.com/uc?id=1VxL2q1bcT9WxJqWdmf54a1IhLrpRUV5j
tar -zxvf xrmocap_data.tar.gz
cd ..
cp -r xrmocap_download/tests ./
cp -r xrmocap_download/xrmocap_data ./
rm -rf xrmocap_download
