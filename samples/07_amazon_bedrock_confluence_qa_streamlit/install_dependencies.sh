pip install --no-build-isolation --force-reinstall \
    ./dependencies/awscli-*-py3-none-any.whl \
    ./dependencies/boto3-*-py3-none-any.whl \
    ./dependencies/botocore-*-py3-none-any.whl

pip install --no-cache-dir -r requirements.txt
sudo yum install -y iproute
sudo yum install -y jq
sudo yum install -y lsof