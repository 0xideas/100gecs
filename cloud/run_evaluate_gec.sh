cd terraform
terraform apply -auto-approve

sleep 1m
cd ..
ssh-keygen -f "/home/leon/.ssh/known_hosts" -R $REMOTE_IP

echo "setting env variables"
source ./set_env_variables.sh
# scp -i ~/auth/leon_ec2.pem -o "StrictHostKeyChecking no" config/config ubuntu@$REMOTE_HOST:/home/ubuntu/config
ssh -i ~/auth/leon_ec2.pem -o "StrictHostKeyChecking no" ubuntu@$REMOTE_HOST bash run.sh

sleep 10s
cd terraform
terraform destroy -auto-approve