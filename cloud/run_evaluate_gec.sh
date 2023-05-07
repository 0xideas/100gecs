cd terraform
terraform apply -auto-approve

sleep 1m
cd ..
bash ./set_env_variables.sh
# scp -i ~/auth/leon_ec2.pem config/config ubuntu@$REMOTE_HOST:/home/ubuntu/config
ssh -i ~/auth/leon_ec2.pem ubuntu@$REMOTE_HOST bash run.sh

sleep 10s
cd terraform
terraform destroy -auto-approve