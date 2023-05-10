cd terraform
terraform apply -auto-approve

sleep 1m
cd ..

echo "setting env variables"
source ./set_env_variables.sh
scp -i $KEY_PATH -o "StrictHostKeyChecking no" config/config.json ubuntu@$REMOTE_HOST:/home/ubuntu/config.json
ssh -i $KEY_PATH -o "StrictHostKeyChecking no" ubuntu@$REMOTE_HOST bash run.sh $AWS_ACCESS_KEY_ID $AWS_SECRET_ACCESS_KEY

sleep 10s
cd terraform
terraform destroy -auto-approve

ssh-keygen -f $SSH_HOST_PATH -R $REMOTE_IP


#scp -i $KEY_PATH -o "StrictHostKeyChecking no" ubuntu@$REMOTE_HOST:/home/ubuntu/gec.json gec.json



