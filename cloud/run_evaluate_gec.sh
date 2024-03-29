cd terraform
terraform apply -auto-approve

cd ..

echo "setting env variables"
source ./set_env_variables.sh
ssh-keygen -f $SSH_HOST_PATH -R $REMOTE_HOST
scp -i $KEY_PATH -o "StrictHostKeyChecking no" config/config.json ubuntu@$REMOTE_HOST:/home/ubuntu/config.json
ssh -t -i $KEY_PATH -o "StrictHostKeyChecking no" ubuntu@$REMOTE_HOST nohup bash run.sh $AWS_ACCESS_KEY_ID $AWS_SECRET_ACCESS_KEY &

sleep 2m

bash wait_and_shutdown.sh

#scp -i $KEY_PATH -o "StrictHostKeyChecking no" ubuntu@$REMOTE_HOST:/home/ubuntu/gec.json gec.json
