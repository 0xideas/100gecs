cd terraform
terraform apply -auto-approve

cd ..

echo "setting env variables"
source ./set_env_variables.sh $1 $2
ssh-keygen -f $SSH_HOST_PATH -R $REMOTE_IP
scp -i $KEY_PATH -o "StrictHostKeyChecking no" config/config.json ubuntu@$REMOTE_HOST:/home/ubuntu/config.json
ssh -t -i $KEY_PATH -o "StrictHostKeyChecking no" ubuntu@$REMOTE_HOST bash run.sh $AWS_ACCESS_KEY_ID $AWS_SECRET_ACCESS_KEY

current_screens_length=1
while [ $current_screens_length != 0 ]
do
    echo -n "."
    sleep 1s
    current_screens=$(ssh -t -i $KEY_PATH -q -o "StrictHostKeyChecking no" ubuntu@$REMOTE_HOST screen -list | grep "gec")
    current_screens_length=${#current_screens}
done

echo "loop done, quitting in 1m"

sleep 1m
cd terraform
terraform destroy -auto-approve



#scp -i $KEY_PATH -o "StrictHostKeyChecking no" ubuntu@$REMOTE_HOST:/home/ubuntu/gec.json gec.json



