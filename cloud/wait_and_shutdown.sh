
current_screens_length=1
while [ $current_screens_length != 0 ]
do
    echo -n "."
    sleep 1s
    current_screens=$(ssh -t -i $KEY_PATH -q -o "StrictHostKeyChecking no" ubuntu@$REMOTE_HOST screen -list)
    current_screens_length=${#current_screens}
done

echo "loop done, quitting in 1m"

sleep 1m
cd terraform
terraform destroy -auto-approve