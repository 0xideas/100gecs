
while [ true ]
do
    echo -n "."
    sleep 1s
    process_text=$(ssh -t -i $KEY_PATH -q -o "StrictHostKeyChecking no" ubuntu@$REMOTE_HOST ps -u ubuntu | grep "python3")
    process_match_length=${#process_text}
    if [[ process_match_length -eq "0" ]]
    then
        break
    fi
done

echo "loop done, quitting in 1m"

sleep 1m
cd terraform
terraform destroy -auto-approve
cd ..