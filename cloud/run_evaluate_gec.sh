cd cloud/terraform
terraform apply

cd ../..
scp -i ~/auth/leon_ec2 cloud/config/config /home/ubuntu/config
ssh -i ~/auth/leon_ec2 user@remote.host nohup python3 100gecs/cloud/scripts/evaluate_gec.py

cd cloud/terraform
terraform destroy