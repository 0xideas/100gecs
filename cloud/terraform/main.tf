provider "aws" {
    shared_credentials_files = ["~/.aws/credentials"]
    profile = "default"
    region = "eu-central-1"
}

resource "aws_instance" "ec2_instance" {
    ami = "${var.ami_id}"
    subnet_id = "${var.subnet_id}"
    instance_type = "${var.instance_type}"
    key_name = "${var.ami_key_pair_name}"
    vpc_security_group_ids = ["${var.aws_security_group}"]
} 

output "server_private_ip" {
    value = aws_instance.ec2_instance.private_ip
}
output "server_public_ipv4" {
    value = aws_instance.ec2_instance.public_ip
}
output "server_id" {
    value = aws_instance.ec2_instance.id
}