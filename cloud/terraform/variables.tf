variable "instance_name" {
        default = "run-evaluate-gec-inst"
}

variable "instance_type" {
        default = "t2.2xlarge"
}

variable "subnet_id" {
        description = "The VPC subnet the instance(s) will be created in"
}

variable "ami_id" {
        description = "The AMI to use"
}


variable "ami_key_pair_name" {
        description = "ami key pair name"
}

variable "aws_security_group" {
        description = "aws security group"
}

variable "elastic_ip" {
        description = "elastic ip"
}
