variable "instance_name" {
        default = "run-evaluate-gec-inst"
}

variable "instance_type" {
        default = "t2.micro"
}

variable "subnet_id" {
        description = "The VPC subnet the instance(s) will be created in"
}

variable "ami_id" {
        description = "The AMI to use"
}

variable "number_of_instances" {
        description = "number of instances to be created"
        default = 1
}


variable "ami_key_pair_name" {
        description = "ami key pair name"
}