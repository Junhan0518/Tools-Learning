provider "aws" {
    region = "us-east-1"
    access_key = "AKIAUYMDIQ7A4N7LQ6EM"
    secret_key = "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"
  
}

# 1. create vpc
resource "aws_vpc" "project_vpc" {
   cidr_block = "10.0.0.0/16"
   tags = {
     Name = "pro-vpc"
   }
}

# 2. create internet gateway
resource "aws_internet_gateway" "gateway" {
  vpc_id = aws_vpc.project_vpc.id
}

# 3. create custom route table
resource "aws_route_table" "route-table" {
  vpc_id = aws_vpc.project_vpc.id
  route {
      cidr_block = "0.0.0.0/0"
      gateway_id = aws_internet_gateway.gateway.id
  }

  route {
    ipv6_cidr_block = "::/0"
    gateway_id = aws_internet_gateway.gateway.id
  }

  tags = {
    Name = "Project1"
  }
}

# 建立變數，可以不用預設值，在apply時會要求輸入
variable "subnet_prefix" {
    description = "cidr block for the subnet"
    # default = "" 可預設在此
    # type = string 指定變數型態
}

# 4. create a subnet
resource "aws_subnet" "subnet-1" {
  vpc_id = aws_vpc.project_vpc.id
  cidr_block = var.subnet_prefix[0].cidr_block
  availability_zone = "us-east-1a"

  tags = {
    Name = var.subnet_prefix[0].name
  }
}

# 5. Associatie subnet with route table 
resource "aws_route_table_association" "rta" {
  subnet_id = aws_subnet.subnet-1.id
  route_table_id = aws_route_table.route-table.id
}

# 6. Create Security group to allow port 22, 80, 443
resource "aws_security_group" "security-group" {
  name        = "allow_web_traffic"
  description = "Allow Web inbound traffic"
  vpc_id = aws_vpc.project_vpc.id

  ingress {
      description = "HTTPS"
      from_port = 443
      to_port = 443
      protocol = "tcp"
      cidr_blocks = [var.subnet_prefix[1].cidr_block]
  }
  ingress {
      description = "HTTP"
      from_port = 80
      to_port = 80
      protocol = "tcp"
      cidr_blocks = [var.subnet_prefix[1].cidr_block]
  }
  ingress {
      description = "SSH"
      from_port = 22
      to_port = 22
      protocol = "tcp"
      cidr_blocks = [var.subnet_prefix[1].cidr_block]
  }

  egress {
      from_port = 0
      to_port = 0
      protocol = "-1"
      cidr_blocks = [var.subnet_prefix[1].cidr_block]
  }

  tags = {
    Name = var.subnet_prefix[1].name
  }
}

# 7. create a network interface with an ip in the subnet that was created in step 4

resource "aws_network_interface" "net-interface" {
  subnet_id = aws_subnet.subnet-1.id
  private_ips = ["10.0.1.50"]
  security_groups = [aws_security_group.security-group.id]
}

# 8. Assign elastic Ip to the network interface created in step 7
resource "aws_eip" "eip" {
  vpc = true
  network_interface = aws_network_interface.net-interface.id
  associate_with_private_ip = "10.0.1.50"
  depends_on = [aws_internet_gateway.gateway]
}

output "server_public_ip" {
  value = aws_eip.eip.public_ip
}

# 9. Create Ubuntu server and install/enable apache2
resource "aws_instance" "project1-instance" {
  ami = "ami-0e472ba40eb589f49"
  instance_type = "t2.micro"
  availability_zone = "us-east-1a"
  key_name = "terraform-project1-key"

  network_interface {
    device_index = 0
    network_interface_id = aws_network_interface.net-interface.id
  }

  user_data = <<-EOF
                #!/bin/bash
                sudo apt update -y
                sudo apt install apache2 -y
                sudo systemctl start apache2
                sudo bash -c 'echo your very first web server > /var/www/html/index.html'
                EOF
  tags = {
    Name = "web-server"
  }
}

output "server_private_ip" {
  value = aws_instance.project1-instance.private_ip
}

output "server_id" {
  value = aws_instance.project1-instance.id
}





# 服務提供商, 建立provider才能使用提供商的服務
# provider "aws" {
#   region = "us-east-1"
#   access_key = "AKIAUYMDIQ7A4N7LQ6EM"
#   secret_key = "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"
# }

# 創建服務，每個resource代表一個服務
# aws EC2
# resource "aws_instance" "myfirstserver" {
#     ami = "ami-0e472ba40eb589f49"
#     instance_type = "t2.micro"

#     tags = {
#       # Name = "ubuntu"
#     }
# }


# aws vpc
# resource "aws_vpc" "first-vpc" {
#   cidr_block = "10.0.0.0/16"
#   tags = {
#     Name = "production"
#   }
# }

# resource "aws_vpc" "second-vpc" {
#   cidr_block = "10.1.0.0/16"
#   tags = {
#     Name = "Dev"
#   }
  
# }

# resource "aws_subnet" "subnet-1" {
#   vpc_id = aws_vpc.first-vpc.id
#   cidr_block = "10.0.1.0/24"
#   tags = {
#     Name = "prod-subnet"
#   }
# }

# resource "aws_subnet" "subnet-2" {
#   vpc_id = aws_vpc.second-vpc.id
#   cidr_block = "10.1.1.0/24"

#   tags = {
#     Name = "dev-subnet1"
#   }
  
# }




## resource template
# resource "<provider>_<resource_type>" "name" {
#     # config options
#     key=""
#     key2=""
# }

## terraform command
# terraform init 初始化讓工作資料夾讓其他指令可以運行
# terraform plan 查看是否有更動(新增, 更改, 刪除)
# terraform validate Check whether the configuration is valid
# terraform apply 執行程式
  # --auto-approve 執行時不用再打yes
  # -target resource_name 部屬指定資源
  # -var "variable_name = value" 可在command line 就輸入變數值
# terraform destory 將resource刪除, 也可以單純將不要的resource註解掉
  # --auto-approve 執行時不用再打yes
  # -target resource_name 可指定要刪除的資源
# terraform state list 列出所有部屬的資源
# terraform state show resource_name 列出資源詳細資料
# terraform refrash 不用使用apply就可更新資源狀態



## note
# 在terraform 中, resource的順序不影響程式執行, 如:vpc的子網路resource可以寫在vpc前
# tfstate file非常重要，它記錄了所有資源的資料，若是搞亂了會造成服務無法架起來
# output一次只能放一個value
# 變數通常不使用terminal輸入，會建立檔案來存放變數值
# tfvars file預設名稱為terraform.trfvars，也可以使用-var-file filename來指定file
# variable:可以使用default, command line, apply時, tfvar file等方法輸入變數