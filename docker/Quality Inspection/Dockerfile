# base image
FROM python:3.7.12-slim
MAINTAINER jimmysu

# creat and direct the folder

RUN mkdir -p /app
WORKDIR /app

# copy all file
COPY . .
# install packages
RUN python -m pip install --upgrade pip &&\ 
	pip install -r requirements.txt

# Run
ENTRYPOINT ["python3"]
CMD ["app.py"]