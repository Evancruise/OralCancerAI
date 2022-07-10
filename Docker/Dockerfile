# Dockerfile, Image, Container
FROM python:3.7

# WOEKDIR /app
ADD oralcancerapp.py .

RUN pip install PyQt5 imantics matplotlib

# COPY requirements.txt
# RUN pip install -r requirements.txt
# COPY ./app ./app

RUN apt-get update && apt-get install -y python3-opencv
RUN pip install opencv-python

CMD ["python", "./oralcancerapp.py"]

# docker build -t oralcancer001 .
# docker run -p 8000:8000 -t -i oralcancer001 
# (sudo terminal / interaction)
# docker exec -it {CONTAINER_ID} /bin/sh