FROM pytorch/pytorch:1.3-cuda10.1-cudnn7-devel
USER root
ADD requirements.txt requirements.txt
RUN pip install -r requirements.txt
WORKDIR /app
ENTRYPOINT ["python3"]
