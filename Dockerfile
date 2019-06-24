FROM tensorflow/tensorflow:1.10.0-gpu-py3

LABEL maintainer=goofy.gao@51job.com

COPY deepseg /root/deepseg

WORKDIR /root/deepseg

# RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple -r /root/deepseg/requirements.txt

