FROM docker.io/frank1chude1qian/dt-machine-learning-base-environment:daffy-amd64
#FROM frank1chude1qian/aido6:main

ARG PIP_INDEX_URL="https://pypi.org/simple"
ENV PIP_INDEX_URL=${PIP_INDEX_URL}

# Setup any additional pip packages
RUN pip3 install -U "pip"
COPY requirements.* ./
RUN cat requirements.* > .requirements.txt
RUN pip3 install --use-feature=2020-resolver -r .requirements.txt

# Copy all solution files to workspace
WORKDIR /submission
COPY solution.py /submission
COPY helperFncs.py /submission
COPY cbcNet.py /submission
COPY cbcNet.h5 /submission
#COPY cbcNetv2.py /submission
#COPY cbcNetv2_anomaly.h5 /submission
#COPY cbcNetv2_bc.h5 /submission
ENV FORCE_CPU_INFERENCE False

ENTRYPOINT ["python3", "solution.py"]
