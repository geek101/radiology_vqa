#FROM nvidia/cuda:8.0-cudnn6-devel-ubuntu16.04
FROM nvidia/cuda:9.2-cudnn7-devel-ubuntu18.04 
LABEL maintainer caffe-maint@googlegroups.com

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        vim-tiny \
        cmake \
        git \
        wget \
        libatlas-base-dev \
        libboost-all-dev \
        libgflags-dev \
        libgoogle-glog-dev \
        libhdf5-serial-dev \
        libleveldb-dev \
        liblmdb-dev \
        libopencv-dev \
        libprotobuf-dev \
        libsnappy-dev \
        protobuf-compiler \
        python-dev \
        python-numpy \
        python-pip \
        python-setuptools \
        python-scipy && \
    rm -rf /var/lib/apt/lists/*

COPY ./bottom-up-attention /opt/butd

ENV CAFFE_ROOT=/opt/butd/caffe
WORKDIR $CAFFE_ROOT

# Build and install caffe
RUN pip2 install --upgrade pip && \
    cd python && for req in $(cat requirements.txt) pydot; do pip install $req; done && cd .. && \
    make -j"$(nproc)" && \
    make pycaffe

# Build fast rcnn lib
RUN cd /opt/butd/lib && make  

# Set ENV
ENV PYCAFFE_ROOT $CAFFE_ROOT/python
ENV PYTHONPATH $PYCAFFE_ROOT:$PYTHONPATH
ENV PATH $CAFFE_ROOT/build/tools:$PYCAFFE_ROOT:$PATH
RUN echo "$CAFFE_ROOT/build/lib" >> /etc/ld.so.conf.d/caffe.conf && ldconfig

WORKDIR /workspace
