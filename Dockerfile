# debian:stretch-20200130
FROM debian@sha256:872b72a2b8487e4b91ae27855c7de1671635d3dc2cc0b89651103e55c74ed34a AS base

FROM base as build
RUN apt-get update && apt-get -y install --no-install-recommends \
	gcc \
	g++ \
	gfortran \
	libopenblas-dev \
	libblas-dev \
	liblapack-dev \
	libatlas-base-dev \
	libhdf5-dev \
	libhdf5-100 \
	pkg-config \
	python3 \
	python3-dev \
	python3-pip \
	python3-setuptools \
	pybind11-dev \
	wget

RUN pip3 install wheel==0.34.2 cython==0.29.14 pybind11==2.4.3
RUN pip3 wheel numpy==1.18.1 && pip3 install numpy-*.whl
RUN pip3 wheel scipy==1.4.1
RUN pip3 wheel --no-deps h5py==2.10.0





FROM base
RUN apt-get update && apt-get -y install --no-install-recommends \
	python3-pip \
	python3-setuptools \
	libopenblas-base \
	wget \
&& rm -rf /var/lib/apt/lists/*
COPY --from=build /*.whl /
RUN pip3 install *.whl
RUN wget https://storage.googleapis.com/tensorflow/raspberrypi/tensorflow-2.3.0-cp35-none-linux_armv7l.whl  

RUN pip3 install tensorflow-*.whl

RUN rm *.whl
