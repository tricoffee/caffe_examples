
CFLAGS = -O3 -Wall -Wno-sign-compare -Wno-write-strings -Wno-unused-result -std=c++11 
IFLAGS = -I $(CAFFE_HOME)/include -I $(CAFFE_HOME)/build/src/ -I /usr/local/cuda/include `pkg-config --cflags opencv`
LFLAGS = -L $(CAFFE_HOME)/build/lib -lcaffe -L /usr/local/cuda/lib64 `pkg-config --libs opencv` -lglog -lboost_system

all:
	g++ -o train train.cpp read_mnist.cpp $(CFLAGS) $(IFLAGS) $(LFLAGS)
