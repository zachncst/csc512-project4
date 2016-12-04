PROGS= norm opt1 opt2 op3 optAll
SM=
INCLUDES=
OPTS=
#OPTS=-arch=compute_30 -code="sm_30,compute_30" 

all: norm opt1 opt2 opt3 optAll

norm: norm.cu
	nvcc $(OPTS) $(INCLUDES) -o $@ $^ $(SM)

opt1: opt1.cu
	nvcc $(OPTS) $(INCLUDES) -o $@ $^ $(SM)

opt2: opt2.cu
	nvcc $(OPTS) $(INCLUDES) -o $@ $^ $(SM)

opt3: opt3.cu
	nvcc $(OPTS) $(INCLUDES) -o $@ $^ $(SM)

optAll: optAll.cu
	nvcc $(OPTS) $(INCLUDES) -o $@ $^ $(SM)

clean:
	rm $(PROGS)
