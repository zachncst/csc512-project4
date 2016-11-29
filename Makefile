PROGS= norm
SM=
LD_LIBRARY_PATH=C:\Program Files (x86)\Windows Kits\10\Include\10.0.10240.0\ucrt

norm: norm.cu
	nvcc -o $@ $^ $(SM)

clean:
	rm $(PROGS)
