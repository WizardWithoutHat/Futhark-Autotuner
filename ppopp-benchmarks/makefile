tune_exhaustive: ../exhaustive-auto.py backprop.fut lavaMD.fut LocVolCalib.fut nn.fut nw.fut pathfinder.fut srad.fut
	$(FLAGS) python ../exhaustive-auto.py backprop.fut lavaMD.fut LocVolCalib.fut nn.fut nw.fut pathfinder.fut srad.fut

moderate_flattening: backprop.fut lavaMD.fut LocVolCalib.fut nn.fut nw.fut pathfinder.fut srad.fut
	FUTHARK_INCREMENTAL_FLATTENING=0 futhark bench --backend=opencl backprop.fut
	FUTHARK_INCREMENTAL_FLATTENING=0 futhark bench --backend=opencl lavaMD.fut
	FUTHARK_INCREMENTAL_FLATTENING=0 futhark bench --backend=opencl LocVolCalib.fut
	FUTHARK_INCREMENTAL_FLATTENING=0 futhark bench --backend=opencl nn.fut
	FUTHARK_INCREMENTAL_FLATTENING=0 futhark bench --backend=opencl nw.fut
	FUTHARK_INCREMENTAL_FLATTENING=0 futhark bench --backend=opencl pathfinder.fut
	FUTHARK_INCREMENTAL_FLATTENING=0 futhark bench --backend=opencl srad.fut

untuned: backprop.fut lavaMD.fut LocVolCalib.fut nn.fut nw.fut pathfinder.fut srad.fut
	FUTHARK_INCREMENTAL_FLATTENING=1 futhark bench --backend=opencl backprop.fut
	FUTHARK_INCREMENTAL_FLATTENING=1 futhark bench --backend=opencl lavaMD.fut
	FUTHARK_INCREMENTAL_FLATTENING=1 futhark bench --backend=opencl LocVolCalib.fut
	FUTHARK_INCREMENTAL_FLATTENING=1 futhark bench --backend=opencl nn.fut
	FUTHARK_INCREMENTAL_FLATTENING=1 futhark bench --backend=opencl nw.fut
	FUTHARK_INCREMENTAL_FLATTENING=1 futhark bench --backend=opencl pathfinder.fut
	FUTHARK_INCREMENTAL_FLATTENING=1 futhark bench --backend=opencl srad.fut

.PHONY: clean

clean:
	rm backprop lavaMD LocVolCalib nn nw pathfinder srad *.c *.actual *.expected
