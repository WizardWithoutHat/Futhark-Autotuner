FUTHARK=futhark
BACKEND=opencl
FLAGS=FUTHARK_INCREMENTAL_FLATTENING=1

tune_exhaustive_0: ../Futhark-Autotuner/exhaustive-auto.py backprop.fut lavaMD.fut LocVolCalib.fut nn.fut nw.fut pathfinder.fut srad.fut
	$(FLAGS) CUDA_VISIBLE_DEVICES=0 python ../Futhark-Autotuner/exhaustive-auto.py backprop.fut lavaMD.fut LocVolCalib.fut nn.fut nw.fut pathfinder.fut srad.fut

tune_exhaustive_1: ../Futhark-Autotuner/exhaustive-auto.py backprop.fut lavaMD.fut LocVolCalib.fut nn.fut nw.fut pathfinder.fut srad.fut
	$(FLAGS) CUDA_VISIBLE_DEVICES=1 python ../Futhark-Autotuner/exhaustive-auto.py backprop.fut lavaMD.fut LocVolCalib.fut nn.fut nw.fut pathfinder.fut srad.fut

moderate_flattening: backprop.fut lavaMD.fut LocVolCalib.fut nn.fut nw.fut pathfinder.fut srad.fut
	futhark bench --backend=opencl backprop.fut
	futhark bench --backend=opencl lavaMD.fut
	futhark bench --backend=opencl LocVolCalib.fut
	futhark bench --backend=opencl nn.fut
	futhark bench --backend=opencl nw.fut
	futhark bench --backend=opencl pathfinder.fut
	futhark bench --backend=opencl srad.fut

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
