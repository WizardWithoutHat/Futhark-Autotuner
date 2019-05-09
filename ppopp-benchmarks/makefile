FLAGS=FUTHARK_INCREMENTAL_FLATTENING=1

tune_exhaustive: ../exhaustive-auto.py backprop.fut lavaMD.fut LocVolCalib.fut nn.fut nw.fut pathfinder.fut srad.fut heston32.fut OptionPricing.fut
	$(FLAGS) python ../exhaustive-auto.py backprop.fut lavaMD.fut LocVolCalib.fut nn.fut nw.fut pathfinder.fut srad.fut heston32.fut OptionPricing.fut

moderate_flattening: backprop.fut lavaMD.fut LocVolCalib.fut nn.fut nw.fut pathfinder.fut srad.fut
	futhark bench --backend=opencl backprop.fut
	futhark bench --backend=opencl lavaMD.fut
	futhark bench --backend=opencl LocVolCalib.fut
	futhark bench --backend=opencl nn.fut
	futhark bench --backend=opencl nw.fut
	futhark bench --backend=opencl pathfinder.fut
	futhark bench --backend=opencl srad.fut
	futhark bench --backend=opencl heston32.fut
	futhark bench --backend=opencl OptionPricing.fut

untuned: backprop.fut lavaMD.fut LocVolCalib.fut nn.fut nw.fut pathfinder.fut srad.fut
	FUTHARK_INCREMENTAL_FLATTENING=1 futhark bench --backend=opencl backprop.fut
	FUTHARK_INCREMENTAL_FLATTENING=1 futhark bench --backend=opencl lavaMD.fut
	FUTHARK_INCREMENTAL_FLATTENING=1 futhark bench --backend=opencl LocVolCalib.fut
	FUTHARK_INCREMENTAL_FLATTENING=1 futhark bench --backend=opencl nn.fut
	FUTHARK_INCREMENTAL_FLATTENING=1 futhark bench --backend=opencl nw.fut
	FUTHARK_INCREMENTAL_FLATTENING=1 futhark bench --backend=opencl pathfinder.fut
	FUTHARK_INCREMENTAL_FLATTENING=1 futhark bench --backend=opencl srad.fut
	FUTHARK_INCREMENTAL_FLATTENING=1 futhark bench --backend=opencl heston32.fut
	FUTHARK_INCREMENTAL_FLATTENING=1 futhark bench --backend=opencl OptionPricing.fut

.PHONY: clean

clean:
	rm backprop lavaMD LocVolCalib nn nw pathfinder srad heston32 heston64 OptionPricing *.c *.actual *.expected
