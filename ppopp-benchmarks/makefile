FLAGS=FUTHARK_INCREMENTAL_FLATTENING=1

tune_exhaustive: ../exhaustive-auto.py backprop.fut lavaMD.fut LocVolCalib.fut nn.fut nw.fut pathfinder.fut srad.fut heston32.fut OptionPricing.fut
	$(FLAGS) python ../exhaustive-auto.py backprop.fut lavaMD.fut LocVolCalib.fut nn.fut nw.fut pathfinder.fut srad.fut heston32.fut OptionPricing.fut

moderate_flattening: backprop.fut lavaMD.fut LocVolCalib.fut nn.fut nw.fut pathfinder.fut srad.fut
	futhark bench --backend=opencl backprop.fut --exclude-case=tune --json=moderate-backprop.json
	futhark bench --backend=opencl lavaMD.fut --exclude-case=tune --json=moderate-lavaMD.json
	futhark bench --backend=opencl LocVolCalib.fut --exclude-case=tune --json=moderate-LocVolCalib.json
	futhark bench --backend=opencl nn.fut --exclude-case=tune --json=moderate-nn.json
	futhark bench --backend=opencl nw.fut --exclude-case=tune --json=moderate-nw.json
	futhark bench --backend=opencl pathfinder.fut --exclude-case=tune --json=moderate-pathfinder.json
	futhark bench --backend=opencl srad.fut --exclude-case=tune --json=moderate-srad.json
	futhark bench --backend=opencl heston32.fut --exclude-case=tune --json=moderate-heston32.json
	futhark bench --backend=opencl OptionPricing.fut --exclude-case=tune --json=moderate-OptionPricing.json

untuned: backprop.fut lavaMD.fut LocVolCalib.fut nn.fut nw.fut pathfinder.fut srad.fut
	FUTHARK_INCREMENTAL_FLATTENING=1 futhark bench --backend=opencl backprop.fut --exclude-case=tune --json=untuned-backprop.json
	FUTHARK_INCREMENTAL_FLATTENING=1 futhark bench --backend=opencl lavaMD.fut --exclude-case=tune --json=untuned-lavaMD.json
	FUTHARK_INCREMENTAL_FLATTENING=1 futhark bench --backend=opencl LocVolCalib.fut --exclude-case=tune --json=untuned-LocVolCalib.json
	FUTHARK_INCREMENTAL_FLATTENING=1 futhark bench --backend=opencl nn.fut --exclude-case=tune --json=untuned-nn.json
	FUTHARK_INCREMENTAL_FLATTENING=1 futhark bench --backend=opencl nw.fut --exclude-case=tune --json=untuned-nw.json
	FUTHARK_INCREMENTAL_FLATTENING=1 futhark bench --backend=opencl pathfinder.fut --exclude-case=tune --json=untuned-pathfinder.json
	FUTHARK_INCREMENTAL_FLATTENING=1 futhark bench --backend=opencl srad.fut --exclude-case=tune --json=untuned-srad.json
	FUTHARK_INCREMENTAL_FLATTENING=1 futhark bench --backend=opencl heston32.fut --exclude-case=tune --json=untuned-heston32.json
	FUTHARK_INCREMENTAL_FLATTENING=1 futhark bench --backend=opencl OptionPricing.fut --exclude-case=tune --json=untuned-OptionPricing.json

.PHONY: clean

clean:
	rm backprop lavaMD LocVolCalib nn nw pathfinder srad heston32 heston64 OptionPricing *.c *.actual *.expected
