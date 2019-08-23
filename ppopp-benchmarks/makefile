FLAGS=FUTHARK_INCREMENTAL_FLATTENING=1
PREFIX=gtx

tune_exhaustive: ../exhaustive-auto.py backprop.fut lavaMD.fut LocVolCalib.fut nn.fut nw.fut pathfinder.fut srad.fut heston32.fut OptionPricing.fut bfast.fut
	$(FLAGS) python ../exhaustive-auto.py backprop.fut lavaMD.fut LocVolCalib.fut nn.fut nw.fut pathfinder.fut srad.fut heston32.fut OptionPricing.fut bfast.fut
	mv simple-backprop.json $(PREFIX)-simple-backprop.json
	mv simple-lavaMD.json $(PREFIX)-simple-lavaMD.json
	mv simple-LocVolCalib.json $(PREFIX)-simple-LocVolCalib.json
	mv simple-nn.json $(PREFIX)-simple-nn.json
	mv simple-nw.json $(PREFIX)-simple-nw.json
	mv simple-pathfinder.json $(PREFIX)-simple-pathfinder.json
	mv simple-srad.json $(PREFIX)-simple-srad.json
	mv simple-OptionPricing.json $(PREFIX)-simple-OptionPricing.json
	mv simple-bfast.json $(PREFIX)-simple-bfast.json

moderate_flattening: backprop.fut lavaMD.fut LocVolCalib.fut nn.fut nw.fut pathfinder.fut srad.fut bfast.fut
	futhark bench --backend=opencl backprop.fut --exclude-case=tune --json=$(PREFIX)-moderate-backprop.json
	futhark bench --backend=opencl lavaMD.fut --exclude-case=tune --json=$(PREFIX)-moderate-lavaMD.json
	futhark bench --backend=opencl LocVolCalib.fut --exclude-case=tune --json=$(PREFIX)-moderate-LocVolCalib.json
	futhark bench --backend=opencl nn.fut --exclude-case=tune --json=$(PREFIX)-moderate-nn.json
	futhark bench --backend=opencl nw.fut --exclude-case=tune --json=$(PREFIX)-moderate-nw.json
	futhark bench --backend=opencl pathfinder.fut --exclude-case=tune --json=$(PREFIX)-moderate-pathfinder.json
	futhark bench --backend=opencl srad.fut --exclude-case=tune --json=$(PREFIX)-moderate-srad.json
	futhark bench --backend=opencl heston32.fut --exclude-case=tune --json=$(PREFIX)-moderate-heston32.json
	futhark bench --backend=opencl OptionPricing.fut --exclude-case=tune --json=$(PREFIX)-moderate-OptionPricing.json
	futhark bench --backend=opencl bfast.fut --exclude-case=tune --json=$(PREFIX)-moderate-bfast.json

untuned: backprop.fut lavaMD.fut LocVolCalib.fut nn.fut nw.fut pathfinder.fut srad.fut
	FUTHARK_INCREMENTAL_FLATTENING=1 futhark bench --backend=opencl backprop.fut --exclude-case=tune --json=$(PREFIX)-untuned-backprop.json
	FUTHARK_INCREMENTAL_FLATTENING=1 futhark bench --backend=opencl lavaMD.fut --exclude-case=tune --json=$(PREFIX)-untuned-lavaMD.json
	FUTHARK_INCREMENTAL_FLATTENING=1 futhark bench --backend=opencl LocVolCalib.fut --exclude-case=tune --json=$(PREFIX)-untuned-LocVolCalib.json
	FUTHARK_INCREMENTAL_FLATTENING=1 futhark bench --backend=opencl nn.fut --exclude-case=tune --json=$(PREFIX)-untuned-nn.json
	FUTHARK_INCREMENTAL_FLATTENING=1 futhark bench --backend=opencl nw.fut --exclude-case=tune --json=$(PREFIX)-untuned-nw.json
	FUTHARK_INCREMENTAL_FLATTENING=1 futhark bench --backend=opencl pathfinder.fut --exclude-case=tune --json=$(PREFIX)-untuned-pathfinder.json
	FUTHARK_INCREMENTAL_FLATTENING=1 futhark bench --backend=opencl srad.fut --exclude-case=tune --json=$(PREFIX)-untuned-srad.json
	FUTHARK_INCREMENTAL_FLATTENING=1 futhark bench --backend=opencl heston32.fut --exclude-case=tune --json=$(PREFIX)-untuned-heston32.json
	FUTHARK_INCREMENTAL_FLATTENING=1 futhark bench --backend=opencl OptionPricing.fut --exclude-case=tune --json=$(PREFIX)-untuned-OptionPricing.json
	FUTHARK_INCREMENTAL_FLATTENING=1 futhark bench --backend=opencl bfast.fut --exclude-case=tune --json=$(PREFIX)-untuned-bfast.json


.PHONY: clean

clean:
	rm backprop lavaMD LocVolCalib nn nw pathfinder srad heston32 heston64 OptionPricing *.c *.actual *.expected *.json
