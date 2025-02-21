FLAGS=FUTHARK_INCREMENTAL_FLATTENING=1
PREFIX=gtx

compile: backprop.fut lavaMD.fut LocVolCalib.fut nn.fut nw.fut pathfinder.fut srad.fut heston32.fut OptionPricing.fut bfast.fut
	$(FLAGS) ~/futhark-0.10/bin/futhark opencl bfast.fut
	$(FLAGS) futhark opencl backprop.fut
	$(FLAGS) futhark opencl lavaMD.fut
	$(FLAGS) futhark opencl LocVolCalib.fut
	$(FLAGS) futhark opencl nn.fut
	$(FLAGS) futhark opencl nw.fut
	$(FLAGS) futhark opencl pathfinder.fut
	$(FLAGS) futhark opencl srad.fut
	$(FLAGS) futhark opencl heston32.fut
	$(FLAGS) futhark opencl OptionPricing.fut

tune_exhaustive: ../instr-tuner.py backprop.fut lavaMD.fut LocVolCalib.fut nn.fut nw.fut pathfinder.fut srad.fut heston32.fut OptionPricing.fut bfast
	$(FLAGS) python ../instr-tuner.py backprop.fut lavaMD.fut LocVolCalib.fut nn.fut nw.fut pathfinder.fut srad.fut heston32.fut OptionPricing.fut bfast.fut
	mv instr-backprop.json $(PREFIX)-simple-backprop.json
	mv instr-lavaMD.json $(PREFIX)-simple-lavaMD.json
	mv instr-LocVolCalib.json $(PREFIX)-simple-LocVolCalib.json
	mv instr-nn.json $(PREFIX)-simple-nn.json
	mv instr-nw.json $(PREFIX)-simple-nw.json
	mv instr-pathfinder.json $(PREFIX)-simple-pathfinder.json
	mv instr-srad.json $(PREFIX)-simple-srad.json
	mv instr-heston32.json $(PREFIX)-simple-heston32.json
	mv instr-OptionPricing.json $(PREFIX)-simple-OptionPricing.json
	mv instr-bfast.json $(PREFIX)-simple-bfast.json

reference: ../instr-tuner.py backprop.fut lavaMD.fut LocVolCalib.fut nn.fut nw.fut pathfinder.fut srad.fut heston32.fut OptionPricing.fut bfast
	$(FLAGS) python ../instr-tuner.py backprop.fut lavaMD.fut LocVolCalib.fut nn.fut nw.fut pathfinder.fut srad.fut heston32.fut OptionPricing.fut bfast.fut
	mv instr-backprop.json $(PREFIX)-reference-backprop.json
	mv instr-lavaMD.json $(PREFIX)-reference-lavaMD.json
	mv instr-LocVolCalib.json $(PREFIX)-reference-LocVolCalib.json
	mv instr-nn.json $(PREFIX)-reference-nn.json
	mv instr-nw.json $(PREFIX)-reference-nw.json
	mv instr-pathfinder.json $(PREFIX)-reference-pathfinder.json
	mv instr-srad.json $(PREFIX)-reference-srad.json
	mv instr-heston32.json $(PREFIX)-reference-heston32.json
	mv instr-OptionPricing.json $(PREFIX)-reference-OptionPricing.json
	mv instr-bfast.json $(PREFIX)-reference-bfast.json

moderate_flattening: backprop.fut lavaMD.fut LocVolCalib.fut nn.fut nw.fut pathfinder.fut srad.fut bfast.fut
	futhark bench --backend=opencl backprop.fut --exclude-case=tune --json=$(PREFIX)-moderate-backprop.json
	futhark bench --backend=opencl lavaMD.fut --exclude-case=tune --json=$(PREFIX)-moderate-lavaMD.json
	futhark bench --backend=opencl LocVolCalib.fut --exclude-case=tune --json=$(PREFIX)-moderate-LocVolCalib.json
	futhark bench --backend=opencl nn.fut --exclude-case=tune --json=$(PREFIX)-moderate-nn.json --skip-compilation
	futhark bench --backend=opencl nw.fut --exclude-case=tune --json=$(PREFIX)-moderate-nw.json --skip-compilation
	futhark bench --backend=opencl pathfinder.fut --exclude-case=tune --json=$(PREFIX)-moderate-pathfinder.json
	futhark bench --backend=opencl srad.fut --exclude-case=tune --json=$(PREFIX)-moderate-srad.json
	futhark bench --backend=opencl heston32.fut --exclude-case=tune --json=$(PREFIX)-moderate-heston32.json 
	futhark bench --backend=opencl OptionPricing.fut --exclude-case=tune --json=$(PREFIX)-moderate-OptionPricing.json
	~/futhark-0.10/bin/futhark opencl bfast.fut
	futhark bench --backend=opencl bfast.fut --exclude-case=tune --json=$(PREFIX)-moderate-bfast.json --skip-compilation
	rm bfast

untuned: backprop.fut lavaMD.fut LocVolCalib.fut nn.fut nw.fut pathfinder.fut srad.fut bfast
	FUTHARK_INCREMENTAL_FLATTENING=1 futhark bench --backend=opencl backprop.fut --exclude-case=tune --json=$(PREFIX)-untuned-backprop.json --skip-compilation
	FUTHARK_INCREMENTAL_FLATTENING=1 futhark bench --backend=opencl lavaMD.fut --exclude-case=tune --json=$(PREFIX)-untuned-lavaMD.json --skip-compilation
	FUTHARK_INCREMENTAL_FLATTENING=1 futhark bench --backend=opencl LocVolCalib.fut --exclude-case=tune --json=$(PREFIX)-untuned-LocVolCalib.json --skip-compilation
	FUTHARK_INCREMENTAL_FLATTENING=1 futhark bench --backend=opencl nn.fut --exclude-case=tune --json=$(PREFIX)-untuned-nn.json --skip-compilation
	FUTHARK_INCREMENTAL_FLATTENING=1 futhark bench --backend=opencl nw.fut --exclude-case=tune --json=$(PREFIX)-untuned-nw.json --skip-compilation
	FUTHARK_INCREMENTAL_FLATTENING=1 futhark bench --backend=opencl pathfinder.fut --exclude-case=tune --json=$(PREFIX)-untuned-pathfinder.json --skip-compilation
	FUTHARK_INCREMENTAL_FLATTENING=1 futhark bench --backend=opencl srad.fut --exclude-case=tune --json=$(PREFIX)-untuned-srad.json --skip-compilation
	FUTHARK_INCREMENTAL_FLATTENING=1 futhark bench --backend=opencl heston32.fut --exclude-case=tune --json=$(PREFIX)-untuned-heston32.json --skip-compilation
	FUTHARK_INCREMENTAL_FLATTENING=1 futhark bench --backend=opencl OptionPricing.fut --exclude-case=tune --json=$(PREFIX)-untuned-OptionPricing.json --skip-compilation
	FUTHARK_INCREMENTAL_FLATTENING=1 futhark bench --backend=opencl bfast.fut --exclude-case=tune --json=$(PREFIX)-untuned-bfast.json --skip-compilation

all:
	make compile
	make tune_exhaustive
	make untuned
	make moderate_flattening

.PHONY: clean

clean:
	rm backprop lavaMD LocVolCalib nn nw pathfinder srad heston32 heston64 OptionPricing *.c *.actual *.expected *.json bfast
