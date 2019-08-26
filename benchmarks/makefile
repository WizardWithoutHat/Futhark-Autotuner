FUTHARK=futhark
BACKEND=opencl
FLAGS=FUTHARK_INCREMENTAL_FLATTENING=1
PREFIX=gtx

variant_datasets: variant.fut
	futhark dataset -b --g [32][32]f32 --g [32][32]f32 > data/variant-data/32.in
	futhark dataset -b --g [64][64]f32 --g [64][64]f32 > data/variant-data/64.in
	futhark dataset -b --g [128][128]f32 --g [128][128]f32 > data/variant-data/128.in
	futhark dataset -b --g [256][256]f32 --g [256][256]f32 > data/variant-data/256.in
	futhark dataset -b --g [512][512]f32 --g [512][512]f32 > data/variant-data/512.in
	futhark dataset -b --g [1024][1024]f32 --g [1024][1024]f32 > data/variant-data/1024.in
	futhark dataset -b --g [2048][2048]f32 --g [2048][2048]f32 > data/variant-data/2048.in
	futhark c variant.fut
	./variant < data/variant-data/32.in > data/variant-data/32.out
	./variant < data/variant-data/64.in > data/variant-data/64.out
	./variant < data/variant-data/128.in > data/variant-data/128.out
	./variant < data/variant-data/256.in > data/variant-data/256.out
	./variant < data/variant-data/512.in > data/variant-data/512.out
	./variant < data/variant-data/1024.in > data/variant-data/1024.out
	./variant < data/variant-data/2048.in > data/variant-data/2048.out

runall_exhaustive: ../exhaustive-auto.py srad.fut LocVolCalib.fut bfast-ours.fut variant.fut lud-clean.fut
	$(FLAGS) python ../exhaustive-auto.py srad.fut LocVolCalib.fut bfast-ours.fut variant.fut lud-clean.fut

runall_cma_branch: ../cma-branch-auto.py srad.fut LocVolCalib.fut bfast-ours.fut variant.fut lud-clean.fut
	$(FLAGS) python ../cma-branch-auto.py srad.fut LocVolCalib.fut bfast-ours.fut variant.fut lud-clean.fut

runall_cma: ../cma-auto.py srad.fut LocVolCalib.fut bfast-ours.fut variant.fut lud-clean.fut
	$(FLAGS) python ../cma-auto.py srad.fut LocVolCalib.fut bfast-ours.fut variant.fut lud-clean.fut

run_loop_cma_L: ../cma-branch-auto.py lud-clean.fut
	$(FLAGS) python ../cma-branch-auto.py lud-clean.fut | tee $(PREFIX)-CMA1-L
	mv cma-lud-clean.json $(PREFIX)-cma-lud-clean1.json
	$(FLAGS) python ../cma-branch-auto.py lud-clean.fut | tee $(PREFIX)-CMA2-L
	mv cma-lud-clean.json $(PREFIX)-cma-lud-clean2.json
	$(FLAGS) python ../cma-branch-auto.py lud-clean.fut | tee $(PREFIX)-CMA3-L
	mv cma-lud-clean.json $(PREFIX)-cma-lud-clean3.json

run_loop_cmaindex_L: ../cma-branch-auto-index.py lud-clean.fut
	$(FLAGS) python ../cma-branch-auto-index.py lud-clean.fut  | tee $(PREFIX)-CMAINDEX1-L
	mv cmaindex-lud-clean.json $(PREFIX)-cmaindex-lud-clean1.json
	$(FLAGS) python ../cma-branch-auto-index.py lud-clean.fut  | tee $(PREFIX)-CMAINDEX2-L
	mv cmaindex-lud-clean.json $(PREFIX)-cmaindex-lud-clean2.json
	$(FLAGS) python ../cma-branch-auto-index.py lud-clean.fut  | tee $(PREFIX)-CMAINDEX3-L
	mv cmaindex-lud-clean.json $(PREFIX)-cmaindex-lud-clean3.json

run_loop_binary_L: ../binary-tuner.py lud-clean.fut
	$(FLAGS) python ../binary-tuner.py lud-clean.fut  | tee $(PREFIX)-BINARY1-L
	mv binary-lud-clean.json $(PREFIX)-binary-lud-clean1.json
	$(FLAGS) python ../binary-tuner.py lud-clean.fut  | tee $(PREFIX)-BINARY2-L
	mv binary-lud-clean.json $(PREFIX)-binary-lud-clean2.json
	$(FLAGS) python ../binary-tuner.py lud-clean.fut  | tee $(PREFIX)-BINARY3-L
	mv binary-lud-clean.json $(PREFIX)-binary-lud-clean3.json

run_loop_active_L: ../active-tuner.py lud-clean.fut
	$(FLAGS) python ../active-tuner.py lud-clean.fut | tee $(PREFIX)-ACTIVE1-L
	mv active-lud-clean.json $(PREFIX)-active-lud-clean1.json
	$(FLAGS) python ../active-tuner.py lud-clean.fut | tee $(PREFIX)-ACTIVE2-L
	mv active-lud-clean.json $(PREFIX)-active-lud-clean2.json
	$(FLAGS) python ../active-tuner.py lud-clean.fut | tee $(PREFIX)-ACTIVE3-L
	mv active-lud-clean.json $(PREFIX)-active-lud-clean3.json

run_loop_instr_L: ../instr-tuner.py lud-clean.fut
	$(FLAGS) python ../instr-tuner.py lud-clean.fut | tee $(PREFIX)-INSTR1-L
	mv instr-lud-clean.json $(PREFIX)-instr-lud-clean1.json
	$(FLAGS) python ../instr-tuner.py lud-clean.fut | tee $(PREFIX)-INSTR2-L
	mv instr-lud-clean.json $(PREFIX)-instr-lud-clean2.json
	$(FLAGS) python ../instr-tuner.py lud-clean.fut | tee $(PREFIX)-INSTR3-L
	mv instr-lud-clean.json $(PREFIX)-instr-lud-clean3.json



run_loop_cma_V: ../cma-branch-auto.py lud-clean.fut variant.fut
	$(FLAGS) python ../cma-branch-auto.py variant.fut | tee $(PREFIX)-CMA1-V
	mv cma-variant.json $(PREFIX)-cma-variant1.json
	$(FLAGS) python ../cma-branch-auto.py variant.fut | tee $(PREFIX)-CMA2-V
	mv cma-variant.json $(PREFIX)-cma-variant2.json
	$(FLAGS) python ../cma-branch-auto.py variant.fut | tee $(PREFIX)-CMA3-V
	mv cma-variant.json $(PREFIX)-cma-variant3.json

run_loop_cmaindex_V: ../cma-branch-auto-index.py lud-clean.fut variant.fut
	$(FLAGS) python ../cma-branch-auto-index.py variant.fut | tee $(PREFIX)-CMAINDEX1-V
	mv cmaindex-variant.json $(PREFIX)-cmaindex-variant1.json
	$(FLAGS) python ../cma-branch-auto-index.py variant.fut | tee $(PREFIX)-CMAINDEX2-V
	mv cmaindex-variant.json $(PREFIX)-cmaindex-variant2.json
	$(FLAGS) python ../cma-branch-auto-index.py variant.fut | tee $(PREFIX)-CMAINDEX3-V
	mv cmaindex-variant.json $(PREFIX)-cmaindex-variant3.json

run_loop_binary_V: ../binary-tuner.py lud-clean.fut variant.fut
	$(FLAGS) python ../binary-tuner.py variant.fut | tee $(PREFIX)-BINARY1-V
	mv binary-variant.json $(PREFIX)-binary-variant1.json
	$(FLAGS) python ../binary-tuner.py variant.fut | tee $(PREFIX)-BINARY2-V
	mv binary-variant.json $(PREFIX)-binary-variant2.json
	$(FLAGS) python ../binary-tuner.py variant.fut | tee $(PREFIX)-BINARY3-V
	mv binary-variant.json $(PREFIX)-binary-variant3.json

run_loop_active_V: ../active-tuner.py lud-clean.fut variant.fut
	$(FLAGS) python ../active-tuner.py variant.fut | tee $(PREFIX)-ACTIVE1-V
	mv active-variant.json $(PREFIX)-active-variant1.json
	$(FLAGS) python ../active-tuner.py variant.fut | tee $(PREFIX)-ACTIVE2-V
	mv active-variant.json $(PREFIX)-active-variant2.json
	$(FLAGS) python ../active-tuner.py variant.fut | tee $(PREFIX)-ACTIVE3-V
	mv active-variant.json $(PREFIX)-active-variant3.json

run_loop_instr_V: ../instr-tuner.py lud-clean.fut variant.fut
	$(FLAGS) python ../instr-tuner.py variant.fut | tee $(PREFIX)-INSTR1-V
	mv instr-variant.json $(PREFIX)-instr-variant1.json
	$(FLAGS) python ../instr-tuner.py variant.fut | tee $(PREFIX)-INSTR2-V
	mv instr-variant.json $(PREFIX)-instr-variant2.json
	$(FLAGS) python ../instr-tuner.py variant.fut | tee $(PREFIX)-INSTR3-V
	mv instr-variant.json $(PREFIX)-instr-variant3.json

all_V:
	make run_loop_cma_V
	make run_loop_cmaindex_V
	make run_loop_binary_V
	make run_loop_active_V
	make run_loop_instr_V

all_L:
	make run_loop_cma_L
	make run_loop_cmaindex_L
	make run_loop_binary_L
	make run_loop_active_L
	make run_loop_instr_L

all:
	make all_L
	make all_V

.PHONY: clean

clean:
	rm srad LocVolCalib bfast-ours variant lud-clean *.c *.actual *.expected
