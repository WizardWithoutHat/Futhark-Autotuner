FUTHARK=futhark
BACKEND=opencl
FLAGS=FUTHARK_INCREMENTAL_FLATTENING=1
PREFIX=gtx

runall_exhaustive: ../exhaustive-auto.py srad.fut LocVolCalib.fut bfast-ours.fut variant.fut lud-clean.fut
	$(FLAGS) python ../exhaustive-auto.py srad.fut LocVolCalib.fut bfast-ours.fut variant.fut lud-clean.fut

runall_cma_branch: ../cma-branch-auto.py srad.fut LocVolCalib.fut bfast-ours.fut variant.fut lud-clean.fut
	$(FLAGS) python ../cma-branch-auto.py srad.fut LocVolCalib.fut bfast-ours.fut variant.fut lud-clean.fut

runall_cma: ../cma-auto.py srad.fut LocVolCalib.fut bfast-ours.fut variant.fut lud-clean.fut
	$(FLAGS) python ../cma-auto.py srad.fut LocVolCalib.fut bfast-ours.fut variant.fut lud-clean.fut

run_loop_cma: ../cma-branch-auto.py lud-clean.fut variant.fut 
	$(FLAGS) python ../cma-branch-auto.py lud-clean.fut variant.fut
	mv cma-lud-clean.json $(PREFIX)-cma-lud-clean1.json
	mv cma-variant.json $(PREFIX)-cma-variant1.json
	$(FLAGS) python ../cma-branch-auto.py lud-clean.fut variant.fut
	mv cma-lud-clean.json $(PREFIX)-cma-lud-clean2.json
	mv cma-variant.json $(PREFIX)-cma-variant2.json
	$(FLAGS) python ../cma-branch-auto.py lud-clean.fut variant.fut
	mv cma-lud-clean.json $(PREFIX)-cma-lud-clean3.json
	mv cma-variant.json $(PREFIX)-cma-variant3.json

run_loop_cmaindex: ../cma-branch-auto-index.py lud-clean.fut variant.fut 
	$(FLAGS) python ../cma-branch-auto-index.py lud-clean.fut variant.fut
	mv cmaindex-lud-clean.json $(PREFIX)-cmaindex-lud-clean1.json
	mv cmaindex-variant.json $(PREFIX)-cmaindex-variant1.json
	$(FLAGS) python ../cma-branch-auto-index.py lud-clean.fut variant.fut
	mv cmaindex-lud-clean.json $(PREFIX)-cmaindex-lud-clean2.json
	mv cmaindex-variant.json $(PREFIX)-cmaindex-variant2.json
	$(FLAGS) python ../cma-branch-auto-index.py lud-clean.fut variant.fut
	mv cmaindex-lud-clean.json $(PREFIX)-cmaindex-lud-clean3.json
	mv cmaindex-variant.json $(PREFIX)-cmaindex-variant3.json

run_loop_binary: ../binary-tuner.py lud-clean.fut variant.fut 
	$(FLAGS) python ../binary-tuner.py lud-clean.fut variant.fut
	mv binary-lud-clean.json $(PREFIX)-binary-lud-clean1.json
	mv binary-variant.json $(PREFIX)-binary-variant1.json
	$(FLAGS) python ../binary-tuner.py lud-clean.fut variant.fut
	mv binary-lud-clean.json $(PREFIX)-binary-lud-clean2.json
	mv binary-variant.json $(PREFIX)-binary-variant2.json
	$(FLAGS) python ../binary-tuner.py lud-clean.fut variant.fut
	mv binary-lud-clean.json $(PREFIX)-binary-lud-clean3.json
	mv binary-variant.json $(PREFIX)-binary-variant3.json

run_loop_active: ../active-tuner.py lud-clean.fut variant.fut 
	$(FLAGS) python ../active-tuner.py lud-clean.fut variant.fut
	mv active-lud-clean.json $(PREFIX)-active-lud-clean1.json
	mv active-variant.json $(PREFIX)-active-variant1.json
	$(FLAGS) python ../active-tuner.py lud-clean.fut variant.fut
	mv active-lud-clean.json $(PREFIX)-active-lud-clean2.json
	mv active-variant.json $(PREFIX)-active-variant2.json
	$(FLAGS) python ../active-tuner.py lud-clean.fut variant.fut
	mv active-lud-clean.json $(PREFIX)-active-lud-clean3.json
	mv active-variant.json $(PREFIX)-active-variant3.json

run_loop_instr: ../instr-tuner.py lud-clean.fut variant.fut 
	$(FLAGS) python ../instr-tuner.py lud-clean.fut variant.fut
	mv instr-lud-clean.json $(PREFIX)-instr-lud-clean1.json
	mv instr-variant.json $(PREFIX)-instr-variant1.json
	$(FLAGS) python ../instr-tuner.py lud-clean.fut variant.fut
	mv instr-lud-clean.json $(PREFIX)-instr-lud-clean2.json
	mv instr-variant.json $(PREFIX)-instr-variant2.json
	$(FLAGS) python ../instr-tuner.py lud-clean.fut variant.fut
	mv instr-lud-clean.json $(PREFIX)-instr-lud-clean3.json
	mv instr-variant.json $(PREFIX)-instr-variant3.json


.PHONY: clean

clean:
	rm srad LocVolCalib bfast-ours variant lud-clean *.c *.actual *.expected
