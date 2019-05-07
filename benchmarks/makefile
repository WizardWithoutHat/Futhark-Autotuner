FUTHARK=futhark
BACKEND=opencl
FLAGS=FUTHARK_INCREMENTAL_FLATTENING=1

runall_exhaustive: ../exhaustive-auto.py srad.fut LocVolCalib.fut bfast-ours.fut variant.fut lud-clean.fut
	$(FLAGS) python ../exhaustive-auto.py srad.fut LocVolCalib.fut bfast-ours.fut variant.fut lud-clean.fut

runall_cma_branch: ../cma-branch-auto.py srad.fut LocVolCalib.fut bfast-ours.fut variant.fut lud-clean.fut
	$(FLAGS) python ../cma-branch-auto.py srad.fut LocVolCalib.fut bfast-ours.fut variant.fut lud-clean.fut

runall_cma: ../cma-auto.py srad.fut LocVolCalib.fut bfast-ours.fut variant.fut lud-clean.fut
	$(FLAGS) python ../cma-auto.py srad.fut LocVolCalib.fut bfast-ours.fut variant.fut lud-clean.fut

.PHONY: clean

clean:
	rm srad LocVolCalib bfast-ours variant lud-clean *.c *.actual *.expected
