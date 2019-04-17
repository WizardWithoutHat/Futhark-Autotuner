FUTHARK=futhark
BACKEND=opencl
FLAGS=FUTHARK_INCREMENTAL_FLATTENING=1

runall_exhaustive_0: ../exhaustive-auto.py srad.fut LocVolCalib.fut bfast-ours.fut variant.fut lud-clean.fut
	$(FLAGS) CUDA_VISIBLE_DEVICES=0 python ../exhaustive-auto.py srad.fut LocVolCalib.fut bfast-ours.fut variant.fut lud-clean.fut

runall_exhaustive_1: ../exhaustive-auto.py srad.fut LocVolCalib.fut bfast-ours.fut variant.fut lud-clean.fut
	$(FLAGS) CUDA_VISIBLE_DEVICES=1 python ../exhaustive-auto.py srad.fut LocVolCalib.fut bfast-ours.fut variant.fut lud-clean.fut

runall_cma_branch_0: ../cma-branch-auto.py srad.fut LocVolCalib.fut bfast-ours.fut variant.fut lud-clean.fut
	$(FLAGS) CUDA_VISIBLE_DEVICES=0 python ../cma-branch-auto.py srad.fut LocVolCalib.fut bfast-ours.fut variant.fut lud-clean.fut

runall_cma_branch_1: ../cma-branch-auto.py srad.fut LocVolCalib.fut bfast-ours.fut variant.fut lud-clean.fut
	$(FLAGS) CUDA_VISIBLE_DEVICES=1 python ../cma-branch-auto.py srad.fut LocVolCalib.fut bfast-ours.fut variant.fut lud-clean.fut

runall_cma_0: ../cma-auto.py srad.fut LocVolCalib.fut bfast-ours.fut variant.fut lud-clean.fut
	$(FLAGS) CUDA_VISIBLE_DEVICES=0 python ../cma-auto.py srad.fut LocVolCalib.fut bfast-ours.fut variant.fut lud-clean.fut

runall_cma_1: ../cma-auto.py srad.fut LocVolCalib.fut bfast-ours.fut variant.fut lud-clean.fut
	$(FLAGS) CUDA_VISIBLE_DEVICES=1 python ../cma-auto.py srad.fut LocVolCalib.fut bfast-ours.fut variant.fut lud-clean.fut

.PHONY: clean

clean:
	rm srad LocVolCalib bfast-ours variant lud-clean *.c *.actual *.expected
