default: 
	riscv64-unknown-elf-gcc main.c PackA.c PackB.c ./Gemm_4x4Kernel_RVV_inline.c -march=rv64gv -static -o gemm_rvv
