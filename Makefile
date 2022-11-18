CFLAGS=-O3 -I$(BLAS_INC) -m64 -mavx2 -std=c99 -march=native -fopenmp -D_POSIX_C_SOURCE=200809L

default: 
	riscv64-unknown-elf-gcc main.c PackA.c PackB.c ./Gemm_4x4Kernel_RVV_inline.c -march=rv64gv -static -o gemm_rvv

intel:
	gcc PackA.c PackB.c Gemm_4x4Kernel_Packed_Intel.c main.c $(CFLAGS) -o gemm_intel
