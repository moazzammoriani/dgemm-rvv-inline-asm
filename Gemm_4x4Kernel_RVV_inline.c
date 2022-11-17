#define alpha( i,j ) A[ (j)*ldA + (i) ]   // map alpha( i,j ) to array A
#define beta( i,j )  B[ (j)*ldB + (i) ]   // map beta( i,j ) to array B
#define gamma( i,j ) C[ (j)*ldC + (i) ]   // map gamma( i,j ) to array C

#define MR 4
#define NR 4

#include<stdio.h>
#include<riscv_vector.h>


void Gemm_MRxNRKernel_Packed( int k,
		        double *MP_A, double *MP_B, double *C, int ldC )
{

    double *C0 = C;
    double *C1 = C0 + ldC;
    double *C2 = C1 + ldC;
    double *C3 = C2 + ldC;


    double alpha = 1.0;

            asm volatile(

                "mv            t0, zero        \n\t"
                "add           t0, t0,2        \n\t"
                "vsetvli       t0, t0, e64     \n\t"
                "fmv.w.x    ft11, zero         \n\t"
                "mv         t0,   %[BK]        \n\t"

                "vfmv.v.f   v16,  ft11         \n\t" // v16[i] = ft11              // Column 0 of microkernel C
                "vfmv.v.f   v17,  ft11         \n\t" // v17[i] = ft11

                "vfmv.v.f   v20,  ft11         \n\t" // v20[i] = ft11              // Column 1 of microkernel C
                "vfmv.v.f   v21,  ft11         \n\t" // v21[i] = ft11

                "vfmv.v.f   v24,  ft11         \n\t" // v24[i] = ft11              // Column 2 of microkernel C
                "vfmv.v.f   v25,  ft11         \n\t" // v25[i] = ft11

                "vfmv.v.f   v28,  ft11         \n\t" // v28[i] = ft11              // Column 3 of microkernel C
                "vfmv.v.f   v29,  ft11         \n\t" // v29[i] = ft11

                ".align 4                      \n\t"
                "M4x4_TAILLOOP:                \n\t"
                "fld        ft0,  (%[PB])      \n\t" // ft0 = (%[PB])            // ft0 = B[0, 0]
                "addi       %[PB], %[PB], 4*8  \n\t" // %[PB] = %[PB] + (4*8)    // B[1, 0]
                "vle64.v      v0,   (%[PA])    \n\t" // v0[0 : 2] = (%[PA])      // v0[0 : 2] = A[0 : 2, 0]
                "add        %[PA], %[PA], 8*8  \n\t" // %[PA] += 8*8             // A[0 , 1]
                "vle64.v      v1,   (t4)       \n\t" // v1[0 : 2] = (t4)         // v1[0 : 2] = A[2 : 4, 0]
                "addi       t4,    t4,    8*8  \n\t" // t4 = t4 + (8*8)          // A[2, 1] 

                "vfmv.v.f   v8,   ft0          \n\t" // v8[i] = ft0              // v8[i] = B[0, 0]
                "fld        ft1,  (t1)         \n\t" // ft1 = (t1)               // ft1 = B[0, 1]
                "addi       t1,   t1,     4*8  \n\t" // t1 = t1 + (4*8)          // B[1, 1]

                "vfmacc.vv  v16,  v8,    v0    \n\t" // v16[i] = (v8[i] * v0[i]) + v16[i]    // B[0, 0] * A[0 : 2, 0] + v16[i] 
                "fld        ft2,  (t2)         \n\t" // ft2 = (t2)                           // ft2 = B[0, 2]
                "addi       t2,   t2,     4*8  \n\t" // t2 = t2 + (4*8)                      // B[1, 2]
                "vfmacc.vv  v17,  v8,    v1    \n\t" // v17[i] = (v8[i] * v1[i]) + v17[i]    // B[0, 0] * A[2 : 4, 0] + v17[i]
                "vfmv.v.f   v9,   ft1          \n\t" // v9[i] = ft1                          // v9[i] = B[0, 1]

                "vfmacc.vv  v20,  v9,    v0    \n\t" // v20[i] = (v9[i] * v0[i]) + v20[i]    // B[0, 1] * A[0 : 2, 0] + v20[i]  
                "fld        ft3,  (t3)         \n\t" // ft3 = (t3)                           // B[0, 3]
                "addi       t3,   t3,     4*8  \n\t" // t3 = t3 + (4*8)
                "vfmacc.vv  v21,  v9,    v1    \n\t" // v21[i] = (v9[i] * v1[i]) + v21[i]    // B[0, 1] * A[2 : 4, 0] + v21[i]
                "vfmv.v.f   v10,  ft2          \n\t" // v10[i] = ft2                         // v10[i] = B[0, 2]

                "vfmv.v.f   v11,  ft3          \n\t"  // v11[i] = ft3                        // v11[i] = B[0, 3]
                "vfmacc.vv  v24,  v10,    v0    \n\t" // v24[i] = (v10[i] * v0[i]) + v24[i]  // B[0, 2] * A[0 : 2, 0] + v24[i]
                "vfmacc.vv  v25,  v10,    v1    \n\t" // v25[i] = (v10[i] * v1[i]) + v25[i]  // B[0, 2] * A[2 : 4, 0] + v25[i]

                "vfmacc.vv  v28,  v11,    v0    \n\t" // v28[i] = (v11[i] * v0[i]) + v28[i]  // B[0, 3] * A[0 : 2, 0] + v28[i]
                "vfmacc.vv  v29,  v11,    v1    \n\t" // v29[i] = (v11[i] * v1[i]) + v29[i]  // B[0, 3] * A[2 : 4, 0] + v29[i]

                "addi       t0,   t0, -1       \n\t" // t0 = t0 + (-1)
                "bgtz       t0,   M4x4_TAILLOOP \n\t" // If t0 > 0 goto M4x4_MAINLOOP

                // Save result
                // load C
                "M4x4_SAVERESULT:              \n\t"
                // use v8 to store alpha
                "vfmv.v.f   v8,   %[ALPHA]     \n\t" // v8[i] = %[ALPHA]     
                "vle64.v      v0,   (%[C0])      \n\t" // v0[0 : 2] = (%[C0])    // v0[0 : 2] = C[0 : 2, 0]
                "addi       t4,   %[C0], 2*8   \n\t" // t4 = %[C0] + (2*8)     // t4 = (C[2, 0])
                "vle64.v      v1,   (%[C1])      \n\t" // v1[0 : 2] = (%[C1])    // v1[0 : 2] = C[0 : 2, 1]
                "addi       t5,   %[C1], 2*8   \n\t" // t5 = %[C1] + (2*8)     // t5 = (C[2, 1])
                "vle64.v      v2,   (%[C2])      \n\t" // v2[0 : 2] = (%[C2])    // v2[0 : 2] = C[0 : 2, 2]
                "addi       t6,   %[C2], 2*8   \n\t" // t6 = %[C2] + (2*8)     // t6 = (C[2, 2])
                "vle64.v      v3,   (%[C3])      \n\t" // v3[0 : 2] = (%[C3])    // v1[0 : 2] = C[0 : 2, 3]
                "addi       t3,   %[C3], 2*8   \n\t" // t3 = %[C3] + (2*8)     // t3 = (C[2, 3])

                // Multiply Alpha
                "vfmacc.vv  v0,   v8, v16 \n\t"       // v0[i] = (v8[i] * v16[i]) + v0[i]  // C[0 : 2, 0] + (alpha * v16[0 : 2])
                "vle64.v      v4,   (t4)          \n\t" // v4[0 : 2] = (t4)                  // v4[0 : 2] = (C[2 : 4, 0])
                "vfmacc.vv  v1,   v8, v20 \n\t"       // v1[i] = (v8[i] * v20[i]) + v1[i]  // C[0 : 2, 1] + (alpha * v20[0 : 2])
                "vle64.v      v5,   (t5)          \n\t" // v5[0 : 2] = (t5)                  // v5[0 : 2] = (C[2 : 4, 1])
                "vfmacc.vv  v2,   v8, v24 \n\t"       // v2[i] = (v8[i] * v24[i]) + v2[i]  // C[0 : 2, 2] + (alpha * v24[0 : 2])
                "vle64.v      v6,   (t6)          \n\t" // v6[0 : 2] = (t6)                  // v6[0 : 2] = (C[2 : 4, 2])
                "vfmacc.vv  v3,   v8, v28 \n\t"       // v3[i] = (v8[i] * v28[i]) + v3[i]  // C[0 : 2, 3] + (alpha * v28[0 : 2])
                "vle64.v      v7,   (t3)          \n\t" // v7[0 : 2] = (t3))                 // v7[0 : 2] = (C[2 : 4, 3])
                "vfmacc.vv  v4,   v8, v17 \n\t"       // v4[i] = (v8[i] * v17[i]) + v4[i]  // C[2 : 4, 0] + (alpha * v17[0 : 2])
                "vse64.v      v0,   (%[C0])      \n\t"  // store updated C[0 : 2, 0] at (%[C0])
                "add        %[C0], %[C0], 4*8  \n\t"  // %[C0] += (4*8)                    // (C[4, 0])
                "vfmacc.vv  v5,   v8, v21 \n\t"       // v5[i] = (v8[i] * v21[i]) + v5[i]  // C[2 : 4, 1] + (alpha * v17[0 : 2])
                "vse64.v      v1,   (%[C1])      \n\t"  // store updated C[0 : 2, 1] at (%[C1])          
                "add        %[C1], %[C1], 4*8  \n\t"  // %[C1] += (4*8)                    // (C[4, 1])

                "vfmacc.vv  v6,   v8, v25 \n\t"       // v6[i] = (v8[i] * v25[i]) + v6[i]  // C[2 : 4, 2] + (alpha * v17[0 : 2])
                "vse64.v      v2,   (%[C2])      \n\t"  // store updated C[0 : 2, 1] at (%[C2])
                "add        %[C2], %[C2], 4*8  \n\t"  // %[C2] += (4*8)                    // (C[4, 2])

                "vfmacc.vv  v7,   v8, v29 \n\t"       // v7[i] = (v8[i] * v29[i]) + v7[i]  // C[2 : 4, 3] + (alpha * v17[0 : 2])
                "vse64.v      v3,   (%[C3])      \n\t"  // store updated C[0 : 2, 1] at (%[C2])
                "add        %[C3], %[C3], 4*8  \n\t"  // %[C3] += (4*8)                    // (C[4, 3])

                "M4x4_END:                     \n\t"

                : [C0] "+r"(C0), [C1] "+r"(C1), [C2] "+r"(C2), [C3] "+r"(C3),
                  [PA] "+r"(MP_A), [PB] "+r"(MP_B)
                : [ALPHA] "f"(alpha), [BK] "r"(k)
                : "cc", "t0", "t4", "t5", "t6", "t3", "t1", "t2", "ft11", "ft0",
                  "ft1", "ft2", "ft3", "ft4", "ft5", "ft6", "ft7", "v0", "v1",
                  "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11",
                  "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20",
                  "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29",
                  "v30", "v31");


}
