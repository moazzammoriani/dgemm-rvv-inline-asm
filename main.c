#include <stdio.h>
#include <stdlib.h>

#define alpha( i,j ) A[ (j)*ldA + (i) ]   // map alpha( i,j ) to array A
#define beta( i,j )  B[ (j)*ldB + (i) ]   // map beta( i,j ) to array B
#define gamma( i,j ) C[ (j)*ldC + (i) ]   // map gamma( i,j ) to array C

#define min( x, y ) ( ( x ) < ( y ) ? x : y )

#define MR 4
#define NR 4

#define MC 264
#define KC 48
#define NC 2016

void Gemm_MRxNRKernel_Packed( int, double *, double *, double *, int );
void PackBlockA_MCxKC( int, int, double *, int, double * );
void PackPanelB_KCxNC( int, int, double *, int, double * );

void LoopOne( int m, int n, int k, double *Atilde, double *MicroPanelB, double *C, int ldC )
{
  for ( int i=0; i<m; i+=MR ) {
    int ib = min( MR, m-i );
    Gemm_MRxNRKernel_Packed( k, &Atilde[ i*k ], MicroPanelB, &gamma( i,0 ), ldC );
  }
}


void LoopTwo( int m, int n, int k, double *Atilde, double *Btilde, double *C, int ldC )
{
  for ( int j=0; j<n; j+=NR ) {
    int jb = min( NR, n-j );
    LoopOne( m, jb, k, Atilde, &Btilde[ j*k ], &gamma( 0,j ), ldC );
  }
}

void LoopThree( int m, int n, int k, double *A, int ldA, double *Btilde, double *C, int ldC )
{
  double *Atilde = ( double * ) malloc( MC * KC * sizeof( double ) );
       
  for ( int i=0; i<m; i+=MC ) {
    int ib = min( MC, m-i );    /* Last loop may not involve a full block */
    PackBlockA_MCxKC( ib, k, &alpha( i, 0 ), ldA, Atilde );
    LoopTwo( ib, n, k, Atilde, Btilde, &gamma( i,0 ), ldC );
  }

  free( Atilde);
}

void LoopFour( int m, int n, int k, double *A, int ldA, double *B, int ldB,
	       double *C, int ldC )
{
  double *Btilde = ( double * ) malloc( KC * NC * sizeof( double ) );
  
  for ( int p=0; p<k; p+=KC ) {
    int pb = min( KC, k-p );    /* Last loop may not involve a full block */
    PackPanelB_KCxNC( pb, n, &beta( p, 0 ), ldB, Btilde );
    LoopThree( m, n, pb, &alpha( 0, p ), ldA, Btilde, C, ldC );
  }

  free( Btilde); 
}

void LoopFive( int m, int n, int k, double *A, int ldA,
		   double *B, int ldB, double *C, int ldC )
{
  for ( int j=0; j<n; j+=NC ) {
    int jb = min( NC, n-j );    /* Last loop may not involve a full block */
    LoopFour( m, jb, k, A, ldA, &beta( 0,j ), ldB, &gamma( 0,j ), ldC );
  } 
}












  
void MyGemm( int m, int n, int k, double *A, int ldA,
	     double *B, int ldB, double *C, int ldC )
{
  if ( m % MR != 0 || MC % MR != 0 ){
    printf( "m and MC must be multiples of MR\n" );
    exit( 0 );
  }
  if ( n % NR != 0 || NC % NR != 0 ){
    printf( "n and NC must be multiples of NR\n" );
    exit( 0 );
  }

  LoopFive( m, n, k, A, ldA, B, ldB, C, ldC );
}



void printMat(double *M, int m, int n, int ldM) {
    int i, j;

    for (i = 0; i < m; ++i) {
        for (j = 0; j < n; ++j) {
            printf("%f ", M[((j * ldM) + i)]);
        }
        printf("\n");

    }
}

int main() {
    double A[16] = {1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  
                    9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0};/*, 
                    17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 
                    25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0};*/

    double B[16] = {1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  
                    9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0};/*, 
                    17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 
                    25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0};*/

    double C[16] = {1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};/*,
                    1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0,                   
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};*/


    int ldA = 4;
    int ldB = 4;
    int ldC = 4;


    printf("A =\n");
    printMat(A, 4, 4, 4);
    
    printf("b =\n");
    printMat(B, 4, 4, 4);

    MyGemm(4, 4, 4, A, 4, B, 4, C, 4);
    printf("C =\n");

    printMat(C, 4, 4, 4);
    return 0;
}
