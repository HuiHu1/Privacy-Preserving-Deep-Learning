#include <stdio.h>
#define N 8
void matrix_transpose(int num_rows, int num_cols, double a[num_rows][num_cols],double transpose[num_cols][num_rows]);
void multiplyMatrices(int rowFirst, int columnFirst, int rowSecond, int columnSecond,double firstMatrix[][columnFirst], 
                                                                                     double secondMatrix[][columnSecond],
                                                                                     double mult[][columnSecond]);
void getCofactor(int p, int q, int n,double A[N][N], double temp[N][N]);
double determinant(int n,double A[N][N]);
void adjoint(double A[N][N],double adj[N][N]);
void inverse(int n,double A[N][N], double inverse[N][N]);
void addMatrices(int row, int column,double firstMatrix[][column], 
                                     double secondMatrix[][column],
                                     double add[][column]);
void multiply_matrice_vector(int row,int column,double matrix[][column], 
                                                double vector[column],
                                                double mult[row]);
void multiply_vector_matrice(int row,int column,double vector [row], 
                                                double matrix[][column], 
                                                double mult[column]);
// Transpose matrice
void matrix_transpose(int num_rows, int num_cols, double a[num_rows][num_cols],
                                                  double transpose[num_cols][num_rows]){
    for(int i=0;i<num_rows;i++){
        for(int j=0;j<num_cols;j++){
            transpose[j][i]=a[i][j];
        }
    }
}
//Multipy two matrices
void multiplyMatrices(int rowFirst, int columnFirst, int rowSecond, int columnSecond,double firstMatrix[][columnFirst], 
                                                                                     double secondMatrix[][columnSecond],
                                                                                     double mult[][columnSecond]){   
    int i, j, k;
	for(i = 0; i < rowFirst; ++i)
	{  for(j = 0; j < columnSecond; ++j)
		{
			mult[i][j] = 0;
		}
	}
	for(i = 0; i < rowFirst; ++i)
	{  for(j = 0; j < columnSecond; ++j)
		{  for(k=0; k<columnFirst; ++k)
			{
				mult[i][j] = mult[i][j]+firstMatrix[i][k] * secondMatrix[k][j];
			}
		}
	}
}
//Matrice addition
void addMatrices(int row, int column, double firstMatrix[][column], 
                                      double secondMatrix[][column],
                                      double add[][column]){ 
for (int i=0; i<row;i++)  
    for (int j=0; j<row;j++)
    {
      add[i][j] = firstMatrix[i][j]+secondMatrix[i][j];
    }
}
//Matrice multiplies vector
void multiply_matrice_vector(int row,int column,double matrix[][column], 
                                                double vector[column],
                                                double mult[row]){  
for(int i = 0; i < row; ++i)
	{
        mult[i] = 0;
     }
	for(int i = 0; i < row; ++i)
	  for(int j = 0; j < column; ++j)
		{   
			mult[i] = mult[i]+matrix[i][j] * vector[j];
		}
}
//Vector multiplies matrice
void multiply_vector_matrice(int row,int column,double vector [row], 
                                                double matrix[][column], 
                                                double mult[column]){  
for(int i = 0; i < column; ++i)
	{
        mult[i] = 0;
     }
	for(int i = 0; i < column; ++i)
	  for(int j = 0; j < row; ++j)
		{   
			mult[i] = mult[i]+vector[j]*matrix[j][i];
		}
}
//Get cofactor of A[p][q] in temp[][]. n is current dimension of A[][].
void getCofactor(int p, int q, int n,double A[N][N], double temp[N][N])
{
    int i = 0, j = 0;
    for (int row = 0; row < n; row++)
    {
        for (int col = 0; col < n; col++)
        {
            if (row != p && col != q)
            {
                temp[i][j++] = A[row][col];
                if (j == n - 1)
                {
                    j = 0;
                    i++;
                }
            }
        }
    }
}
// Get determinant of matrix.n is current dimension of A[][]
double determinant(int n,double A[N][N])
{
    double D = 0.0;  
    if (n == 1)
        return A[0][0];
    double temp[N][N]; 
    int sign = 1;  
    for (int f = 0; f < n; f++)
    {
        getCofactor(0,f,n,A,temp); // Getting Cofactor of A[0][f]
        D += sign * A[0][f] * determinant( n - 1, temp);
        sign = -sign;
    }
    return D;
}
 // Get adjoint of A[N][N].
void adjoint(double A[N][N],double adj[N][N])
{
    if (N == 1)
    {
        adj[0][0] = 1.0;
        return;
    }
    int sign = 1;
    double temp[N][N];
  
    for (int i=0; i<N; i++)
    {
        for (int j=0; j<N; j++)
        {
            getCofactor(i, j, N,A, temp);
            sign = ((i+j)%2==0)? 1: -1;
            adj[j][i] = (sign)*(determinant(N-1,temp));
        }
    }
} 
// Calculate inverse matrice.n is current dimension of A[][]
void inverse(int n,double A[N][N], double inverse[N][N])
{
    double det = determinant(n,A);
    if (det == 0.0)
    { 
        printf("No inverse!");
        return;
    }
    double adj[N][N];
    adjoint(A, adj);
    for (int i=0; i<N; i++)
        for (int j=0; j<N; j++)
            inverse[i][j] = adj[i][j]/det; // Find Inverse using formula "inverse(A) = adj(A)/det(A)"
          
}