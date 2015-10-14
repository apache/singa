#include <iostream>

#include "singa_op.h"
#include "math_addr.h"

using namespace std;

void test_gemm1()
{
            float A[3][2] = {};
            float B[3][2] = {};
            float C[2][2] = {};
            for(int i = 0; i < 3; i++)
                for(int j = 0; j < 2; j++)
                {
                A[i][j] = i+j;
                B[i][j] = i+j - i*j;
                }
            cpu_gemm(A[0], B[0], 2, 2, 3 , 1, 0, true, false, C[0]);
            float D[2][2] = {};
            for(int i = 0; i < 2; i++)
                for(int j = 0; j < 2; j++)
                {
                    D[i][j] = 0;
                    for(int k = 0; k < 3; k++)
                    D[i][j] += A[k][i]*B[k][j];
                }
            for(int i = 0; i < 2; i++)
                for(int j = 0; j < 2; j++)
                {
                cout<<C[i][j] - D[i][j]<<endl;
                }
}


void test_gemm2()
{
            float A[2][3] = {};
            float B[3][2] = {};
            float C[2][2] = {};
            for(int i = 0; i < 3; i++)
                for(int j = 0; j < 2; j++)
                {
                A[j][i] = i-j;
                B[i][j] = i+j + i*j;
                }
            cpu_gemm(A[0], B[0], 2, 2, 3 , 1, 0, false, false, C[0]);
            float D[2][2] = {};
            for(int i = 0; i < 2; i++)
                for(int j = 0; j < 2; j++)
                {
                    D[i][j] = 0;
                    for(int k = 0; k < 3; k++)
                    D[i][j] += A[i][k]*B[k][j];
                }
            for(int i = 0; i < 2; i++)
                for(int j = 0; j < 2; j++)
                {
                cout<<C[i][j] - D[i][j]<<endl;
                }
}


void test_gemv()
{
        float A[4][3] = {};
        float B[4]= {};
        float C[3] = {};
        float D[3] = {};
        for(int i = 0; i < 4; i++)
        {
            for(int j = 0; j < 3; j++)
                    {
                    A[j][i] = i-j + i*j;
                    }
        }
        for(int i = 0; i < 4; i++)B[i] = i;
        for(int i = 0; i < 3; i++)C[i] = 10;
        cpu_gemv(A[0], B, 4, 3, 1, 1, true, C);
        for(int i = 0; i < 3; i++)
                for(int j = 0; j < 4; j++)
                {
                    D[i] += A[j][i]*B[j];
                }
        for(int i = 0; i < 3; i++)cout<<C[i] - D[i] - 10<<endl;
}

void test_axpy()
{
        float A[4][3] = {};
        float C[4][3] = {};
        float B[3][4] = {};
        float D[3][4] = {};
        for(int i = 0; i < 4; i++)
        {
            for(int j = 0; j < 3; j++)
                    {
                    A[i][j] = i-j + i*j;
                    B[j][i] = i-j + i*j;
                    C[i][j] = A[i][j];
                    D[j][i] = B[j][i];
                    }
        }
        cpu_axpy(A[0], 12, 2, B[0]);
        for(int i = 0; i < 12; i++)D[0][i] += 2*C[0][i];
        for(int i = 0; i < 3; i++)
        {
            for(int j = 0; j < 4; j++)
                    {
                    cout<<B[i][j] - D[i][j]<<endl;
                    }
        }
}

void test_eop()
{
        float A[10] = {};
        float B[10] = {};
        float C[10] = {};
        float D[10] = {};
        float O[10] = {};
        for(int i = 0; i < 10; i++)
        {
            A[i] = i;
            B[i] = -i;
            C[i] = i;
        }
        cpu_e_f<op::Set>(5, 15, O);
        for(int i = 0; i < 5; i++)cout<<O[i] - 15<<endl;
        for(int i = 5; i < 10; i++)cout<<O[i]<<endl;
        cpu_e_f<op::Scale>(10, C, 2, C);
        for(int i = 0; i < 10; i++)cout<<C[i] - 2* i<<endl;
        cpu_e_f<op::Add>(10, A, B, 0, 0, O);
        for(int i = 0; i < 10; i++)cout<<O[i]<<endl;
}

void test_exrd()
{
        float A[3][10] = {};
        float B[3] = {};
        for(int i = 0; i < 3; i++)
            for(int j = 0; j < 10; j++)
            {
                A[i][j] = (i + 1)*j;
            }
        cpu_reduce_f<op::Sum>(A[0], 3, 10, B);
        for(int i = 0; i < 3; i++) B[i] -= 45*(i+1);
        for(int i = 0; i < 3; i++)cout<<B[i]<<endl;
        cpu_expand_f<op::Repmat>(B, 3, 10, A[0]);
        cpu_reduce_f<op::Sum>(A[0], 3, 10, B);
        for(int i = 0; i < 3; i++)cout<<B[i]<<endl;
}

int main()
{
    test_gemm1()  ;
	test_gemm2();
	test_gemv();
	test_axpy();
	test_eop();
	test_exrd();
    return 0;
}


