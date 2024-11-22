#define  _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <malloc.h>
#include <string.h>

#include "TestSuite.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

extern __global__ void kerCarpTest(void);

void CTestSuite::CarpenterSolve()
{
   unsigned int size = 5;
   kerCarpTest<<<1, size>>>();
}

__global__ void kerCarpTest(void)
{
   int i = threadIdx.x;
   i++;
}
