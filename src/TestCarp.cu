#define  _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <malloc.h>
#include <string.h>

#include "TestSuite.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

void CTestSuite::CarpenterSolve()
{
   Carpenter carp;
   carp.SmallTest();
}

extern __global__ void kerCarpTest(void);

__global__ void kerCarpTest(void)
{
   int i = threadIdx.x;
   i++;
}

Carpenter::Carpenter()
{
}

Carpenter::~Carpenter()
{
}

void Carpenter::SmallTest()
{
   printf("Testing Carpenter()");
   bool isAllright = true;

   unsigned int size = 5;
   kerCarpTest << <1, size >> > ();

   printf("\n==============================\n"
            "Carpenter test: %s\n"
            "==============================\n",
            (isAllright ? "SUCCESS" : "FAIL"));
}
