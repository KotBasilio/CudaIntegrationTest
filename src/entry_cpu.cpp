#include <stdio.h>
#include <conio.h>

extern bool CudaWork(int *c, const int *a, const int *b, unsigned int size);
extern bool CudaClose();
extern void DoSelfTests();
#define PLATFORM_GETCH _getch

void WaitKey()
{
   printf("{Any key}\n");
   PLATFORM_GETCH();
}

void TestCudaWays()
{
   const int arraySize = 20;
   const int a[arraySize] = { 1,   2,   3,   4,   5 };
   const int b[arraySize] = { 100, 200, 300, 400, 500 };
   int c[arraySize] = { 0 };
   const int check[arraySize] = { 
      111,213,314,416,517,
      10, 10, 10, 10, 10,
      10, 10, 10, 10, 10,
      10, 10, 10, 10, 10
   };

   if (!CudaWork(c, a, b, arraySize)) {
      return;
   }

   printf("=======================\n");
   printf("{100,200,300,400,500} +\n");
   printf("{  1,  2,  3,  4,  5} +\n");
   printf("{  1,  2,  3,  4,  5} / 2 +\n");
   printf("{ 10, 10, 10, 10, 10} =\n");
   printf("{%d,%d,%d,%d,%d}\n", c[0], c[1], c[2], c[3], c[4]);
   printf("======================= ");

   bool isAllright = true;
   for (auto i = 0; i < arraySize; i++) {
      isAllright = isAllright && (c[i] == check[i]);
   }
   printf("%s\n", (isAllright ? "SUCCESS" : "FAIL"));
}

void main()
{
   TestCudaWays();
   CudaClose();
   WaitKey();

   DoSelfTests();
   WaitKey();
}

