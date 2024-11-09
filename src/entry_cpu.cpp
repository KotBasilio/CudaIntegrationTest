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

void main()
{
   const int arraySize = 20;
   const int a[arraySize] = { 1, 2, 3, 4, 5 };
   const int b[arraySize] = { 10, 20, 30, 40, 50 };
   int c[arraySize] = { 0 };

   if (CudaWork(c, a, b, arraySize)) {
      printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
         c[0], c[1], c[2], c[3], c[4]);
   }

   CudaClose();
   WaitKey();

   DoSelfTests();
   WaitKey();

}

