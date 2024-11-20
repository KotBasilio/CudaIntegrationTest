#include <stdio.h>
#include <conio.h>

extern bool CudaWork(int *c, const int *a, const int *b, unsigned int size);
extern bool CudaClose();
void TestHeap(void);

extern void DoSelfTests();
#define PLATFORM_GETCH _getch

void WaitKey(bool yes = true)
{
   if (yes) {
      printf("{Any key}\n");
      PLATFORM_GETCH();
   }
}

//#pragma warning("Here can be whatever main")

void TestCudaWays()
{
   const int arraySize = 20;
   const int a[arraySize] = { 1,   2,   3,   4,   5 };
   const int b[arraySize] = { 100, 200, 300, 400, 500 };
   int c[arraySize] = { 0 };
   const int check[arraySize] = { 
      111,223,334,446,557,
      60, 70, 80, 90, 100,
      110, 120, 130, 140, 150,
      160, 170, 180, 190, 200
   };

   if (!CudaWork(c, a, b, arraySize)) {
      return;
   }

   printf("======================= CHECKING DATA\n");
   printf("{100,200,300,400,500} +\n");
   printf("{  1,  2,  3,  4,  5} +\n");
   printf("{  1,  2,  3,  4,  5} / 2 +\n");
   printf("{ 10, 20, 30, 40, 50} const =\n");
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
   DoSelfTests();

   TestCudaWays();
   CudaClose();
   TestHeap();
   WaitKey();
}

