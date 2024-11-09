#include <stdio.h>
#include <malloc.h>

void IncInCPP(int* dest)
{
   (*dest)++;
}

void TestHeap(void)
{
   // Check heap status, source:
   // https://learn.microsoft.com/en-us/cpp/c-runtime-library/reference/heapwalk?view=msvc-170
   _HEAPINFO hinfo;
   int heapstatus;
   int numLoops;
   hinfo._pentry = NULL;
   numLoops = 0;
   while ((heapstatus = _heapwalk(&hinfo)) == _HEAPOK)
   {
      // printf("%8s block at %Fp of size %4.4X\n",
      //    (hinfo._useflag == _USEDENTRY ? "USED" : "FREE"),
      //    hinfo._pentry, hinfo._size);
      numLoops++;
   }

   switch (heapstatus)
   {
   case _HEAPEMPTY: //printf("OK - empty heap\n");
      break;
   case _HEAPEND:// 
      printf("Heapcheck OK\n");
      break;
   case _HEAPBADPTR:
      printf("ERROR - bad pointer to heap\n");
      break;
   case _HEAPBADBEGIN:
      printf("ERROR - bad start of heap\n");
      break;
   case _HEAPBADNODE:
      printf("ERROR - bad node in heap\n");
      break;
   }
}

void DoSelfTests()
{
   //sample_main_PlayBin();
   //sample_main_SolveBoard();
   //sample_main_SolveBoard_S1();
   //sample_main_JK_Solve();
   TestHeap();
}

