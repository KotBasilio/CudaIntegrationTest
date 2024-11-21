#define  _CRT_SECURE_NO_WARNINGS

#include "../dds-develop/include/dll.h"
#include "../dds-develop/examples/hands.h"

void WaitKey(bool yes);

//#define VERBOSE_DDS_TEST
#ifdef VERBOSE_DDS_TEST
   #define VERBOSE  printf
#else
   #define VERBOSE(...)
#endif
const int MAX_THREADS_IN_TEST = 12;

class CTestSuite
{
   futureTricks mFut1[MAX_THREADS_IN_TEST][TEST_SOLVE_SAME*2]; // solutions == 1
   futureTricks mFut2[MAX_THREADS_IN_TEST][3]; // solutions == 2
   futureTricks mFut3[MAX_THREADS_IN_TEST][3]; // solutions == 3

   void PrintFut(char title[], const futureTricks* fut);
   uint CalcNSLineHCP(const deal& dl, uint& ctrl);
   void qaPrintHand(char title[], const deal& dl, char tail[]);
   void FillDeal(deal& dl, int handno);
   void NoticeErrorDDS(int res, bool& isAllright);
   void VerboseLogOn23(deal& dl, const futureTricks* fut3, const futureTricks* fut2, bool match3, bool match2);

   public:
   CTestSuite();
      
   void SolveLinear();
   void SeparatedSolve();
   void CarpenterSolve();
};
