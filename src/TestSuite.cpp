#define  _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <malloc.h>
#include <string.h>

#include "TestSuite.h"

static bool _verboseDDS = false;

#define DDS_FULL_LINE 76
#define DDS_HAND_OFFSET 12
#define DDS_HAND_LINES 12

extern unsigned short int dbitMapRank[16];
extern unsigned char dcardSuit[5];
extern unsigned char dcardHand[4];
extern unsigned char dcardRank[16];

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

   switch (heapstatus) {
      case _HEAPEMPTY: //printf("OK - empty heap\n");
         break;
      case _HEAPEND: 
         //printf("Heapcheck OK\n");
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

// ret: HCP on NS line
void CTestSuite::PrintFut(char title[], const futureTricks* fut)
{
   VERBOSE("%s\n", title);

   VERBOSE("%6s %-6s %-6s %-6s %-6s\n",
      "card", "suit", "rank", "equals", "score");

   for (int i = 0; i < fut->cards; i++) {
      char res[15] = "";
      equals_to_string(fut->equals[i], res);
      VERBOSE("%6d %-6c %-6c %-6s %-6d\n",
         i,
         dcardSuit[fut->suit[i]],
         dcardRank[fut->rank[i]],
         res,
         fut->score[i]);
   }
   VERBOSE("\n");
}

uint CTestSuite::CalcNSLineHCP(const deal& dl, uint& ctrl)
{
   // requires Walrus types
   ctrl = 9;
   return 25;

   //const auto &cards = dl.remainCards;
   //u64 facecards (RA | RK | RQ | RJ);
   //SplitBits reducedHand (
   //   (((cards[SOUTH][SOL_SPADES  ] | cards[NORTH][SOL_SPADES  ]) & facecards) << (1 + 16*3)) |
   //   (((cards[SOUTH][SOL_HEARTS  ] | cards[NORTH][SOL_HEARTS  ]) & facecards) << (1 + 16*2)) |
   //   (((cards[SOUTH][SOL_DIAMONDS] | cards[NORTH][SOL_DIAMONDS]) & facecards) << (1 + 16*1)) |
   //   (((cards[SOUTH][SOL_CLUBS   ] | cards[NORTH][SOL_CLUBS   ]) & facecards) << (1))
   //);

   //twlControls controls(reducedHand);
   //ctrl = controls.total;

   //twlHCP hcp(reducedHand);
   //return hcp.total;
}

void CTestSuite::qaPrintHand(char title[], const deal& dl, char tail[])
{
   int c, h, s, r;
   char text[DDS_HAND_LINES][DDS_FULL_LINE];

   // clear virtual screen
   for (int l = 0; l < DDS_HAND_LINES; l++) {
      memset(text[l], ' ', DDS_FULL_LINE);
      text[l][DDS_FULL_LINE - 1] = '\0';
   }

   // for each hand
   for (h = 0; h < DDS_HANDS; h++) {
      // detect location
      int offset, line;
      if (h == 0) {
         offset = DDS_HAND_OFFSET;
         line = 0;
      } else if (h == 1) {
         offset = 2 * DDS_HAND_OFFSET;
         line = 4;
      } else if (h == 2) {
         offset = DDS_HAND_OFFSET;
         line = 8;
      } else {
         offset = 0;
         line = 4;
      }

      // print hand to v-screen
      for (s = 0; s < DDS_SUITS; s++) {
         c = offset;
         for (r = 14; r >= 2; r--) {
            if ((dl.remainCards[h][s] >> 2) & dbitMapRank[r])
               text[line + s][c++] = static_cast<char>(dcardRank[r]);
         }

         if (c == offset)
            text[line + s][c++] = '-';

         if (h == SOUTH || h == EAST)
            text[line + s][c] = '\0';
      }
   }

   // print HCP and controls
   uint ctrl;
   sprintf(text[DDS_STATS_LINE] + DDS_STATS_OFFSET, "HCP : %d", CalcNSLineHCP(dl, ctrl));
   sprintf(text[DDS_STATS_LINE + 1] + DDS_STATS_OFFSET, "CTRL: %d", ctrl);

   // start with title and underline it
   VERBOSE(title);
   char dashes[80];
   int l = static_cast<int>(strlen(title)) - 1;
   for (int i = 0; i < l; i++)
      dashes[i] = '-';
   dashes[l] = '\0';
   VERBOSE("%s\n", dashes);

   // print the v-screen
   for (int i = 0; i < DDS_HAND_LINES; i++)
      VERBOSE("   %s\n", text[i]);
   VERBOSE(tail);
}

void CTestSuite::FillDeal(deal& dl, int handno)
{
   if (handno < 3) {
      dl.trump = trump[handno];
      dl.first = first[handno];
   } else {
      dl.trump = 3;
      dl.first = 3;
   }

   dl.currentTrickSuit[0] = 0;
   dl.currentTrickSuit[1] = 0;
   dl.currentTrickSuit[2] = 0;

   dl.currentTrickRank[0] = 0;
   dl.currentTrickRank[1] = 0;
   dl.currentTrickRank[2] = 0;

   for (int h = 0; h < DDS_HANDS; h++)
      for (int s = 0; s < DDS_SUITS; s++)
         dl.remainCards[h][s] = holdings[handno][s][h];
}

void CTestSuite::NoticeErrorDDS(int res, bool& isAllright)
{
   if (res != RETURN_NO_FAULT) {
      char line[80];
      ErrorMessage(res, line);
      printf("DDS error: %s\n", line);
      isAllright = false;
}
   }

CTestSuite::CTestSuite()
{
   #ifdef VERBOSE_DDS_TEST
      _verboseDDS = true;
   #endif

   memset(mFut1, 0, sizeof(mFut1));
   memset(mFut2, 0, sizeof(mFut2));
   memset(mFut3, 0, sizeof(mFut3));
}

void CTestSuite::VerboseLogOn23(deal& dl, const futureTricks* fut3, const futureTricks* fut2, bool match3, bool match2)
{
   char line[80];
   char tail[60];

   sprintf(line, "solutions == 3 leads %s, trumps: %s\n", haPlayerToStr(dl.first), haTrumpToStr(dl.trump));
   PrintFut(line, fut3);
   sprintf(line, "solutions == 2 leads %s, trumps: %s\n", haPlayerToStr(dl.first), haTrumpToStr(dl.trump));
   PrintFut(line, fut2);
   sprintf(tail,
      "Checking: sol=3 %s, sol=2 %s\n",
      (match3 ? "OK" : "ERROR"),
      (match2 ? "OK" : "ERROR"));
   sprintf(line, "The board:\n");
   qaPrintHand(line, dl, tail);
}

void CTestSuite::SolveLinear()
{
   printf("Linear test for SolveBoard()");
   SetResources();
   bool isAllright = true;

   deal dl;
   futureTricks fut1; // solutions == 1
   futureTricks fut2; // solutions == 2
   futureTricks fut3; // solutions == 3 TEST_HOLDINGS_COUNT

   int target = -1;
   int solutions;
   int mode = 0;
   int res = RETURN_NO_FAULT;
   bool match1;
   bool match2;
   bool match3;
   int threadBegin = MAX_THREADS_IN_TEST - 1;

   for (int threadIndex = threadBegin; threadIndex >= 0; threadIndex--) {
      int handno = 0;
      for (; handno < 3; handno++) {
         FillDeal(dl, handno);

         // solve with auto-control vs expected results
         solutions = 3;
         res = SolveBoard(dl, target, solutions, mode, &fut3, threadIndex);
         NoticeErrorDDS(res, isAllright);
         match3 = CompareFut(&fut3, handno, solutions);

         // solve with auto-control vs expected results
         solutions = 2;
         res = SolveBoard(dl, target, solutions, mode, &fut2, threadIndex);
         NoticeErrorDDS(res, isAllright);
         match2 = CompareFut(&fut2, handno, solutions);

         // out
         VERBOSE("--------------\nSolveBoard, thrid=%d hand %d:\n", threadIndex, handno);
         VerboseLogOn23(dl, &fut3, &fut2, match3, match2);
         isAllright = isAllright && match2 && match3;
      }

      solutions = 1;
      for (; handno < TEST_HOLDINGS_COUNT; handno++) {
         FillDeal(dl, handno);
         res = SolveBoard(dl, target, solutions, mode, &fut1, threadIndex);
         NoticeErrorDDS(res, isAllright);
         match1 = CompareFut(&fut1, handno, solutions);
         isAllright = isAllright && match1;
         dl.trump = 0;
         dl.first = 0;
         res = SolveBoard(dl, target, solutions, mode, &fut1, threadIndex);
         // can use SolveSameBoard(threadIndex, dl, &fut1, fut1.score[0]); 
         // but only for the same suit. 
         // So, it's handy only for par calculaton
         NoticeErrorDDS(res, isAllright);
         match1 = CompareFut(&fut1, handno + TEST_SOLVE_SAME, solutions);
         isAllright = isAllright && match1;
      }
      WaitKey(_verboseDDS);

      printf(".");
   }

   printf("\n===============================\n"
            "One-threaded DDS solve test: %s\n"
            "===============================\n",
            (isAllright ? "SUCCESS" : "FAIL"));
}

// separate solving and testing
void CTestSuite::SeparatedSolve()
{
   SetResources();
   printf("Testing Separated()");
   bool isAllright = true;

   deal dl;

   int target = -1;
   int mode = 0;
   int res = RETURN_NO_FAULT;
   int threadBegin = MAX_THREADS_IN_TEST - 1;

   // solve & store
   for (int threadIndex = threadBegin; threadIndex >= 0; threadIndex--) {
      int i = 0;
      for (; i < 3; i++) {
         int handno = (i + threadIndex) % 3; // mix up the order of solving
         FillDeal(dl, handno);
         auto out3 = &mFut3[threadIndex][handno];
         auto out2 = &mFut2[threadIndex][handno];

         int solutions = 3;
         res = SolveBoard(dl, target, solutions, mode, out3, threadIndex);
         NoticeErrorDDS(res, isAllright);

         solutions = 2;
         res = SolveBoard(dl, target, solutions, mode, out2, threadIndex);
         NoticeErrorDDS(res, isAllright);
      }
      printf(".");

      int solutions = 1;
      for (int handno = i; handno < TEST_HOLDINGS_COUNT; handno++) {
         FillDeal(dl, handno);
         auto out1 = &mFut1[threadIndex][handno - i];
         res = SolveBoard(dl, target, solutions, mode, out1, threadIndex);
         NoticeErrorDDS(res, isAllright);
         dl.trump = 0;
         dl.first = 0;
         auto out4 = &mFut1[threadIndex][handno - i + TEST_SOLVE_SAME];
         res = SolveBoard(dl, target, solutions, mode, out4, threadIndex);
         NoticeErrorDDS(res, isAllright);
      }
   }

   // control
   bool match2;
   bool match3;
   for (int threadIndex = threadBegin; threadIndex >= 0; threadIndex--) {
      int handno = 0;
      for (; handno < 3; handno++) {
         auto cmp3 = &mFut3[threadIndex][handno];
         auto cmp2 = &mFut2[threadIndex][handno];
         match3 = CompareFut(cmp3, handno, 3);
         match2 = CompareFut(cmp2, handno, 2);

         VERBOSE("--------------\nSeparated Solve, thrid=%d hand %d:\n", threadIndex, handno);
         FillDeal(dl, handno);
         VerboseLogOn23(dl, cmp3, cmp2, match3, match2);
         isAllright = isAllright && match2 && match3;
         WaitKey(_verboseDDS);
      }
      int solutions = 1;
      for (int idx = 0; handno < TEST_HOLDINGS_COUNT; handno++, idx++) {
         FillDeal(dl, handno);
         auto cmp1A = &mFut1[threadIndex][idx];
         auto cmp1B = &mFut1[threadIndex][idx + TEST_SOLVE_SAME];
         isAllright = isAllright && CompareFut(cmp1A, handno, solutions);
         isAllright = isAllright && CompareFut(cmp1B, handno + TEST_SOLVE_SAME, solutions);
      }
   }

   printf("\n==============================\n"
            "Separated DDS solve test: %s\n"
            "==============================\n",
            (isAllright ? "SUCCESS" : "FAIL"));
}

void DoSelfTests()
{
   auto tst = new CTestSuite;

   tst->SolveLinear();
   tst->SeparatedSolve();
   tst->CarpenterSolve();
   //tst_JK_Solve();

   delete tst;
   TestHeap();
   WaitKey(_verboseDDS);
}

