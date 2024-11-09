#define  _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <malloc.h>
#include <string.h>

#include "../dds-develop/include/dll.h"
#include "../dds-develop/examples/hands.h"

void WaitKey();

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

void STDCALL ErrorMessage(int code, char line[80])
{
   switch (code) {
      case RETURN_NO_FAULT:
         strcpy(line, TEXT_NO_FAULT);
         break;
      case RETURN_UNKNOWN_FAULT:
         strcpy(line, TEXT_UNKNOWN_FAULT);
         break;
      case RETURN_ZERO_CARDS:
         strcpy(line, TEXT_ZERO_CARDS);
         break;
      case RETURN_TARGET_TOO_HIGH:
         strcpy(line, TEXT_TARGET_TOO_HIGH);
         break;
      case RETURN_DUPLICATE_CARDS:
         strcpy(line, TEXT_DUPLICATE_CARDS);
         break;
      case RETURN_TARGET_WRONG_LO:
         strcpy(line, TEXT_TARGET_WRONG_LO);
         break;
      case RETURN_TARGET_WRONG_HI:
         strcpy(line, TEXT_TARGET_WRONG_HI);
         break;
      case RETURN_SOLNS_WRONG_LO:
         strcpy(line, TEXT_SOLNS_WRONG_LO);
         break;
      case RETURN_SOLNS_WRONG_HI:
         strcpy(line, TEXT_SOLNS_WRONG_HI);
         break;
      case RETURN_TOO_MANY_CARDS:
         strcpy(line, TEXT_TOO_MANY_CARDS);
         break;
      case RETURN_SUIT_OR_RANK:
         strcpy(line, TEXT_SUIT_OR_RANK);
         break;
      case RETURN_PLAYED_CARD:
         strcpy(line, TEXT_PLAYED_CARD);
         break;
      case RETURN_CARD_COUNT:
         strcpy(line, TEXT_CARD_COUNT);
         break;
      case RETURN_THREAD_INDEX:
         strcpy(line, TEXT_THREAD_INDEX);
         break;
      case RETURN_MODE_WRONG_LO:
         strcpy(line, TEXT_MODE_WRONG_LO);
         break;
      case RETURN_MODE_WRONG_HI:
         strcpy(line, TEXT_MODE_WRONG_HI);
         break;
      case RETURN_TRUMP_WRONG:
         strcpy(line, TEXT_TRUMP_WRONG);
         break;
      case RETURN_FIRST_WRONG:
         strcpy(line, TEXT_FIRST_WRONG);
         break;
      case RETURN_PLAY_FAULT:
         strcpy(line, TEXT_PLAY_FAULT);
         break;
      case RETURN_PBN_FAULT:
         strcpy(line, TEXT_PBN_FAULT);
         break;
      case RETURN_TOO_MANY_BOARDS:
         strcpy(line, TEXT_TOO_MANY_BOARDS);
         break;
      case RETURN_THREAD_CREATE:
         strcpy(line, TEXT_THREAD_CREATE);
         break;
      case RETURN_THREAD_WAIT:
         strcpy(line, TEXT_THREAD_WAIT);
         break;
      case RETURN_THREAD_MISSING:
         strcpy(line, TEXT_THREAD_MISSING);
         break;
      case RETURN_NO_SUIT:
         strcpy(line, TEXT_NO_SUIT);
         break;
      case RETURN_TOO_MANY_TABLES:
         strcpy(line, TEXT_TOO_MANY_TABLES);
         break;
      case RETURN_CHUNK_SIZE:
         strcpy(line, TEXT_CHUNK_SIZE);
         break;
      default:
         strcpy(line, "Not a DDS error code");
         break;
   }
}

extern unsigned short int dbitMapRank[16];
extern unsigned char dcardSuit[5];
extern unsigned char dcardHand[4];
extern unsigned char dcardRank[16];

void PrintFut(char title[], futureTricks * fut)
{
   printf("%s\n", title);

   printf("%6s %-6s %-6s %-6s %-6s\n",
      "card", "suit", "rank", "equals", "score");

   for (int i = 0; i < fut->cards; i++)
   {
      char res[15] = "";
      equals_to_string(fut->equals[i], res);
      printf("%6d %-6c %-6c %-6s %-6d\n",
         i,
         dcardSuit[ fut->suit[i] ],
         dcardRank[ fut->rank[i] ],
         res,
         fut->score[i]);
   }
   printf("\n");
}

void sample_main_SolveBoard()
{
   printf("Testing SolveBoard()\n");
   bool isAllright = true;

   //SetMaxThreads(0);
   //InitConstants();

   deal dl;
   futureTricks fut2, // solutions == 2
                fut3; // solutions == 3

   int target;
   int solutions;
   int mode;
   int threadIndex = 0;
   int res = RETURN_NO_FAULT;
   char line[80];
   bool match2;
   bool match3;

   for (int handno = 0; handno < 3; handno++) {
      dl.trump = trump[handno];
      dl.first = first[handno];

      dl.currentTrickSuit[0] = 0;
      dl.currentTrickSuit[1] = 0;
      dl.currentTrickSuit[2] = 0;

      dl.currentTrickRank[0] = 0;
      dl.currentTrickRank[1] = 0;
      dl.currentTrickRank[2] = 0;

      for (int h = 0; h < DDS_HANDS; h++)
         for (int s = 0; s < DDS_SUITS; s++)
            dl.remainCards[h][s] = holdings[handno][s][h];

      target = -1;
      solutions = 3;
      mode = 0;
      //   res = SolveBoard(dl, target, solutions, mode, &fut3, threadIndex);

      if (res != RETURN_NO_FAULT) {
         ErrorMessage(res, line);
         printf("DDS error: %s\n", line);
      }

      // auto-control vs expected results
      match3 = CompareFut(&fut3, handno, solutions);

      solutions = 2;
      //   res = SolveBoard(dl, target, solutions, mode, &fut2, threadIndex);
      if (res != RETURN_NO_FAULT) {
         ErrorMessage(res, line);
         printf("DDS error: %s\n", line);
      }

      // auto-control vs expected results
      match2 = CompareFut(&fut2, handno, solutions);

      sprintf(line,
         "SolveBoard, hand %d: solutions 3 %s, solutions 2 %s\n",
         handno + 1,
         (match3 ? "OK" : "ERROR"),
         (match2 ? "OK" : "ERROR"));
      //   qaPrintHand(line, dl);
      isAllright = isAllright && match2 && match3;

      sprintf(line, "solutions == 3 leads %s, trumps: %s\n",  haPlayerToStr(dl.first), haTrumpToStr(dl.trump) );
      PrintFut(line, &fut3);
      sprintf(line, "solutions == 2 leads %s, trumps: %s\n",  haPlayerToStr(dl.first), haTrumpToStr(dl.trump) );
      PrintFut(line, &fut2);
      WaitKey();
   }

   printf("\n=======================================\nThe testing ended with: %s\n",
      (isAllright ? "SUCCESS" : "FAIL"));
}

void DoSelfTests()
{
   //sample_main_PlayBin();
   sample_main_SolveBoard();
   //sample_main_SolveBoard_S1();
   //sample_main_JK_Solve();
   TestHeap();
}

