// -----------------------------------------
// GPU acceleration part
// requires CUDA technology
// its toolkit is available here:
// https://developer.nvidia.com/cuda-toolkit

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

class Carpenter {
public:
   // host part
   Carpenter();
   ~Carpenter();

   void SmallTest();
   void SolveChunk(boards& chunk);

   // device part
   __device__ void Solve(deal* myDeal);
   __device__ int  SolveBoard(
      const struct deal &dl,
      const int target,
      const int solutions,
      const int mode,
      struct futureTricks * futp,
      struct ThreadData* thrp);

};
