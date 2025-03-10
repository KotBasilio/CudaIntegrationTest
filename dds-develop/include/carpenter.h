// -----------------------------------------
// GPU acceleration part
// requires CUDA technology
// its toolkit is available here:
// https://developer.nvidia.com/cuda-toolkit

// host class with very loose coupling
class Carpenter {
public:
   Carpenter();
   ~Carpenter();

   void Overlook(const futureTricks *h_Futures, int maxFut);
   void SolveChunk(boards& chunk);
   void SyncDown(futureTricks *h_Futures, int maxFut);

private:
   class CarpImpl* Himpl = nullptr;
};
