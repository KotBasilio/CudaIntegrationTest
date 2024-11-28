// ==================================== 
// LOG FOR KERNELS
// ==================================== 

__managed__ LogSubsystem myLog;

void LogSubsystem::Initialize()
{
   pos = 0;
   buffer[0] = '=';
   buffer[1] = '4';
   buffer[2] = '2';
   buffer[3] = ' ';
   buffer[4] = '\0';
}

__host__ void LogSubsystem::PrintLog() const
{
   printf("Device Log:%s <--- %d chars\n", buffer, pos);
}

__device__ const char* ErrorCodeToString(ErrorCode code) {
   switch (code) {
      case ErrorCode::SUCCESS: return " OK-";
      case ErrorCode::FAILURE: return "FLR-";
      case ErrorCode::OUT_OF_BOUNDS: return "BND-";
      case ErrorCode::INVALID_ARGUMENT: return "ARG-";
      case ErrorCode::UNKNOWN_ERROR: 
      default: return "UER-";
   }
}

__device__ void LogSubsystem::AppendToBuffer(const char* str)
{
   for (; (*str) && (pos < LOG_BUFFER_SIZE); ++str) {
      buffer[atomicAdd(&pos, 1)] = *str;
   }
   if (pos < LOG_BUFFER_SIZE) buffer[pos] = '\0';
}


inline __device__ int devStrlen(const char* s)
{
   if (!s) {
      return 0;
   }
   auto start = s;
   for (; *s; s++) {}
   return s - start;
}

inline __device__ void devItoa(int value, char* str) 
{
   int i = 0;
   int base = 10;

   // Process individual digits
   do {
      int digit = value % base;
      str[i++] = digit + '0';
      value /= base;
   } while (value != 0);

   str[i]   = '-';
   str[i+1] = '\0';  // Null-terminate the string

   // Reverse the string as the digits are in reverse order
   int start = 0;
   int end = i - 1;
   while (start < end) {
      char temp = str[start];
      str[start] = str[end];
      str[end] = temp;
      start++;
      end--;
   }
}

__device__ void LogSubsystem::AddStr(const char* str)
{
   int estEnd = pos + devStrlen(str);
   if (estEnd < LOG_BUFFER_SIZE) {
      AppendToBuffer(str);
   }
}

__device__ void LogSubsystem::Log(ErrorCode code, const char* module, int line)
{
   char lbuf[5];
   AddStr(ErrorCodeToString(code));
   devItoa(line, lbuf);
   AddStr(lbuf);
   auto tail = module + devStrlen(module) - 1;
   while (*tail != '\\') {
      tail--;
   }
   tail++;
   AddStr(tail);
}

// Kernel to demonstrate usage of the LogSubsystem
//__global__ void ExampleKernel(LogSubsystem* log) {
//   int idx = threadIdx.x + blockIdx.x * blockDim.x;
//
//   // Simulate log entries using the LOG macro
//   if (idx % 2 == 0) {
//      LOG(SUCCESS);
//   } else {
//      LOG(INVALID_ARGUMENT);
//   }
//}
//

__global__ void kerCarpTest(void)
{
   int i = threadIdx.x;
   i++;
}

void Carpenter::SmallTest()
{
   printf("...");
   unsigned int size = 100;
   kerCarpTest << <1, size >> > ();
}

