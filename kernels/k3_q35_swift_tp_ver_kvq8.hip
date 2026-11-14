// Q8 TP verify: the F16 TP/verify ops with the unpacked Q8 attention reader.
// Keep the packed F16 reader and the original persistent object unchanged.
#define K3_Q35_KV_Q8
#define K3_TP
#define K3_XCHG_PUSH_ONLY
#define K3_Q3M_VER
#define K3_Q35_VER
#include "k3_q35_swift.hip"
