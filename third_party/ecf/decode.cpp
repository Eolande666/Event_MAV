#include "ecf_codec.h"
extern "C" size_t decode(const unsigned char *p,size_t n,void *out){try { ECF::Decoder d; return d(p,p+n,(ECF::EventCD*)out); }catch(...){return 0;}}
