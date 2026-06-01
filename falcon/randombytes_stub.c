/* Link-only `randombytes` for verify builds.  Signing/keygen must not
 * reach this symbol. */

#include <stdlib.h>

int
randombytes( unsigned char * x, unsigned long long xlen ) {
  (void)x; (void)xlen;
  abort();
}
