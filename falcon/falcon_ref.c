/* Namespaced wrapper around the vendored Falcon round 3 NIST verifier. */

#include "falcon.h"

/* Vendored NIST entry point; uses `unsigned long long` lengths. */
int crypto_sign_open( unsigned char *m, unsigned long long *mlen,
                      const unsigned char *sm, unsigned long long smlen,
                      const unsigned char *pk );

int
falcon_ref_crypto_sign_open( uint8_t       * m,  size_t * mlen,
                             uint8_t const * sm, size_t   smlen,
                             uint8_t const * pk ) {
  unsigned long long ull = 0;
  int r = crypto_sign_open( m, &ull, sm, (unsigned long long)smlen, pk );
  if( mlen ) *mlen = (size_t)ull;
  return r;
}
