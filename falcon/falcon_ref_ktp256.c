/* Pornin verifier with non-standard KTP256 hash-to-point.
 *
 * KTP256 absorbs nonce||msg with TurboSHAKE256 padding and 12-round
 * Keccak-p[1600], then copies that state into 8 lanes, xors counters
 * into capacity lane 17, permutes all lanes, and rejection-samples the
 * concatenated rate blocks.  The resulting `c` is not Falcon round 3. */

#include "falcon.h"

#include <stdint.h>
#include <stddef.h>
#include <string.h>

#define NONCELEN     40
#define N            FALCON_N
#define LOGN         FALCON_LOGN
#define Q            FALCON_Q
#define K_REJ        ( ( 1 << 16 ) / Q ) /* 5 */
#define SHAKE_RATE   136                 /* (1600 - 2*256) / 8 */
#define COUNTER_LANE 17                  /* capacity lane */

/* Pornin internals. */
extern size_t falcon_inner_modq_decode  ( uint16_t * x, unsigned logn,
                                          void const * in, size_t max_in_len );
extern size_t falcon_inner_comp_decode  ( int16_t  * x, unsigned logn,
                                          void const * in, size_t max_in_len );
extern void   falcon_inner_to_ntt_monty ( uint16_t * h, unsigned logn );
extern int    falcon_inner_verify_raw   ( uint16_t const * c0,
                                          int16_t  const * s2,
                                          uint16_t const * h,
                                          unsigned logn, uint8_t * tmp );

#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512DQ__)
#define HAVE_AVX512 1
#else
#define HAVE_AVX512 0
#endif

#if HAVE_AVX512

#include <immintrin.h>

/* XKCP plain64 Keccak-p[1600], used for the base state. */

typedef struct { uint64_t A[ 25 ]; } kp1600_state_t;

extern void KeccakP1600_plain64_Initialize        ( kp1600_state_t * st );
extern void KeccakP1600_plain64_AddBytes          ( kp1600_state_t * st,
                                                    unsigned char const * data,
                                                    unsigned int offset,
                                                    unsigned int length );
extern void KeccakP1600_plain64_Permute_12rounds  ( kp1600_state_t * st );

/* XKCP times8 state: A[L][instance i] is at byte offset L*64 + i*8. */

typedef struct { __m512i A[ 25 ]; } kp1600_x8_state_t;

extern void KeccakP1600times8_AVX512_InitializeAll      ( kp1600_x8_state_t * st );
extern void KeccakP1600times8_AVX512_AddBytes           ( kp1600_x8_state_t * st,
                                                          unsigned int instanceIndex,
                                                          unsigned char const * data,
                                                          unsigned int offset,
                                                          unsigned int length );
extern void KeccakP1600times8_AVX512_PermuteAll_12rounds( kp1600_x8_state_t * st );
extern void KeccakP1600times8_AVX512_ExtractBytes       ( kp1600_x8_state_t const * st,
                                                          unsigned int instanceIndex,
                                                          unsigned char * data,
                                                          unsigned int offset,
                                                          unsigned int length );

/* KTP256 hash-to-point, shared by all `_ktp256` verifiers. */
void
fa512_hash_to_point_ktp256( uint16_t * out, uint8_t const * in, size_t in_len ) {
  /* Base sponge: TurboSHAKE padding, 12-round Keccak. */
  kp1600_state_t base;
  KeccakP1600_plain64_Initialize( &base );
  while( in_len >= SHAKE_RATE ) {
    KeccakP1600_plain64_AddBytes( &base, in, 0, SHAKE_RATE );
    KeccakP1600_plain64_Permute_12rounds( &base );
    in += SHAKE_RATE; in_len -= SHAKE_RATE;
  }
  if( in_len ) KeccakP1600_plain64_AddBytes( &base, in, 0, (unsigned)in_len );
  unsigned char ds  = 0x1F;
  unsigned char fin = 0x80;
  KeccakP1600_plain64_AddBytes( &base, &ds,  (unsigned)in_len, 1 );
  KeccakP1600_plain64_AddBytes( &base, &fin, SHAKE_RATE - 1,    1 );

  unsigned remaining    = N;
  unsigned counter_base = 0;

  while( remaining ) {
    /* Broadcast S to 8 lanes; xor counters into capacity lane 17. */
    __m512i st[25] __attribute__((aligned(64)));
    for( int L=0; L<25; L++ ) st[L] = _mm512_set1_epi64( (long long)base.A[L] );
    __m512i ctrs = _mm512_setr_epi64(
        (long long)(counter_base + 0), (long long)(counter_base + 1),
        (long long)(counter_base + 2), (long long)(counter_base + 3),
        (long long)(counter_base + 4), (long long)(counter_base + 5),
        (long long)(counter_base + 6), (long long)(counter_base + 7) );
    st[ COUNTER_LANE ] = _mm512_xor_epi64( st[ COUNTER_LANE ], ctrs );

    /* Permute all 8 lanes. */
    KeccakP1600times8_AVX512_PermuteAll_12rounds( (kp1600_x8_state_t *)st );

    /* Extract 17 rate lanes per instance; stop once c has 512 coeffs. */
    uint64_t const * lanes = (uint64_t const *)st;
    for( unsigned i=0; i<8 && remaining > 0; i++ ) {
      uint64_t rate_lanes[ 17 ];
      for( int L=0; L<17; L++ ) rate_lanes[ L ] = lanes[ L*8 + i ];
      uint8_t const * buf = (uint8_t const *)rate_lanes;
      for( size_t j=0; j+1 < SHAKE_RATE && remaining > 0; j += 2 ) {
        uint32_t w = ( (uint32_t)buf[ j ] << 8 ) | (uint32_t)buf[ j+1 ];
        if( w < (uint32_t)( K_REJ * Q ) ) { /* 5*Q = 61445 */
          while( w >= Q ) w -= Q;
          *out++ = (uint16_t)w;
          remaining--;
        }
      }
    }
    counter_base += 8;
  }
}

int
falcon_ref_ktp256_crypto_sign_open( uint8_t       * m,  size_t * mlen,
                                    uint8_t const * sm, size_t   smlen,
                                    uint8_t const * pk ) {
  uint8_t  tmp[ 2 * 512 ];
  uint16_t h[ 512 ], c0[ 512 ];
  int16_t  sig[ 512 ];

  if( pk[ 0 ] != 0x00 + LOGN ) return -1;
  if( falcon_inner_modq_decode( h, LOGN, pk + 1,
                                FALCON_PUBKEY_SIZE - 1 )
      != FALCON_PUBKEY_SIZE - 1 ) return -1;
  falcon_inner_to_ntt_monty( h, LOGN );

  if( smlen < 2 + NONCELEN ) return -1;
  size_t sig_len = ( (size_t)sm[ 0 ] << 8 ) | (size_t)sm[ 1 ];
  if( sig_len > smlen - 2 - NONCELEN ) return -1;
  size_t msg_len = smlen - 2 - NONCELEN - sig_len;

  uint8_t const * esig = sm + 2 + NONCELEN + msg_len;
  if( sig_len < 1 || esig[ 0 ] != 0x20 + LOGN ) return -1;
  if( falcon_inner_comp_decode( sig, LOGN, esig + 1,
                                sig_len - 1 ) != sig_len - 1 ) return -1;

  fa512_hash_to_point_ktp256( c0, sm + 2, NONCELEN + msg_len );

  /* `verify_raw` still runs the full NTT path when this non-standard
   * `c0` fails the round 3 vector. */
  int ok = falcon_inner_verify_raw( c0, sig, h, LOGN, tmp );

  memmove( m, sm + 2 + NONCELEN, msg_len );
  if( mlen ) *mlen = msg_len;
  return ok ? 0 : -1;
}

#else /* !HAVE_AVX512 */

int
falcon_ref_ktp256_crypto_sign_open( uint8_t       * m,  size_t * mlen,
                                    uint8_t const * sm, size_t   smlen,
                                    uint8_t const * pk ) {
  /* Non-AVX-512 builds keep the harness runnable; no KTP path here. */
  return falcon_ref_xkcp_crypto_sign_open( m, mlen, sm, smlen, pk );
}

#endif /* HAVE_AVX512 */
