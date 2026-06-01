/* Shared parsing, hash-to-point, and norm-check code for AVX-512
 * verifiers.  The variants differ in their NTT field multiplication. */

#ifndef CONTRIB_FALCON_AVX512_COMMON_H
#define CONTRIB_FALCON_AVX512_COMMON_H

#include "falcon.h"

#include <stdint.h>
#include <stddef.h>
#include <string.h>

#define Q     FALCON_Q
#define N     FALCON_N
#define LOGN  FALCON_LOGN
#define BETA2 FALCON_BETA2
#define K_REJ ( ( 1 << 16 ) / Q )

#define LIKELY(x)   __builtin_expect(!!(x), 1)
#define UNLIKELY(x) __builtin_expect(!!(x), 0)

#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512DQ__)
#define HAVE_AVX512 1
#else
#define HAVE_AVX512 0
#endif

typedef uint32_t falcon_fq_t;
typedef struct { falcon_fq_t h [ N ]; } falcon_pubkey_t;
typedef struct { uint8_t nonce[ 40 ]; falcon_fq_t s2[ N ]; } falcon_signature_t;

/* Pornin SHAKE256. */
typedef struct {
  union { uint64_t A[25]; uint8_t dbuf[200]; } st;
  uint64_t dptr;
} inner_shake256_context;
void falcon_inner_i_shake256_init   ( inner_shake256_context * sc );
void falcon_inner_i_shake256_inject ( inner_shake256_context * sc,
                                      const void * data, size_t len );
void falcon_inner_i_shake256_flip   ( inner_shake256_context * sc );
void falcon_inner_i_shake256_extract( inner_shake256_context * sc,
                                      void * out, size_t len );

/* KTP256 hash-to-point over contiguous nonce||msg. */
void fa512_hash_to_point_ktp256( uint16_t * out,
                                 uint8_t const * in, size_t in_len );

#if HAVE_AVX512

#include <immintrin.h>

static inline uint32_t fa512_load_u32( void const * p ) { uint32_t v; memcpy(&v,p,4); return v; }
static inline uint64_t fa512_load_u64( void const * p ) { uint64_t v; memcpy(&v,p,8); return v; }

/* Public-key parsing: 16 coefficients/iteration. */

#define FA512_PK_LEN     16
#define FA512_PK_STRIDE  ( ( FA512_PK_LEN / 4 ) * 7 )

static inline int
fa512_parse_pk( falcon_pubkey_t * pubkey, uint8_t const * h_packed ) {
  __m512i mask    = _mm512_set1_epi32( ( 1 << 14 ) - 1 );
  __m512i Qv      = _mm512_set1_epi32( Q );
  __m512i offsets = _mm512_setr_epi32( 18, 4, 14, 0, 18, 4, 14, 0,
                                       18, 4, 14, 0, 18, 4, 14, 0 );

  for( int i=0; i<N/FA512_PK_LEN; i++ ) {
    uint8_t const * in  = h_packed + i * FA512_PK_STRIDE;
    falcon_fq_t   * out = pubkey->h + i * FA512_PK_LEN;

    __m512i compressed = _mm512_setr_epi32(
        (int)fa512_load_u32( in +  0 ), (int)fa512_load_u32( in +  0 ),
        (int)fa512_load_u32( in +  3 ), (int)fa512_load_u32( in +  3 ),
        (int)fa512_load_u32( in +  7 ), (int)fa512_load_u32( in +  7 ),
        (int)fa512_load_u32( in + 10 ), (int)fa512_load_u32( in + 10 ),
        (int)fa512_load_u32( in + 14 ), (int)fa512_load_u32( in + 14 ),
        (int)fa512_load_u32( in + 17 ), (int)fa512_load_u32( in + 17 ),
        (int)fa512_load_u32( in + 21 ), (int)fa512_load_u32( in + 21 ),
        (int)fa512_load_u32( in + 24 ), (int)fa512_load_u32( in + 24 ) );

    __m512i bswap_idx = _mm512_set4_epi32(
        0x0C0D0E0F, 0x08090A0B, 0x04050607, 0x00010203 );
    __m512i swapped = _mm512_shuffle_epi8( compressed, bswap_idx );
    __m512i shifted = _mm512_srlv_epi32( swapped, offsets );
    __m512i masked  = _mm512_and_si512(  shifted, mask );

    __mmask16 ge = _mm512_cmpge_epu32_mask( masked, Qv );
    if( UNLIKELY( ge ) ) return -1;

    _mm512_storeu_si512( (void *)out, masked );
  }
  return 0;
}

/* Compressed s2 parsing: 64-bit bswap/lzcnt window. */

static inline int
fa512_parse_comp_s2( falcon_fq_t * out_s2,
                     uint8_t const * in, size_t in_len ) {
  uint8_t const * s2     = in;
  size_t          s2_len = in_len;
  size_t          length = s2_len * 8;

  uint8_t padded[ 1024 + 8 ] __attribute__((aligned(64)));
  if( UNLIKELY( s2_len + 8 > sizeof(padded) ) ) return -1;
  memcpy( padded,          s2, s2_len );
  memset( padded + s2_len, 0,  8 );

  int results[ N ] __attribute__((aligned(64)));

  size_t   abs_bit = 0;
  uint64_t word    = __builtin_bswap64( fa512_load_u64( padded ) );
  int      avail   = 64;

  for( int i=0; i<N; i++ ) {
    if( UNLIKELY( avail < 16 ) ) {
      if( UNLIKELY( abs_bit + 9 > length ) ) return -1;
      size_t bp = abs_bit >> 3;
      int    sb = (int)( abs_bit & 7 );
      word  = __builtin_bswap64( fa512_load_u64( padded + bp ) ) << sb;
      avail = 64 - sb;
    }

    int      sign = (int)( word >> 63 );
    int      low  = (int)( ( word >> 56 ) & 0x7F );
    uint64_t tail = word << 8;

    int high;
    if( LIKELY( tail ) ) {
      high = (int)__builtin_clzll( tail );
      int advance = 9 + high;
      word    <<= advance;
      avail    -= advance;
      abs_bit  += (size_t)advance;
    } else {
      high = avail - 8;
      abs_bit += 8;

      for(;;) {
        if( UNLIKELY( abs_bit >= length ) ) return -1;
        size_t bp = abs_bit >> 3;
        int    sb = (int)( abs_bit & 7 );
        word  = __builtin_bswap64( fa512_load_u64( padded + bp ) ) << sb;
        avail = 64 - sb;
        if( LIKELY( word ) ) {
          int extra   = (int)__builtin_clzll( word );
          high       += extra;
          int advance = extra + 1;
          word    <<= advance;
          avail    -= advance;
          abs_bit  += (size_t)advance;
          break;
        }
        if( UNLIKELY( (uint32_t)high >= (uint32_t)( Q >> 7 ) ) ) return -1;
        high    += avail;
        abs_bit += (size_t)avail;
        avail    = 0;
      }
    }

    if( UNLIKELY( abs_bit > length ) ) return -1;

    int mag = ( high << 7 ) | low;
    if( UNLIKELY( mag >= Q ) ) return -1;
    results[ i ] = sign ? -mag : mag;
  }

  for( int i=0; i<N; i+=16 ) {
    __m512i v        = _mm512_loadu_si512( (void const *)( results + i ) );
    __m512i neg_mask = _mm512_srai_epi32( v, 31 );
    __m512i corr     = _mm512_and_si512( _mm512_set1_epi32( Q ), neg_mask );
    _mm512_storeu_si512( (void *)( out_s2 + i ), _mm512_add_epi32( v, corr ) );
  }

  if( abs_bit < length ) {
    size_t bp = abs_bit >> 3;
    int    sb = (int)( abs_bit & 7 );
    if( sb ) {
      uint8_t trail_mask = (uint8_t)( 0xFF >> sb );
      if( UNLIKELY( s2[ bp ] & trail_mask ) ) return -1;
      bp++;
    }
    for( size_t j=bp; j<s2_len; j++ ) {
      if( UNLIKELY( s2[ j ] ) ) return -1;
    }
  }
  return 0;
}

/* Hash-to-point: Pornin SHAKE + vector rejection sampling. */

static inline void
fa512_hash_to_point( falcon_fq_t * c,
                     uint8_t const * nonce_40,
                     uint8_t const * msg, size_t msg_len ) {
  inner_shake256_context sc[1];
  falcon_inner_i_shake256_init   ( sc );
  falcon_inner_i_shake256_inject ( sc, nonce_40, 40 );
  falcon_inner_i_shake256_inject ( sc, msg,      msg_len );
  falcon_inner_i_shake256_flip   ( sc );

  uint8_t sample[ 128 ] __attribute__((aligned(64)));
  size_t  offset = sizeof( sample );

  for( int i=0; i<N; ) {
    if( UNLIKELY( offset >= sizeof( sample ) ) ) {
      falcon_inner_i_shake256_extract( sc, sample, sizeof( sample ) );
      offset = 0;
    }
    __m256i v16     = _mm256_loadu_si256( (__m256i const *)( sample + offset ) );
    __m256i hi      = _mm256_srli_epi16( v16, 8 );
    __m256i lo      = _mm256_slli_epi16( v16, 8 );
    __m256i v16_be  = _mm256_or_si256( hi, lo );
    __m512i batch   = _mm512_cvtepu16_epi32( v16_be );

    __m512i  kv     = _mm512_set1_epi32( (int)( K_REJ * Q ) );
    __mmask16 mask  = _mm512_cmplt_epu32_mask( batch, kv );
    __m512i  comp   = _mm512_maskz_compress_epi32( mask, batch );

    __m512i Kv      = _mm512_set1_epi32( K_REJ );
    __m512i Qv      = _mm512_set1_epi32( Q );
    __m512i q       = _mm512_srli_epi32( _mm512_mullo_epi32( comp, Kv ), 16 );
    __m512i r       = _mm512_sub_epi32( comp, _mm512_mullo_epi32( q, Qv ) );
    __mmask16 ov    = _mm512_cmpge_epu32_mask( r, Qv );
    __m512i corr    = _mm512_mask_sub_epi32( r, ov, r, Qv );

    _mm512_storeu_si512( (void *)( c + i ), corr );
    offset += 32;
    i      += __builtin_popcount( mask );
  }
}

/* Shared norm-check epilogue (16-wide).  The norm only needs the
 * magnitude of the centered representative: for canonical a, b the
 * single subtraction d = a - b lies in (-Q,Q) and
 * |center(d)| = min(|d|, Q - |d|), with no reduction to [0,Q). */

static inline int
fa512_norm_check_ok( falcon_fq_t const * c, falcon_fq_t const * pmm,
                     falcon_signature_t const * sig ) {
  const __m512i Qv      = _mm512_set1_epi32( Q );
  const __m512i mask_lo = _mm512_set1_epi64( 0x00000000FFFFFFFFLL );
  __m512i acc = _mm512_setzero_si512();
  for( int i=0; i<N; i+=16 ) {
    __m512i a  = _mm512_loadu_si512( (void const *)( c   + i ) );
    __m512i b  = _mm512_loadu_si512( (void const *)( pmm + i ) );
    __m512i d  = _mm512_abs_epi32( _mm512_sub_epi32( a, b ) );
    __m512i s1 = _mm512_min_epu32( d, _mm512_sub_epi32( Qv, d ) );
    __m512i t  = _mm512_loadu_si512( (void const *)( sig->s2 + i ) );
    __m512i s2 = _mm512_min_epu32( t, _mm512_sub_epi32( Qv, t ) );
    /* s1, s2 <= Q/2 so s1^2 + s2^2 < 2^27 per lane. */
    __m512i sq = _mm512_add_epi32( _mm512_mullo_epi32( s1, s1 ),
                                   _mm512_mullo_epi32( s2, s2 ) );
    /* Widen to 64-bit lanes: 512 lanes x 2^27 needs 36 bits. */
    acc = _mm512_add_epi64( acc, _mm512_and_si512( sq, mask_lo ) );
    acc = _mm512_add_epi64( acc, _mm512_srli_epi64( sq, 32 ) );
  }
  return _mm512_reduce_add_epi64( acc ) <= (long)BETA2;
}

#endif /* HAVE_AVX512 */

#endif /* CONTRIB_FALCON_AVX512_COMMON_H */
