/* Barrett AVX-512 verifier with always-reduced butterfly add/sub.
 * Same skeleton as falcon_avx512_barrett, but no lazy reduction. */

#include "falcon_avx512_common.h"
#include "falcon_twiddle.h"

#if HAVE_AVX512

/* Barrett field multiplication. */

#define BARRETT_M 43687U
#define BARRETT_K 29

static inline __m512i
fq_mul_v( __m512i a, __m512i b ) {
  const __m512i Mv = _mm512_set1_epi32( (int)BARRETT_M );
  const __m512i Qv = _mm512_set1_epi32( Q );
  const __m512i mask_even = _mm512_set1_epi64( 0xFFFFFFFFLL );

  __m512i product = _mm512_mullo_epi32( a, b );

  __m512i wide_e = _mm512_mul_epu32( product, Mv );
  __m512i qest_e = _mm512_srli_epi64( wide_e, BARRETT_K );

  __m512i prod_o = _mm512_srli_epi64( product, 32 );
  __m512i wide_o = _mm512_mul_epu32( prod_o, Mv );
  __m512i qest_o = _mm512_srli_epi64( wide_o, BARRETT_K );
  __m512i qest_o_shifted = _mm512_slli_epi64( qest_o, 32 );

  __m512i qest = _mm512_or_si512( _mm512_and_si512( qest_e, mask_even ),
                                  qest_o_shifted );

  __m512i r = _mm512_sub_epi32( product, _mm512_mullo_epi32( qest, Qv ) );

  __m512i d    = _mm512_sub_epi32( r, Qv );
  __m512i sign = _mm512_srai_epi32( d, 31 );
  return _mm512_add_epi32( d, _mm512_and_si512( Qv, sign ) );
}

static inline __m256i
fq_mul_avx2( __m256i a, __m256i b ) {
  const __m256i Mv = _mm256_set1_epi32( (int)BARRETT_M );
  const __m256i Qv = _mm256_set1_epi32( Q );
  const __m256i mask_even = _mm256_set1_epi64x( 0xFFFFFFFFLL );

  __m256i product = _mm256_mullo_epi32( a, b );
  __m256i wide_e = _mm256_mul_epu32( product, Mv );
  __m256i qest_e = _mm256_srli_epi64( wide_e, BARRETT_K );
  __m256i prod_o = _mm256_srli_epi64( product, 32 );
  __m256i wide_o = _mm256_mul_epu32( prod_o, Mv );
  __m256i qest_o = _mm256_srli_epi64( wide_o, BARRETT_K );
  __m256i qest_o_shifted = _mm256_slli_epi64( qest_o, 32 );
  __m256i qest = _mm256_or_si256( _mm256_and_si256( qest_e, mask_even ),
                                  qest_o_shifted );
  __m256i r = _mm256_sub_epi32( product, _mm256_mullo_epi32( qest, Qv ) );
  __m256i d    = _mm256_sub_epi32( r, Qv );
  __m256i sign = _mm256_srai_epi32( d, 31 );
  return _mm256_add_epi32( d, _mm256_and_si256( Qv, sign ) );
}

static inline __m128i
fq_mul_sse( __m128i a, __m128i b ) {
  const __m128i Mv = _mm_set1_epi32( (int)BARRETT_M );
  const __m128i Qv = _mm_set1_epi32( Q );
  const __m128i mask_even = _mm_set1_epi64x( 0xFFFFFFFFLL );

  __m128i product = _mm_mullo_epi32( a, b );
  __m128i wide_e = _mm_mul_epu32( product, Mv );
  __m128i qest_e = _mm_srli_epi64( wide_e, BARRETT_K );
  __m128i prod_o = _mm_srli_epi64( product, 32 );
  __m128i wide_o = _mm_mul_epu32( prod_o, Mv );
  __m128i qest_o = _mm_srli_epi64( wide_o, BARRETT_K );
  __m128i qest_o_shifted = _mm_slli_epi64( qest_o, 32 );
  __m128i qest = _mm_or_si128( _mm_and_si128( qest_e, mask_even ),
                               qest_o_shifted );
  __m128i r = _mm_sub_epi32( product, _mm_mullo_epi32( qest, Qv ) );
  __m128i d    = _mm_sub_epi32( r, Qv );
  __m128i sign = _mm_srai_epi32( d, 31 );
  return _mm_add_epi32( d, _mm_and_si128( Qv, sign ) );
}

/* Always-reduced add/sub. */

static inline __m512i
fq_add_v( __m512i a, __m512i b ) {
  const __m512i Qv = _mm512_set1_epi32( Q );
  __m512i s    = _mm512_add_epi32( a, b );
  __m512i d    = _mm512_sub_epi32( s, Qv );
  __m512i sign = _mm512_srai_epi32( d, 31 );
  return _mm512_add_epi32( d, _mm512_and_si512( Qv, sign ) );
}

static inline __m512i
fq_sub_v( __m512i a, __m512i b ) {
  const __m512i Qv = _mm512_set1_epi32( Q );
  __m512i d    = _mm512_sub_epi32( a, b );
  __m512i sign = _mm512_srai_epi32( d, 31 );
  return _mm512_add_epi32( d, _mm512_and_si512( Qv, sign ) );
}

static inline __m256i
fq_add_avx2( __m256i a, __m256i b ) {
  const __m256i Qv = _mm256_set1_epi32( Q );
  __m256i s    = _mm256_add_epi32( a, b );
  __m256i d    = _mm256_sub_epi32( s, Qv );
  __m256i sign = _mm256_srai_epi32( d, 31 );
  return _mm256_add_epi32( d, _mm256_and_si256( Qv, sign ) );
}

static inline __m256i
fq_sub_avx2( __m256i a, __m256i b ) {
  const __m256i Qv = _mm256_set1_epi32( Q );
  __m256i d    = _mm256_sub_epi32( a, b );
  __m256i sign = _mm256_srai_epi32( d, 31 );
  return _mm256_add_epi32( d, _mm256_and_si256( Qv, sign ) );
}

static inline __m128i
fq_add_sse( __m128i a, __m128i b ) {
  const __m128i Qv = _mm_set1_epi32( Q );
  __m128i s    = _mm_add_epi32( a, b );
  __m128i d    = _mm_sub_epi32( s, Qv );
  __m128i sign = _mm_srai_epi32( d, 31 );
  return _mm_add_epi32( d, _mm_and_si128( Qv, sign ) );
}

static inline __m128i
fq_sub_sse( __m128i a, __m128i b ) {
  const __m128i Qv = _mm_set1_epi32( Q );
  __m128i d    = _mm_sub_epi32( a, b );
  __m128i sign = _mm_srai_epi32( d, 31 );
  return _mm_add_epi32( d, _mm_and_si128( Qv, sign ) );
}

/* Forward NTT, always reduced. */

static void
ntt_fwd_avx512( falcon_fq_t * out, falcon_fq_t const * in ) {
  memcpy( out, in, sizeof(falcon_fq_t) * N );

  uint32_t t = N;
  uint32_t m = 1;
  while( m < N ) {
    t >>= 1;

    if( t >= 16 ) {
      for( uint32_t i=0; i<m; i++ ) {
        uint32_t j1 = 2 * i * t;
        __m512i sv = _mm512_set1_epi32( (int)falcon_psi_positive[ m + i ] );
        for( uint32_t j=j1; j<j1+t; j+=16 ) {
          __m512i u  = _mm512_loadu_si512( (void const *)( out + j ) );
          __m512i v  = fq_mul_v( _mm512_loadu_si512( (void const *)( out + j + t ) ), sv );
          _mm512_storeu_si512( (void *)( out + j     ), fq_add_v( u, v ) );
          _mm512_storeu_si512( (void *)( out + j + t ), fq_sub_v( u, v ) );
        }
      }
    } else if( t == 8 ) {
      for( uint32_t i=0; i<m; i++ ) {
        uint32_t j1 = 2 * i * t;
        __m256i sv = _mm256_set1_epi32( (int)falcon_psi_positive[ m + i ] );
        __m256i u  = _mm256_loadu_si256( (void const *)( out + j1 ) );
        __m256i v  = fq_mul_avx2( _mm256_loadu_si256( (void const *)( out + j1 + t ) ), sv );
        _mm256_storeu_si256( (void *)( out + j1     ), fq_add_avx2( u, v ) );
        _mm256_storeu_si256( (void *)( out + j1 + t ), fq_sub_avx2( u, v ) );
      }
    } else if( t == 4 ) {
      for( uint32_t i=0; i<m; i++ ) {
        uint32_t j1 = 8 * i;
        __m128i sv = _mm_set1_epi32( (int)falcon_psi_positive[ m + i ] );
        __m128i u  = _mm_loadu_si128( (void const *)( out + j1 ) );
        __m128i v  = fq_mul_sse( _mm_loadu_si128( (void const *)( out + j1 + 4 ) ), sv );
        _mm_storeu_si128( (void *)( out + j1     ), fq_add_sse( u, v ) );
        _mm_storeu_si128( (void *)( out + j1 + 4 ), fq_sub_sse( u, v ) );
      }
    } else if( t == 2 ) {
      for( uint32_t i=0; i<m; i+=2 ) {
        uint32_t j1 = 4 * i;
        __m128i d0 = _mm_loadu_si128( (void const *)( out + j1     ) );
        __m128i d1 = _mm_loadu_si128( (void const *)( out + j1 + 4 ) );
        __m128i u  = _mm_unpacklo_epi64( d0, d1 );
        __m128i v  = _mm_unpackhi_epi64( d0, d1 );
        uint32_t s0 = falcon_psi_positive[ m + i     ];
        uint32_t s1 = falcon_psi_positive[ m + i + 1 ];
        __m128i sv = _mm_setr_epi32( (int)s0, (int)s0, (int)s1, (int)s1 );
        __m128i vs = fq_mul_sse( v, sv );
        __m128i ru = fq_add_sse( u, vs );
        __m128i rv = fq_sub_sse( u, vs );
        __m128i o0 = _mm_unpacklo_epi64( ru, rv );
        __m128i o1 = _mm_unpackhi_epi64( ru, rv );
        _mm_storeu_si128( (void *)( out + j1     ), o0 );
        _mm_storeu_si128( (void *)( out + j1 + 4 ), o1 );
      }
    } else { /* t == 1 */
      for( uint32_t i=0; i<m; i+=4 ) {
        __m128i d0  = _mm_loadu_si128( (void const *)( out + 2*i     ) );
        __m128i d1  = _mm_loadu_si128( (void const *)( out + 2*i + 4 ) );
        __m128i d0s = _mm_shuffle_epi32( d0, _MM_SHUFFLE( 3, 1, 2, 0 ) );
        __m128i d1s = _mm_shuffle_epi32( d1, _MM_SHUFFLE( 3, 1, 2, 0 ) );
        __m128i u   = _mm_unpacklo_epi64( d0s, d1s );
        __m128i v   = _mm_unpackhi_epi64( d0s, d1s );
        __m128i sv  = _mm_loadu_si128( (void const *)( falcon_psi_positive + m + i ) );
        __m128i vs  = fq_mul_sse( v, sv );
        __m128i ru  = fq_add_sse( u, vs );
        __m128i rv  = fq_sub_sse( u, vs );
        __m128i o0  = _mm_unpacklo_epi64( ru, rv );
        __m128i o1  = _mm_unpackhi_epi64( ru, rv );
        o0 = _mm_shuffle_epi32( o0, _MM_SHUFFLE( 3, 1, 2, 0 ) );
        o1 = _mm_shuffle_epi32( o1, _MM_SHUFFLE( 3, 1, 2, 0 ) );
        _mm_storeu_si128( (void *)( out + 2*i     ), o0 );
        _mm_storeu_si128( (void *)( out + 2*i + 4 ), o1 );
      }
    }
    m <<= 1;
  }
  /* Already in [0,Q). */
}

/* Inverse NTT, always reduced. */

static void
ntt_inv_avx512( falcon_fq_t * out, falcon_fq_t const * in ) {
  memcpy( out, in, sizeof(falcon_fq_t) * N );

  uint32_t t = 1;
  uint32_t m = N;
  while( m > 1 ) {
    uint32_t h = m >> 1;

    if( t >= 16 ) {
      uint32_t j1 = 0;
      for( uint32_t i=0; i<h; i++ ) {
        __m512i sv = _mm512_set1_epi32( (int)falcon_psi_negative[ h + i ] );
        for( uint32_t j=j1; j<j1+t; j+=16 ) {
          __m512i u    = _mm512_loadu_si512( (void const *)( out + j ) );
          __m512i v    = _mm512_loadu_si512( (void const *)( out + j + t ) );
          __m512i sum  = fq_add_v( u, v );
          __m512i diff = fq_mul_v( fq_sub_v( u, v ), sv );
          _mm512_storeu_si512( (void *)( out + j     ), sum  );
          _mm512_storeu_si512( (void *)( out + j + t ), diff );
        }
        j1 += 2 * t;
      }
    } else if( t == 8 ) {
      uint32_t j1 = 0;
      for( uint32_t i=0; i<h; i++ ) {
        __m256i sv = _mm256_set1_epi32( (int)falcon_psi_negative[ h + i ] );
        __m256i u  = _mm256_loadu_si256( (void const *)( out + j1 ) );
        __m256i v  = _mm256_loadu_si256( (void const *)( out + j1 + t ) );
        __m256i sum  = fq_add_avx2( u, v );
        __m256i diff = fq_mul_avx2( fq_sub_avx2( u, v ), sv );
        _mm256_storeu_si256( (void *)( out + j1     ), sum  );
        _mm256_storeu_si256( (void *)( out + j1 + t ), diff );
        j1 += 2 * t;
      }
    } else if( t == 4 ) {
      uint32_t j1 = 0;
      for( uint32_t i=0; i<h; i++ ) {
        __m128i sv = _mm_set1_epi32( (int)falcon_psi_negative[ h + i ] );
        __m128i u  = _mm_loadu_si128( (void const *)( out + j1     ) );
        __m128i v  = _mm_loadu_si128( (void const *)( out + j1 + 4 ) );
        __m128i sum  = fq_add_sse( u, v );
        __m128i diff = fq_mul_sse( fq_sub_sse( u, v ), sv );
        _mm_storeu_si128( (void *)( out + j1     ), sum  );
        _mm_storeu_si128( (void *)( out + j1 + 4 ), diff );
        j1 += 8;
      }
    } else if( t == 2 ) {
      uint32_t j1 = 0;
      for( uint32_t i=0; i<h; i+=2 ) {
        __m128i d0 = _mm_loadu_si128( (void const *)( out + j1     ) );
        __m128i d1 = _mm_loadu_si128( (void const *)( out + j1 + 4 ) );
        __m128i u  = _mm_unpacklo_epi64( d0, d1 );
        __m128i v  = _mm_unpackhi_epi64( d0, d1 );
        uint32_t s0 = falcon_psi_negative[ h + i     ];
        uint32_t s1 = falcon_psi_negative[ h + i + 1 ];
        __m128i sv = _mm_setr_epi32( (int)s0, (int)s0, (int)s1, (int)s1 );
        __m128i sum  = fq_add_sse( u, v );
        __m128i diff = fq_mul_sse( fq_sub_sse( u, v ), sv );
        __m128i o0 = _mm_unpacklo_epi64( sum, diff );
        __m128i o1 = _mm_unpackhi_epi64( sum, diff );
        _mm_storeu_si128( (void *)( out + j1     ), o0 );
        _mm_storeu_si128( (void *)( out + j1 + 4 ), o1 );
        j1 += 8;
      }
    } else { /* t == 1 */
      for( uint32_t i=0; i<h; i+=4 ) {
        __m128i d0  = _mm_loadu_si128( (void const *)( out + 2*i     ) );
        __m128i d1  = _mm_loadu_si128( (void const *)( out + 2*i + 4 ) );
        __m128i d0s = _mm_shuffle_epi32( d0, _MM_SHUFFLE( 3, 1, 2, 0 ) );
        __m128i d1s = _mm_shuffle_epi32( d1, _MM_SHUFFLE( 3, 1, 2, 0 ) );
        __m128i u   = _mm_unpacklo_epi64( d0s, d1s );
        __m128i v   = _mm_unpackhi_epi64( d0s, d1s );
        __m128i sv  = _mm_loadu_si128( (void const *)( falcon_psi_negative + h + i ) );
        __m128i sum  = fq_add_sse( u, v );
        __m128i diff = fq_mul_sse( fq_sub_sse( u, v ), sv );
        __m128i o0 = _mm_unpacklo_epi64( sum, diff );
        __m128i o1 = _mm_unpackhi_epi64( sum, diff );
        o0 = _mm_shuffle_epi32( o0, _MM_SHUFFLE( 3, 1, 2, 0 ) );
        o1 = _mm_shuffle_epi32( o1, _MM_SHUFFLE( 3, 1, 2, 0 ) );
        _mm_storeu_si128( (void *)( out + 2*i     ), o0 );
        _mm_storeu_si128( (void *)( out + 2*i + 4 ), o1 );
      }
    }
    t <<= 1;
    m >>= 1;
  }

  /* Final scale by N^{-1} = 12265. */
  __m512i n_inv = _mm512_set1_epi32( 12265 );
  for( uint32_t j=0; j<(uint32_t)N; j+=16 ) {
    __m512i x = _mm512_loadu_si512( (void const *)( out + j ) );
    _mm512_storeu_si512( (void *)( out + j ), fq_mul_v( x, n_inv ) );
  }
}

/* Verify (NIST API). */

enum { NONCELEN = 40, PK_HEADER = 0x09, SIG_HEADER = 0x29 };

int
falcon_avx512_barrett_alwaysred_crypto_sign_open( uint8_t       * m,  size_t * mlen,
                                                  uint8_t const * sm, size_t   smlen,
                                                  uint8_t const * pk ) {
  if( UNLIKELY( pk[ 0 ] != PK_HEADER ) )            return -1;
  if( UNLIKELY( smlen < 2 + NONCELEN + 1 ) )        return -1;

  size_t sig_field_len = ( (size_t)sm[ 0 ] << 8 ) | (size_t)sm[ 1 ];
  if( UNLIKELY( sig_field_len < 1 ) )                       return -1;
  if( UNLIKELY( sig_field_len > smlen - 2 - NONCELEN ) )    return -1;

  size_t          msg_len = smlen - 2 - NONCELEN - sig_field_len;
  uint8_t const * nonce   = sm + 2;
  uint8_t const * msg     = sm + 2 + NONCELEN;
  uint8_t const * esig    = sm + 2 + NONCELEN + msg_len;

  if( UNLIKELY( esig[ 0 ] != SIG_HEADER ) )                 return -1;

  falcon_pubkey_t    pubk[1];
  falcon_signature_t sig [1];
  if( UNLIKELY( fa512_parse_pk     ( pubk,    pk + 1                  ) ) ) return -1;
  if( UNLIKELY( fa512_parse_comp_s2( sig->s2, esig + 1, sig_field_len-1 ) ) ) return -1;

  falcon_fq_t c[ N + 16 ] __attribute__((aligned(64)));
  fa512_hash_to_point( c, nonce, msg, msg_len );

  falcon_fq_t s2_ntt[ N ] __attribute__((aligned(64)));
  falcon_fq_t h_ntt [ N ] __attribute__((aligned(64)));
  falcon_fq_t prod  [ N ] __attribute__((aligned(64)));
  ntt_fwd_avx512( s2_ntt, sig->s2 );
  ntt_fwd_avx512( h_ntt,  pubk->h );
  for( int i=0; i<N; i+=16 ) {
    __m512i a = _mm512_loadu_si512( (void const *)( s2_ntt + i ) );
    __m512i b = _mm512_loadu_si512( (void const *)( h_ntt  + i ) );
    _mm512_storeu_si512( (void *)( prod + i ), fq_mul_v( a, b ) );
  }

  falcon_fq_t pmm[ N ] __attribute__((aligned(64)));
  ntt_inv_avx512( pmm, prod );

  if( !fa512_norm_check_ok( c, pmm, sig ) ) return -1;

  if( m && msg_len ) memmove( m, msg, msg_len );
  if( mlen ) *mlen = msg_len;
  return 0;
}

#else /* !HAVE_AVX512 */

int
falcon_avx512_barrett_alwaysred_crypto_sign_open( uint8_t       * m,  size_t * mlen,
                                                  uint8_t const * sm, size_t   smlen,
                                                  uint8_t const * pk ) {
  return falcon_ref_xkcp_crypto_sign_open( m, mlen, sm, smlen, pk );
}

#endif /* HAVE_AVX512 */
