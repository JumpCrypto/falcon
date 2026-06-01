/* AVX-512 Falcon-512 verifier using SIMD NTT and Shoup-Harvey
 * multiplication with lazy reductions.
 *
 * For twiddle s, precompute s' = floor(s*2^32/q), then compute
 * q_hat = high32(v*s') and r = v*s - q_hat*q which is in [0,2q) for
 * any 32-bit v.
 *
 * - forward: u' = u + v*s, v' = u - v*s + 2q; elements grow to
 *   (2k+1)q = 19q after the 9 passes (9 = log2 512), and a trailing
 *   foldless Barrett reduction (M = floor(2^27/q)) returns lanes in
 *   [0,2q).
 * - Hadamard: one-shot Barrett on [0,2q) x [0,2q), output lazy in
 *   [0,2q).
 * - inverse: u' = u + v unreduced across all passes (up to 512q
 *   entering the last), v' = (u - v + off_q*q)*s with the
 *   non-negativity offset off_q doubling every pass from 2.  N^{-1}
 *   is fused into the last pass, which folds outputs to [0,q) for
 *   the norm check. */

#include "falcon_avx512_common.h"
#include "falcon_twiddle.h"

#if HAVE_AVX512

/* Fallback entry point. */
int falcon_ref_xkcp_crypto_sign_open( uint8_t * m, size_t * mlen,
                                 uint8_t const * sm, size_t smlen,
                                 uint8_t const * pk );

/* s' tables for Shoup multiplication. */

static falcon_fq_t s_prime_pos[ N ] __attribute__((aligned(64)));
static falcon_fq_t s_prime_neg[ N ] __attribute__((aligned(64)));

#define S_PRIME_NINV  ( (uint32_t)( ( (uint64_t)12265 << 32 ) / Q ) )  /* (12265<<32)/Q */

/* Last inverse pass twiddle with N^{-1} fused in: psi^{-1}[1]*N^{-1} mod Q. */
static uint32_t s_last_fused;
static uint32_t s_prime_last_fused;

__attribute__((constructor))
static void
init_shoup_tables( void ) {
  for( int i=0; i<N; i++ ) {
    s_prime_pos[ i ] = (uint32_t)(
        ( (uint64_t)falcon_psi_positive[ i ] << 32 ) / Q );
    s_prime_neg[ i ] = (uint32_t)(
        ( (uint64_t)falcon_psi_negative[ i ] << 32 ) / Q );
  }
  s_last_fused       = (uint32_t)( ( (uint64_t)falcon_psi_negative[ 1 ] * 12265 ) % Q );
  s_prime_last_fused = (uint32_t)( ( (uint64_t)s_last_fused << 32 ) / Q );
}

/* Lazy Shoup-Harvey field multiplication, 4/8/16-wide u32.
 * Output in [0,2Q); no conditional subtraction. */

static inline __m128i
fq_mul_shoup_sse( __m128i v, __m128i s, __m128i s_prime ) {
  const __m128i Qv      = _mm_set1_epi32( Q );
  const __m128i mask_lo = _mm_set1_epi64x( 0x00000000FFFFFFFFLL );

  __m128i wide_e = _mm_mul_epu32( v, s_prime );
  __m128i v_o    = _mm_srli_epi64( v,       32 );
  __m128i sp_o   = _mm_srli_epi64( s_prime, 32 );
  __m128i wide_o = _mm_mul_epu32( v_o, sp_o );

  __m128i q_hat_e = _mm_srli_epi64( wide_e, 32 );
  __m128i q_hat_o = _mm_andnot_si128( mask_lo, wide_o );
  __m128i q_hat   = _mm_or_si128( q_hat_e, q_hat_o );

  __m128i vs  = _mm_mullo_epi32( v, s );
  __m128i qq  = _mm_mullo_epi32( q_hat, Qv );
  return _mm_sub_epi32( vs, qq );
}

static inline __m256i
fq_mul_shoup_avx2( __m256i v, __m256i s, __m256i s_prime ) {
  const __m256i Qv      = _mm256_set1_epi32( Q );
  const __m256i mask_lo = _mm256_set1_epi64x( 0x00000000FFFFFFFFLL );

  __m256i wide_e = _mm256_mul_epu32( v, s_prime );
  __m256i v_o    = _mm256_srli_epi64( v,       32 );
  __m256i sp_o   = _mm256_srli_epi64( s_prime, 32 );
  __m256i wide_o = _mm256_mul_epu32( v_o, sp_o );

  __m256i q_hat_e = _mm256_srli_epi64( wide_e, 32 );
  __m256i q_hat_o = _mm256_andnot_si256( mask_lo, wide_o );
  __m256i q_hat   = _mm256_or_si256( q_hat_e, q_hat_o );

  __m256i vs  = _mm256_mullo_epi32( v, s );
  __m256i qq  = _mm256_mullo_epi32( q_hat, Qv );
  return _mm256_sub_epi32( vs, qq );
}

static inline __m512i
fq_mul_shoup_v( __m512i v, __m512i s, __m512i s_prime ) {
  const __m512i Qv      = _mm512_set1_epi32( Q );
  const __m512i mask_lo = _mm512_set1_epi64( 0x00000000FFFFFFFFLL );

  __m512i wide_e = _mm512_mul_epu32( v, s_prime );
  __m512i v_o    = _mm512_srli_epi64( v,       32 );
  __m512i sp_o   = _mm512_srli_epi64( s_prime, 32 );
  __m512i wide_o = _mm512_mul_epu32( v_o, sp_o );

  __m512i q_hat_e = _mm512_srli_epi64( wide_e, 32 );
  __m512i q_hat_o = _mm512_andnot_si512( mask_lo, wide_o );
  __m512i q_hat   = _mm512_or_si512( q_hat_e, q_hat_o );

  __m512i vs  = _mm512_mullo_epi32( v, s );
  __m512i qq  = _mm512_mullo_epi32( q_hat, Qv );
  return _mm512_sub_epi32( vs, qq );
}

/* Folded Shoup-Harvey multiplication (16-wide), for the inverse NTT
 * last pass: input any u32, output in [0,q). */

static inline __m512i
fq_mul_shoup_fold_v( __m512i v, __m512i s, __m512i s_prime ) {
  const __m512i Qv = _mm512_set1_epi32( Q );
  __m512i r = fq_mul_shoup_v( v, s, s_prime );
  __mmask16 ge = _mm512_cmpge_epu32_mask( r, Qv );
  return _mm512_mask_sub_epi32( r, ge, r, Qv );
}

/* Foldless Barrett reduction for the forward NTT output: for x < 19q
 * the product x*M fits 32 bits (M = floor(2^27/q) = 10921), so the
 * quotient estimate needs no widening multiply.  Output is a
 * representative of x mod q in [0,2q) (in fact < q+17). */

#define BARRETT27_M 10921U

static inline __m512i
fq_red27_v( __m512i x ) {
  const __m512i Mv = _mm512_set1_epi32( (int)BARRETT27_M );
  const __m512i Qv = _mm512_set1_epi32( Q );
  __m512i q_hat = _mm512_srli_epi32( _mm512_mullo_epi32( x, Mv ), 27 );
  return _mm512_sub_epi32( x, _mm512_mullo_epi32( q_hat, Qv ) );
}

/* Forward NTT; outputs in [0,2q) via a trailing foldless Barrett
 * reduction sweep. */

static void
ntt_fwd_avx512( falcon_fq_t * out, falcon_fq_t const * in ) {
  memcpy( out, in, sizeof(falcon_fq_t) * N );

  const __m512i Q2v512 = _mm512_set1_epi32( 2 * Q );
  const __m128i Q2v128 = _mm_set1_epi32( 2 * Q );

  uint32_t t = N;
  uint32_t m = 1;
  while( m < N ) {
    t >>= 1;

    if( t >= 16 ) {
      for( uint32_t i=0; i<m; i++ ) {
        uint32_t j1 = 2 * i * t;
        __m512i sv  = _mm512_set1_epi32( (int)falcon_psi_positive[ m + i ] );
        __m512i spv = _mm512_set1_epi32( (int)s_prime_pos        [ m + i ] );
        for( uint32_t j=j1; j<j1+t; j+=16 ) {
          __m512i u = _mm512_loadu_si512( (void const *)( out + j ) );
          __m512i v = fq_mul_shoup_v( _mm512_loadu_si512( (void const *)( out + j + t ) ), sv, spv );
          _mm512_storeu_si512( (void *)( out + j     ), _mm512_add_epi32( u, v ) );
          _mm512_storeu_si512( (void *)( out + j + t ),
                               _mm512_add_epi32( _mm512_sub_epi32( u, v ), Q2v512 ) );
        }
      }
    } else if( t == 8 ) {
      const __m256i Q2v256 = _mm256_set1_epi32( 2 * Q );
      for( uint32_t i=0; i<m; i++ ) {
        uint32_t j1 = 2 * i * t;
        __m256i sv  = _mm256_set1_epi32( (int)falcon_psi_positive[ m + i ] );
        __m256i spv = _mm256_set1_epi32( (int)s_prime_pos        [ m + i ] );
        __m256i u   = _mm256_loadu_si256( (void const *)( out + j1 ) );
        __m256i v   = fq_mul_shoup_avx2( _mm256_loadu_si256( (void const *)( out + j1 + t ) ), sv, spv );
        _mm256_storeu_si256( (void *)( out + j1     ), _mm256_add_epi32( u, v ) );
        _mm256_storeu_si256( (void *)( out + j1 + t ),
                             _mm256_add_epi32( _mm256_sub_epi32( u, v ), Q2v256 ) );
      }
    } else if( t == 4 ) {
      for( uint32_t i=0; i<m; i++ ) {
        uint32_t j1 = 8 * i;
        __m128i sv  = _mm_set1_epi32( (int)falcon_psi_positive[ m + i ] );
        __m128i spv = _mm_set1_epi32( (int)s_prime_pos        [ m + i ] );
        __m128i u   = _mm_loadu_si128( (void const *)( out + j1 ) );
        __m128i v   = fq_mul_shoup_sse( _mm_loadu_si128( (void const *)( out + j1 + 4 ) ), sv, spv );
        _mm_storeu_si128( (void *)( out + j1     ), _mm_add_epi32( u, v ) );
        _mm_storeu_si128( (void *)( out + j1 + 4 ),
                          _mm_add_epi32( _mm_sub_epi32( u, v ), Q2v128 ) );
      }
    } else if( t == 2 ) {
      /* 8 elements = two butterflies; transpose, butterfly, transpose back. */
      for( uint32_t i=0; i<m; i+=2 ) {
        uint32_t j1 = 4 * i;
        __m128i d0 = _mm_loadu_si128( (void const *)( out + j1     ) );
        __m128i d1 = _mm_loadu_si128( (void const *)( out + j1 + 4 ) );
        __m128i u  = _mm_unpacklo_epi64( d0, d1 );
        __m128i v  = _mm_unpackhi_epi64( d0, d1 );
        uint32_t s0  = falcon_psi_positive[ m + i     ];
        uint32_t s1  = falcon_psi_positive[ m + i + 1 ];
        uint32_t sp0 = s_prime_pos        [ m + i     ];
        uint32_t sp1 = s_prime_pos        [ m + i + 1 ];
        __m128i sv  = _mm_setr_epi32( (int)s0,  (int)s0,  (int)s1,  (int)s1 );
        __m128i spv = _mm_setr_epi32( (int)sp0, (int)sp0, (int)sp1, (int)sp1 );
        __m128i vs  = fq_mul_shoup_sse( v, sv, spv );
        __m128i ru  = _mm_add_epi32( u, vs );
        __m128i rv  = _mm_add_epi32( _mm_sub_epi32( u, vs ), Q2v128 );
        __m128i o0  = _mm_unpacklo_epi64( ru, rv );
        __m128i o1  = _mm_unpackhi_epi64( ru, rv );
        _mm_storeu_si128( (void *)( out + j1     ), o0 );
        _mm_storeu_si128( (void *)( out + j1 + 4 ), o1 );
      }
    } else { /* t == 1 */
      /* 8 elements = four butterflies; deinterleave, butterfly, restore. */
      for( uint32_t i=0; i<m; i+=4 ) {
        __m128i d0  = _mm_loadu_si128( (void const *)( out + 2*i     ) );
        __m128i d1  = _mm_loadu_si128( (void const *)( out + 2*i + 4 ) );
        __m128i d0s = _mm_shuffle_epi32( d0, _MM_SHUFFLE( 3, 1, 2, 0 ) );
        __m128i d1s = _mm_shuffle_epi32( d1, _MM_SHUFFLE( 3, 1, 2, 0 ) );
        __m128i u   = _mm_unpacklo_epi64( d0s, d1s );
        __m128i v   = _mm_unpackhi_epi64( d0s, d1s );
        __m128i sv  = _mm_loadu_si128( (void const *)( falcon_psi_positive + m + i ) );
        __m128i spv = _mm_loadu_si128( (void const *)( s_prime_pos         + m + i ) );
        __m128i vs  = fq_mul_shoup_sse( v, sv, spv );
        __m128i ru  = _mm_add_epi32( u, vs );
        __m128i rv  = _mm_add_epi32( _mm_sub_epi32( u, vs ), Q2v128 );
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

  /* Reduce [0,19q) to [0,2q); x*M fits 32 bits so no vpmuludq. */
  for( uint32_t j=0; j<(uint32_t)N; j+=16 ) {
    __m512i x = _mm512_loadu_si512( (void const *)( out + j ) );
    _mm512_storeu_si512( (void *)( out + j ), fq_red27_v( x ) );
  }
}

/* Inverse NTT; output normalized to [0,q) via the fused last pass. */

static void
ntt_inv_avx512( falcon_fq_t * out, falcon_fq_t const * in ) {
  memcpy( out, in, sizeof(falcon_fq_t) * N );

  uint32_t t     = 1;
  uint32_t m     = N;
  uint32_t off_q = 2;   /* Hadamard inputs are lazy in [0,2q) */
  while( m > 2 ) {
    uint32_t h   = m >> 1;
    uint32_t off = off_q * Q;

    if( t >= 16 ) {
      __m512i offv = _mm512_set1_epi32( (int)off );
      uint32_t j1 = 0;
      for( uint32_t i=0; i<h; i++ ) {
        __m512i sv  = _mm512_set1_epi32( (int)falcon_psi_negative[ h + i ] );
        __m512i spv = _mm512_set1_epi32( (int)s_prime_neg        [ h + i ] );
        for( uint32_t j=j1; j<j1+t; j+=16 ) {
          __m512i u    = _mm512_loadu_si512( (void const *)( out + j ) );
          __m512i v    = _mm512_loadu_si512( (void const *)( out + j + t ) );
          __m512i sum  = _mm512_add_epi32( u, v );
          __m512i diff = fq_mul_shoup_v(
              _mm512_add_epi32( _mm512_sub_epi32( u, v ), offv ), sv, spv );
          _mm512_storeu_si512( (void *)( out + j     ), sum );
          _mm512_storeu_si512( (void *)( out + j + t ), diff );
        }
        j1 += 2 * t;
      }
    } else if( t == 8 ) {
      __m256i offv = _mm256_set1_epi32( (int)off );
      uint32_t j1 = 0;
      for( uint32_t i=0; i<h; i++ ) {
        __m256i sv  = _mm256_set1_epi32( (int)falcon_psi_negative[ h + i ] );
        __m256i spv = _mm256_set1_epi32( (int)s_prime_neg        [ h + i ] );
        __m256i u    = _mm256_loadu_si256( (void const *)( out + j1 ) );
        __m256i v    = _mm256_loadu_si256( (void const *)( out + j1 + t ) );
        __m256i sum  = _mm256_add_epi32( u, v );
        __m256i diff = fq_mul_shoup_avx2( _mm256_add_epi32( _mm256_sub_epi32( u, v ), offv ), sv, spv );
        _mm256_storeu_si256( (void *)( out + j1     ), sum );
        _mm256_storeu_si256( (void *)( out + j1 + t ), diff );
        j1 += 2 * t;
      }
    } else if( t == 4 ) {
      __m128i offv = _mm_set1_epi32( (int)off );
      uint32_t j1 = 0;
      for( uint32_t i=0; i<h; i++ ) {
        __m128i sv  = _mm_set1_epi32( (int)falcon_psi_negative[ h + i ] );
        __m128i spv = _mm_set1_epi32( (int)s_prime_neg        [ h + i ] );
        __m128i u    = _mm_loadu_si128( (void const *)( out + j1     ) );
        __m128i v    = _mm_loadu_si128( (void const *)( out + j1 + 4 ) );
        __m128i sum  = _mm_add_epi32( u, v );
        __m128i diff = fq_mul_shoup_sse( _mm_add_epi32( _mm_sub_epi32( u, v ), offv ), sv, spv );
        _mm_storeu_si128( (void *)( out + j1     ), sum );
        _mm_storeu_si128( (void *)( out + j1 + 4 ), diff );
        j1 += 8;
      }
    } else if( t == 2 ) {
      __m128i offv = _mm_set1_epi32( (int)off );
      uint32_t j1 = 0;
      for( uint32_t i=0; i<h; i+=2 ) {
        __m128i d0 = _mm_loadu_si128( (void const *)( out + j1     ) );
        __m128i d1 = _mm_loadu_si128( (void const *)( out + j1 + 4 ) );
        __m128i u  = _mm_unpacklo_epi64( d0, d1 );
        __m128i v  = _mm_unpackhi_epi64( d0, d1 );
        uint32_t s0  = falcon_psi_negative[ h + i     ];
        uint32_t s1  = falcon_psi_negative[ h + i + 1 ];
        uint32_t sp0 = s_prime_neg        [ h + i     ];
        uint32_t sp1 = s_prime_neg        [ h + i + 1 ];
        __m128i sv  = _mm_setr_epi32( (int)s0,  (int)s0,  (int)s1,  (int)s1 );
        __m128i spv = _mm_setr_epi32( (int)sp0, (int)sp0, (int)sp1, (int)sp1 );
        __m128i sum  = _mm_add_epi32( u, v );
        __m128i diff = fq_mul_shoup_sse( _mm_add_epi32( _mm_sub_epi32( u, v ), offv ), sv, spv );
        __m128i o0 = _mm_unpacklo_epi64( sum, diff );
        __m128i o1 = _mm_unpackhi_epi64( sum, diff );
        _mm_storeu_si128( (void *)( out + j1     ), o0 );
        _mm_storeu_si128( (void *)( out + j1 + 4 ), o1 );
        j1 += 8;
      }
    } else { /* t == 1 */
      __m128i offv = _mm_set1_epi32( (int)off );
      for( uint32_t i=0; i<h; i+=4 ) {
        __m128i d0  = _mm_loadu_si128( (void const *)( out + 2*i     ) );
        __m128i d1  = _mm_loadu_si128( (void const *)( out + 2*i + 4 ) );
        __m128i d0s = _mm_shuffle_epi32( d0, _MM_SHUFFLE( 3, 1, 2, 0 ) );
        __m128i d1s = _mm_shuffle_epi32( d1, _MM_SHUFFLE( 3, 1, 2, 0 ) );
        __m128i u   = _mm_unpacklo_epi64( d0s, d1s );
        __m128i v   = _mm_unpackhi_epi64( d0s, d1s );
        __m128i sv  = _mm_loadu_si128( (void const *)( falcon_psi_negative + h + i ) );
        __m128i spv = _mm_loadu_si128( (void const *)( s_prime_neg         + h + i ) );
        __m128i sum  = _mm_add_epi32( u, v );
        __m128i diff = fq_mul_shoup_sse( _mm_add_epi32( _mm_sub_epi32( u, v ), offv ), sv, spv );
        __m128i o0 = _mm_unpacklo_epi64( sum, diff );
        __m128i o1 = _mm_unpackhi_epi64( sum, diff );
        o0 = _mm_shuffle_epi32( o0, _MM_SHUFFLE( 3, 1, 2, 0 ) );
        o1 = _mm_shuffle_epi32( o1, _MM_SHUFFLE( 3, 1, 2, 0 ) );
        _mm_storeu_si128( (void *)( out + 2*i     ), o0 );
        _mm_storeu_si128( (void *)( out + 2*i + 4 ), o1 );
      }
    }
    off_q <<= 1;
    t     <<= 1;
    m     >>= 1;
  }

  /* Last pass (m == 2, t == N/2) with N^{-1} = 12265 fused in:
   * out[j] = (u+v)*N^{-1}, out[j+t] = (u-v+off)*(psi^{-1}[1]*N^{-1}),
   * both folded to [0,q). */
  __m512i offv   = _mm512_set1_epi32( (int)( off_q * Q ) );
  __m512i n_inv  = _mm512_set1_epi32( 12265 );
  __m512i n_invp = _mm512_set1_epi32( (int)S_PRIME_NINV );
  __m512i sv     = _mm512_set1_epi32( (int)s_last_fused );
  __m512i spv    = _mm512_set1_epi32( (int)s_prime_last_fused );
  for( uint32_t j=0; j<(uint32_t)N/2; j+=16 ) {
    __m512i u    = _mm512_loadu_si512( (void const *)( out + j ) );
    __m512i v    = _mm512_loadu_si512( (void const *)( out + j + N/2 ) );
    __m512i sum  = fq_mul_shoup_fold_v( _mm512_add_epi32( u, v ), n_inv, n_invp );
    __m512i diff = fq_mul_shoup_fold_v(
        _mm512_add_epi32( _mm512_sub_epi32( u, v ), offv ), sv, spv );
    _mm512_storeu_si512( (void *)( out + j       ), sum );
    _mm512_storeu_si512( (void *)( out + j + N/2 ), diff );
  }
}

/* Verify (NIST API). */

enum { NONCELEN = 40, PK_HEADER = 0x09, SIG_HEADER = 0x29 };

int
falcon_avx512_parse( uint8_t const *      sm,   size_t smlen,
                     uint8_t const *      pk,
                     falcon_pubkey_t    * pubk,
                     falcon_signature_t * sig,
                     uint8_t const **     out_nonce,
                     uint8_t const **     out_msg,
                     size_t *             out_msg_len ) {
  if( UNLIKELY( pk[ 0 ] != PK_HEADER ) )            return -1;
  if( UNLIKELY( smlen < 2 + NONCELEN + 1 ) )        return -1;

  size_t sig_field_len = ( (size_t)sm[ 0 ] << 8 ) | (size_t)sm[ 1 ];
  if( UNLIKELY( sig_field_len < 1 ) )                       return -1;
  if( UNLIKELY( sig_field_len > smlen - 2 - NONCELEN ) )    return -1;

  size_t          msg_len = smlen - 2 - NONCELEN - sig_field_len;
  uint8_t const * esig    = sm + 2 + NONCELEN + msg_len;
  if( UNLIKELY( esig[ 0 ] != SIG_HEADER ) )                 return -1;

  if( UNLIKELY( fa512_parse_pk(      pubk,    pk + 1                  ) ) ) return -1;
  if( UNLIKELY( fa512_parse_comp_s2( sig->s2, esig + 1, sig_field_len-1 ) ) ) return -1;

  *out_nonce   = sm + 2;
  *out_msg     = sm + 2 + NONCELEN;
  *out_msg_len = msg_len;
  return 0;
}

int
falcon_avx512_finish( falcon_fq_t        const * c,
                      falcon_pubkey_t    const * pubk,
                      falcon_signature_t const * sig ) {
  falcon_fq_t s2_ntt[ N ] __attribute__((aligned(64)));
  falcon_fq_t h_ntt [ N ] __attribute__((aligned(64)));
  falcon_fq_t prod  [ N ] __attribute__((aligned(64)));
  ntt_fwd_avx512( s2_ntt, sig->s2 );  /* both lazy in [0,2q) */
  ntt_fwd_avx512( h_ntt,  pubk->h  );
  /* Hadamard uses one-shot Barrett; Shoup precompute is not reused here.
   * The product is [0,2q) x [0,2q) < 4*q^2 < 2^30, within Barrett's
   * input bound.  The output stays lazy in [0,2q) and is absorbed by
   * the inverse NTT's reduction schedule (off_q starts at 2). */
  for( int i=0; i<N; i+=16 ) {
    __m512i a = _mm512_loadu_si512( (void const *)( s2_ntt + i ) );
    __m512i b = _mm512_loadu_si512( (void const *)( h_ntt  + i ) );
    __m512i Mv  = _mm512_set1_epi32( (int)43687U );
    __m512i Qv  = _mm512_set1_epi32( Q );
    __m512i mask_e = _mm512_set1_epi64( 0xFFFFFFFFLL );
    __m512i product = _mm512_mullo_epi32( a, b );
    __m512i wide_e = _mm512_mul_epu32( product, Mv );
    __m512i qest_e = _mm512_srli_epi64( wide_e, 29 );
    __m512i prod_o = _mm512_srli_epi64( product, 32 );
    __m512i wide_o = _mm512_mul_epu32( prod_o, Mv );
    __m512i qest_o = _mm512_srli_epi64( wide_o, 29 );
    __m512i qest_o_s = _mm512_slli_epi64( qest_o, 32 );
    __m512i qest = _mm512_or_si512( _mm512_and_si512( qest_e, mask_e ), qest_o_s );
    __m512i r = _mm512_sub_epi32( product, _mm512_mullo_epi32( qest, Qv ) );
    _mm512_storeu_si512( (void *)( prod + i ), r );
  }

  falcon_fq_t pmm[ N ] __attribute__((aligned(64)));
  ntt_inv_avx512( pmm, prod );

  return fa512_norm_check_ok( c, pmm, sig );
}

int
falcon_avx512_crypto_sign_open( uint8_t       * m,  size_t * mlen,
                                 uint8_t const * sm, size_t   smlen,
                                 uint8_t const * pk ) {
  falcon_pubkey_t    pubk[1];
  falcon_signature_t sig [1];
  uint8_t const *    nonce;
  uint8_t const *    msg;
  size_t             msg_len;
  if( UNLIKELY( falcon_avx512_parse( sm, smlen, pk, pubk, sig,
                                      &nonce, &msg, &msg_len ) ) ) return -1;

  falcon_fq_t c[ N + 16 ] __attribute__((aligned(64)));
  fa512_hash_to_point( c, nonce, msg, msg_len );

  if( !falcon_avx512_finish( c, pubk, sig ) ) return -1;

  if( m && msg_len ) memmove( m, msg, msg_len );
  if( mlen ) *mlen = msg_len;
  return 0;
}

/* Shoup AVX-512 with non-standard KTP256 hash-to-point. */
int
falcon_avx512_ktp256_crypto_sign_open( uint8_t       * m,  size_t * mlen,
                                          uint8_t const * sm, size_t   smlen,
                                          uint8_t const * pk ) {
  falcon_pubkey_t    pubk[1];
  falcon_signature_t sig [1];
  uint8_t const *    nonce;
  uint8_t const *    msg;
  size_t             msg_len;
  if( UNLIKELY( falcon_avx512_parse( sm, smlen, pk, pubk, sig,
                                      &nonce, &msg, &msg_len ) ) ) return -1;

  /* KTP writes u16 coefficients; the NTT consumes u32. */
  uint16_t    c16[ N + 32 ] __attribute__((aligned(64)));
  falcon_fq_t c  [ N + 16 ] __attribute__((aligned(64)));
  fa512_hash_to_point_ktp256( c16, nonce, NONCELEN + msg_len );
  for( int i=0; i<N; i+=32 ) {
    __m512i lo = _mm512_cvtepu16_epi32( _mm256_loadu_si256( (__m256i const *)( c16 + i      ) ) );
    __m512i hi = _mm512_cvtepu16_epi32( _mm256_loadu_si256( (__m256i const *)( c16 + i + 16 ) ) );
    _mm512_storeu_si512( (void *)( c + i      ), lo );
    _mm512_storeu_si512( (void *)( c + i + 16 ), hi );
  }

  int ok = falcon_avx512_finish( c, pubk, sig );

  if( m && msg_len ) memmove( m, msg, msg_len );
  if( mlen ) *mlen = msg_len;
  return ok ? 0 : -1;
}

/* Bench helper: time finish only. */

int
falcon_avx512_bench_mul( falcon_fq_t        const * c,
                          falcon_pubkey_t    const * pubk,
                          falcon_signature_t const * sig ) {
  return falcon_avx512_finish( c, pubk, sig );
}

#else /* !HAVE_AVX512 */

int
falcon_avx512_crypto_sign_open( uint8_t       * m,  size_t * mlen,
                                 uint8_t const * sm, size_t   smlen,
                                 uint8_t const * pk ) {
  return falcon_ref_xkcp_crypto_sign_open( m, mlen, sm, smlen, pk );
}

int
falcon_avx512_ktp256_crypto_sign_open( uint8_t       * m,  size_t * mlen,
                                          uint8_t const * sm, size_t   smlen,
                                          uint8_t const * pk ) {
  return falcon_ref_xkcp_crypto_sign_open( m, mlen, sm, smlen, pk );
}

/* Link-only bench stub for non-AVX-512 builds. */
int  falcon_avx512_bench_mul ( void const * c, void const * pubk, void const * sig ) {
  (void)c; (void)pubk; (void)sig; return 0;
}

#endif /* HAVE_AVX512 */
