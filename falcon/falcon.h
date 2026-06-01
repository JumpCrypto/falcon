/* Falcon-512 verification variants.  All expose the NIST open API and
 * consume the same signed-message/public-key buffers.  `_ktp256`
 * variants replace SHAKE256 hash-to-point with a non-standard
 * TurboSHAKE12 8-way squeeze, so their challenge polynomial differs
 * from Falcon round 3. */

#ifndef CONTRIB_FALCON_FALCON_H
#define CONTRIB_FALCON_FALCON_H

#include <stddef.h>
#include <stdint.h>

#define FALCON_N            512
#define FALCON_LOGN         9
#define FALCON_Q            12289
#define FALCON_BETA2        34034726L

#define FALCON_PUBKEY_SIZE  ( 1 + ( 14 * FALCON_N / 8 ) ) /* 897 */
#define FALCON_SIG_MAX      690                            /* round 3 max */

#ifdef __cplusplus
extern "C" {
#endif

/* NIST open API.  `sm` is
 * [sig_len:2 BE | nonce:40 | message | esig: 0x29 || comp_s2];
 * `pk` is [0x09 | 14-bit-packed h[512]]. */

/* Round 3 reference: Pornin pipeline and SHAKE. */
int falcon_ref_crypto_sign_open(                  uint8_t       * m, size_t * mlen,
                                                  uint8_t const * sm, size_t   smlen,
                                                  uint8_t const * pk );

/* Pornin pipeline + XKCP SHAKE256; bench baseline. */
int falcon_ref_xkcp_crypto_sign_open(             uint8_t       * m, size_t * mlen,
                                                  uint8_t const * sm, size_t   smlen,
                                                  uint8_t const * pk );

/* Pornin pipeline + non-standard parallel-squeeze hash. */
int falcon_ref_ktp256_crypto_sign_open(         uint8_t       * m, size_t * mlen,
                                                  uint8_t const * sm, size_t   smlen,
                                                  uint8_t const * pk );

/* Auto-vectorisable C: Shoup NTT, no intrinsics. */
int falcon_x86_crypto_sign_open(                  uint8_t       * m, size_t * mlen,
                                                  uint8_t const * sm, size_t   smlen,
                                                  uint8_t const * pk );

int falcon_x86_ktp256_crypto_sign_open(         uint8_t       * m, size_t * mlen,
                                                  uint8_t const * sm, size_t   smlen,
                                                  uint8_t const * pk );

/* AVX-512 variants: Barrett, Shoup, and Pornin Montgomery. */
int falcon_avx512_barrett_crypto_sign_open(       uint8_t       * m, size_t * mlen,
                                                  uint8_t const * sm, size_t   smlen,
                                                  uint8_t const * pk );

int falcon_avx512_crypto_sign_open(               uint8_t       * m, size_t * mlen,
                                                  uint8_t const * sm, size_t   smlen,
                                                  uint8_t const * pk );

int falcon_avx512_from_ref_crypto_sign_open(      uint8_t       * m, size_t * mlen,
                                                  uint8_t const * sm, size_t   smlen,
                                                  uint8_t const * pk );

/* Barrett AVX-512 with always-reduced add/sub. */
int falcon_avx512_barrett_alwaysred_crypto_sign_open(
                                                  uint8_t       * m, size_t * mlen,
                                                  uint8_t const * sm, size_t   smlen,
                                                  uint8_t const * pk );

/* AVX-512 + KTP256 hash combinations. */
int falcon_avx512_barrett_ktp256_crypto_sign_open( uint8_t       * m, size_t * mlen,
                                                   uint8_t const * sm, size_t   smlen,
                                                   uint8_t const * pk );

int falcon_avx512_ktp256_crypto_sign_open(        uint8_t       * m, size_t * mlen,
                                                  uint8_t const * sm, size_t   smlen,
                                                  uint8_t const * pk );

#ifdef __cplusplus
}
#endif

#endif /* CONTRIB_FALCON_FALCON_H */
