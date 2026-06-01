# Falcon Verify on AVX-512

AVX-512 implementation of Falcon signature verification.

This work was developed as part of the
[Firedancer](https://github.com/firedancer-io/firedancer) project, and is described in the paper:\
**Falcon Verify on AVX-512: Speed Records** [2026/??](https://eprint.iacr.org/2026/??).

This repository is self-contained so the benchmarks in the paper can be reproduced without the rest of the Firedancer tree. The fastest, standard, AVX-512 verifier is in [`falcon/falcon_avx512.c`](falcon/falcon_avx512.c).

## Build

```
make            # builds bench and test_falcon
make test       # runs the correctness tests
./bench         # runs the benchmark
```

Default toolchain is `gcc`; pass `CC=clang` to override. The build
assumes `-march=native`. See `bench --help` for options.

## Citation

```bibtex
@misc{falcon-verify-avx512,
  author       = {David Rubin and Emanuele Cesena},
  title        = {Falcon Verify on {AVX-512}: Speed Records},
  howpublished = {Cryptology {ePrint} Archive, Report 2026/??},
  year         = {2026},
  url          = {https://eprint.iacr.org/2026/??}
}
```

## License

Apache 2.0; see [`LICENSE`](LICENSE). The vendored sources in
`vendor/` keep their upstream licenses.

**Disclaimer. This alpha software has been open sourced. All software and code are provided "as is," without any warranty of any kind, and should be used at your own risk.**
