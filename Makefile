# Falcon-512 verifier benchmark harness.
# `make test` builds and runs correctness tests.
# `NATIVE=0` disables -march=native and AVX-512 dispatch.

CC      ?= gcc
NATIVE  ?= 1

# Local verifier sources.
FALCON_DIR = falcon

CFLAGS_LOCAL  ?= -O3 -Wall -Wextra -Wno-unused-parameter -fno-strict-aliasing -std=gnu17 -I$(FALCON_DIR)
CFLAGS_VENDOR ?= -O3 -w
ifeq ($(NATIVE),1)
CFLAGS_LOCAL  += -march=native
CFLAGS_VENDOR += -march=native
endif

# Local verifier source list.
SRC_LOCAL = $(FALCON_DIR)/falcon_ref.c $(FALCON_DIR)/falcon_ref_xkcp.c \
            $(FALCON_DIR)/falcon_ref_ktp256.c $(FALCON_DIR)/falcon_x86.c \
            $(FALCON_DIR)/falcon_avx512_barrett.c $(FALCON_DIR)/falcon_avx512_barrett_alwaysred.c \
            $(FALCON_DIR)/falcon_avx512.c $(FALCON_DIR)/falcon_avx512_from_ref.c \
            $(FALCON_DIR)/randombytes_stub.c
HDR_LOCAL = $(FALCON_DIR)/falcon.h $(FALCON_DIR)/falcon_twiddle.h \
            $(FALCON_DIR)/falcon_avx512_common.h test_vectors.h

OBJ_LOCAL    = $(SRC_LOCAL:.c=.o)

# Pornin NIST round 3 reference.
PORNIN_DIR = vendor/falcon-round3/Reference_Implementation/falcon512/falcon512int
PORNIN_SRC = $(wildcard $(PORNIN_DIR)/*.c)
PORNIN_OBJ = $(PORNIN_SRC:.c=.o)

# XKCP permutation objects used by SHAKE benches.
XKCP_DIR    = vendor/xkcp
XKCP_LIB_P  = $(XKCP_DIR)/bin/generic64/libXKCP.a
XKCP_LIB_A  = $(XKCP_DIR)/bin/AVX512/libXKCP.a
XKCP_OBJ_P  = $(XKCP_DIR)/bin/generic64/KeccakP-1600-opt64.o
XKCP_OBJ_A  = $(XKCP_DIR)/bin/AVX512/KeccakP-1600-AVX512.o
XKCP_OBJ_P8 = $(XKCP_DIR)/bin/AVX512/KeccakP-1600-times8-AVX512.o

OBJ_ALL = $(OBJ_LOCAL) $(PORNIN_OBJ) $(XKCP_OBJ_P) $(XKCP_OBJ_A) $(XKCP_OBJ_P8)

.PHONY: all bench test clean xkcp

all: bench test_falcon

bench: bench.o $(OBJ_ALL)
	$(CC) $(CFLAGS_LOCAL) -o $@ $^

test_falcon: test_falcon.o $(OBJ_ALL)
	$(CC) $(CFLAGS_LOCAL) -o $@ $^

# Local sources: warnings on.
bench.o test_falcon.o $(OBJ_LOCAL): %.o: %.c $(HDR_LOCAL)
	$(CC) $(CFLAGS_LOCAL) -c -o $@ $<

# Vendored reference: warnings off.
$(PORNIN_OBJ): %.o: %.c
	$(CC) $(CFLAGS_VENDOR) -I $(PORNIN_DIR) -c -o $@ $<

# Build XKCP libraries, then extract needed permutation objects.
xkcp: $(XKCP_LIB_P) $(XKCP_LIB_A)

$(XKCP_LIB_P):
	$(MAKE) -C $(XKCP_DIR) generic64/libXKCP.a EXTRA_CFLAGS="$(if $(filter 1,$(NATIVE)),-march=native -mtune=native,)"

$(XKCP_LIB_A):
	$(MAKE) -C $(XKCP_DIR) AVX512/libXKCP.a EXTRA_CFLAGS="$(if $(filter 1,$(NATIVE)),-march=native -mtune=native,)"

$(XKCP_OBJ_P): $(XKCP_LIB_P)
	cd $(XKCP_DIR)/bin/generic64 && ar x libXKCP.a KeccakP-1600-opt64.o

$(XKCP_OBJ_A): $(XKCP_LIB_A)
	cd $(XKCP_DIR)/bin/AVX512 && ar x libXKCP.a KeccakP-1600-AVX512.o

$(XKCP_OBJ_P8): $(XKCP_LIB_A)
	cd $(XKCP_DIR)/bin/AVX512 && ar x libXKCP.a KeccakP-1600-times8-AVX512.o

test: test_falcon
	./test_falcon

clean:
	rm -f *.o $(FALCON_DIR)/*.o $(PORNIN_OBJ) bench test_falcon
	rm -rf $(XKCP_DIR)/bin
