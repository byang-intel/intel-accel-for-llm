DEVICE   ?= cuda


MODULES  := qat_zip cpu_zip
ifeq ($(DEVICE),cuda)
MODULES  += dsa_gd
endif


INCDIR   := include


LIBDIR   := ../../_lib
QATBUILD := ../../_build/qat
QATURL   := https://downloadmirror.intel.com/921047/QAT20.L.1.2.30-00178.tar.gz
QATLIBS  := $(LIBDIR)/libqat_s.so $(LIBDIR)/libusdm_drv_s.so
RPATH    := -Wl,-rpath,$(abspath $(LIBDIR)) -Wl,-rpath,$(abspath .)

CC       := gcc
CXX      := g++
CFLAGS   := -O2 -g -Wall -I$(INCDIR)
LDFLAGS  := -lpthread


ARGS     ?=


qat_zip_SRCS   := qat_zip/qat_zip.c common/envs.c
qat_zip_LDLIBS := $(QATLIBS)
qat_zip_DEPS   := $(QATLIBS)


cpu_zip_SRCS   := cpu_zip/cpu_zip.c common/envs.c
cpu_zip_LDLIBS := -lz


dsa_gd_SRCS   := dsa_gd/dsa_gd.c dsa_gd/dsa_memcpy.c common/envs.c
dsa_gd_CFLAGS := -std=gnu11 -D_GNU_SOURCE -fopenmp
dsa_gd_LDLIBS := -lgdrapi -ldl -fopenmp


test_src = $(strip $(wildcard test/$(1)_test.c) $(wildcard test/$(1)_test.cpp))


LIBS         := $(foreach m,$(MODULES),$(LIBDIR)/lib$(m).so)
TESTS        := $(foreach m,$(MODULES),$(if $(call test_src,$(m)),$(m)_test))
TEST_TARGETS := $(foreach m,$(MODULES),$(if $(call test_src,$(m)),test-$(m)))
TESTS        += zip_compat_test
TEST_TARGETS += test-zip-compat

.PHONY: all test clean qat-userspace $(TEST_TARGETS)

all: $(LIBS) $(TESTS)


$(LIBDIR)/libusdm_drv_s.so: $(LIBDIR)/libqat_s.so

$(LIBDIR)/libqat_s.so:
	mkdir -p $(QATBUILD)
	mkdir -p $(LIBDIR)
	cd $(QATBUILD) \
	    && wget -q "$(QATURL)" \
	    && tar xf *.tar.gz \
	    && export KERNEL_SOURCE_ROOT=$$(ls -dv /usr/src/linux-headers-*-generic | tail -n 1) \
	    && ./configure \
	    && sed "s/^ALL_TARGETS\ *=\ *.*/ALL_TARGETS = lac_user/g" -i quickassist/Makefile \
	    && make -j"$$(nproc)" quickassist-all
	cp -a $(QATBUILD)/build/*.so $(LIBDIR)/


define MODULE_template
$(1)_SRCS := $$(or $$($(1)_SRCS),$$(wildcard $(1)/*.c) $$(wildcard $(1)/*.cpp))
$(1)_CC   := $$(or $$($(1)_CC),$$(CC))
$(1)_TEST_SRC := $$(strip $$(wildcard test/$(1)_test.c) $$(wildcard test/$(1)_test.cpp))

$$(LIBDIR)/lib$(1).so: $$($(1)_SRCS) $$(wildcard $(1)/include/*.h) $$(wildcard $(1)/*.h) $$(wildcard $$(INCDIR)/*.h) $$($(1)_DEPS)
	mkdir -p $$(LIBDIR)
	$$($(1)_CC) $$(CFLAGS) $$($(1)_CFLAGS) -I$(1) -I$(1)/include -fPIC -shared -o $$@ $$($(1)_SRCS) $$($(1)_LDLIBS) $$(RPATH) $$(LDFLAGS)

ifneq ($$($(1)_TEST_SRC),)
$(1)_test: $$($(1)_TEST_SRC) $$(LIBDIR)/lib$(1).so
	$$($(1)_CC) $$(CFLAGS) $$($(1)_CFLAGS) -fopenmp -o $$@ $$< -L$$(LIBDIR) -l$(1) $$($(1)_LDLIBS) $$(RPATH) $$(LDFLAGS)

test-$(1): $(1)_test
	(cd test && numactl --cpunodebind=0 --membind=0 ../$(1)_test $$(ARGS))
endif
endef

$(foreach m,$(MODULES),$(eval $(call MODULE_template,$(m))))


zip_compat_test: test/zip_compat_test.c $(LIBDIR)/libqat_zip.so $(LIBDIR)/libcpu_zip.so
	$(CC) $(CFLAGS) -o $@ $< -L$(LIBDIR) -lqat_zip -lcpu_zip -lz $(RPATH) $(LDFLAGS)

test-zip-compat: zip_compat_test
	(cd test && numactl --cpunodebind=0 --membind=0 ../zip_compat_test $(ARGS))


qat-userspace: $(QATLIBS)


test: $(TEST_TARGETS)

clean:
	rm -f $(LIBS) $(TESTS)
