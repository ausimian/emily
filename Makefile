PRIV_DIR       := $(MIX_APP_PATH)/priv
NIF_SO         := $(PRIV_DIR)/libemily.so
METALLIB       := $(PRIV_DIR)/mlx.metallib

BUILD_DIR := $(MIX_APP_PATH)/obj

# Sources — include ops/* and any other subdirs under c_src.
SOURCES := $(shell find c_src -name '*.cpp')
HEADERS := $(shell find c_src \( -name '*.h' -o -name '*.hpp' \))
OBJECTS := $(patsubst c_src/%.cpp,$(BUILD_DIR)/%.o,$(SOURCES))

# Flags
CXXFLAGS := -std=c++17 -O3 -fPIC -fvisibility=hidden -Wall -Wextra
CXXFLAGS += -I$(ERTS_INCLUDE_DIR) -Ic_src
# Third-party headers: use -isystem so warnings inside them (e.g. MLX's
# -Wdeprecated-copy on _MLX_BFloat16) don't clutter our builds or trip
# -Werror.
CXXFLAGS += -isystem $(FINE_INCLUDE_DIR) -isystem $(MLX_INCLUDE_DIR)

# Static link: embed libmlx.a into the NIF and link system frameworks
# that MLX depends on directly (previously resolved transitively through
# the dylib).
LDFLAGS  := -shared
LDFLAGS  += $(MLX_LIB_DIR)/libmlx.a

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    LDFLAGS += -undefined dynamic_lookup -flat_namespace
    LDFLAGS += -framework Metal -framework Foundation -framework Accelerate
endif

# Optional: AddressSanitizer build. Set EMILY_ASAN=1 to instrument the
# NIF. Requires an OTP built with --enable-sanitizers=address so that
# beam.smp links the ASan runtime at startup (interceptors must install
# before any allocation). macOS SIP strips DYLD_INSERT_LIBRARIES from
# processes launched through /bin/sh, and loading libasan late via
# dlopen fails, so preloading is not an option on stock macOS+OTP.
ifeq ($(EMILY_ASAN),1)
    CXXFLAGS += -fsanitize=address -fno-omit-frame-pointer -g -O1
    LDFLAGS  += -fsanitize=address
endif

.PHONY: all clean bench-native

all: $(NIF_SO) $(METALLIB)

# ------------------------------------------------------------------
# bench-native: standalone C++ microbenchmarks under bench/native/.
#
# Invoked by `mix bench.native` (which sets the same env elixir_make
# uses for the NIF build). Links against the vendored libmlx via the
# same rpath the NIF does, so the binary finds its own shared library
# without relying on global DYLD/LD paths.
# ------------------------------------------------------------------

BENCH_NATIVE_SRC := bench/native/compile_microbench.cpp
BENCH_NATIVE_BIN := $(BUILD_DIR)/compile_microbench
BENCH_NATIVE_METALLIB := $(BUILD_DIR)/mlx.metallib

$(BENCH_NATIVE_BIN): $(BENCH_NATIVE_SRC) | $(BUILD_DIR)
	$(CXX) -std=c++17 -O3 -Wall -Wextra \
	    -isystem $(MLX_INCLUDE_DIR) \
	    $(BENCH_NATIVE_SRC) \
	    $(MLX_LIB_DIR)/libmlx.a \
	    -framework Metal -framework Foundation -framework Accelerate \
	    -o $(BENCH_NATIVE_BIN)

# MLX's Metal device loader looks for `mlx.metallib` colocated with the
# running binary. The NIF build stages it under priv/; the standalone
# bench binary lives in BUILD_DIR, so stage a sibling copy there too.
$(BENCH_NATIVE_METALLIB): $(MLX_LIB_DIR)/mlx.metallib | $(BUILD_DIR)
	cp $< $@

bench-native: $(BENCH_NATIVE_BIN) $(BENCH_NATIVE_METALLIB)
	@echo "Running $(BENCH_NATIVE_BIN)"
	@$(BENCH_NATIVE_BIN) $(BENCH_NATIVE_ARGS)

$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

$(PRIV_DIR):
	@mkdir -p $(PRIV_DIR)

$(BUILD_DIR)/%.o: c_src/%.cpp $(HEADERS) | $(BUILD_DIR)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(NIF_SO): $(OBJECTS) | $(PRIV_DIR)
	$(CXX) $(OBJECTS) -o $(NIF_SO) $(LDFLAGS)

# MLX searches for mlx.metallib colocated with the loaded binary
# (see vendor/mlx/mlx/backend/metal/device.cpp:load_default_library).
# Stage the compiled shader library next to the NIF .so in priv/.
$(METALLIB): $(MLX_LIB_DIR)/mlx.metallib | $(PRIV_DIR)
	cp $< $@

clean:
	rm -rf $(BUILD_DIR) $(NIF_SO) $(METALLIB)
