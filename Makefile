PRIV_DIR       := $(MIX_APP_PATH)/priv
NIF_SO         := $(PRIV_DIR)/libemily.so
MLX_STAGE_DIR  := $(PRIV_DIR)/mlx/lib

BUILD_DIR := $(EMILY_CACHE_DIR)/build-$(EMILY_VERSION)

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

LDFLAGS  := -L$(MLX_LIB_DIR) -lmlx -shared

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    LDFLAGS += -undefined dynamic_lookup -flat_namespace -rpath @loader_path/mlx/lib
    JOBS    := $(shell sysctl -n hw.ncpu)
else
    LDFLAGS += -Wl,-rpath,'$$ORIGIN/mlx/lib'
    JOBS    := $(shell nproc)
endif

MAKE_JOBS ?= $(JOBS)

.PHONY: all clean

all: $(NIF_SO)

$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

$(PRIV_DIR):
	@mkdir -p $(PRIV_DIR)

$(BUILD_DIR)/%.o: c_src/%.cpp $(HEADERS) | $(BUILD_DIR)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(MLX_STAGE_DIR): | $(PRIV_DIR)
	@mkdir -p $(MLX_STAGE_DIR)
	@cp -a $(MLX_LIB_DIR)/. $(MLX_STAGE_DIR)/

$(NIF_SO): $(OBJECTS) $(MLX_STAGE_DIR) | $(PRIV_DIR)
	$(CXX) $(OBJECTS) -o $(NIF_SO) $(LDFLAGS)

clean:
	rm -rf $(BUILD_DIR) $(NIF_SO) $(PRIV_DIR)/mlx
