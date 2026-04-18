CXX = g++
CXXFLAGS = -std=c++17 -O3 -march=native -Wall -Wextra -fopenmp
LDFLAGS = -fopenmp

SRC_DIR = src
OBJ_DIR = obj
BIN_DIR = bin

SOURCES = $(wildcard $(SRC_DIR)/core/*.cpp) $(wildcard $(SRC_DIR)/tests/*.cpp) $(SRC_DIR)/main.cpp
OBJECTS = $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(SOURCES))
TARGET = $(BIN_DIR)/cb

.PHONY: all clean dirs

all: dirs $(TARGET)

dirs:
	@mkdir -p $(OBJ_DIR)/core $(OBJ_DIR)/tests $(BIN_DIR) results reports

$(TARGET): $(OBJECTS)
	$(CXX) $(LDFLAGS) -o $@ $^

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)

# 运行测试
run: all
	./$(TARGET)

# 分析结果
analyze:
	python3 scripts/analyze.py results/latest.json
