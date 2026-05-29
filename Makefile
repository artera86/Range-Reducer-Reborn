CXX       ?= g++
CXXFLAGS  ?= -std=c++17 -O2 -Wall -Wextra

TEST_SRC  = tests/test_host_functions.cpp
TEST_BIN  = tests/test_host_functions

.PHONY: test clean

test: $(TEST_BIN)
	./$(TEST_BIN)

$(TEST_BIN): $(TEST_SRC)
	$(CXX) $(CXXFLAGS) -o $@ $<

clean:
	rm -f $(TEST_BIN)
