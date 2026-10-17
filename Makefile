# compiler
CC = g++

# C++ compiler option
CXXFLAGS = -Wall -O2
CXXFLAGS_DBG = -Wall -fno-inline-functions -DDEBUG -g

# 编译标志和链接标志分开
PKG_CFLAGS = -I/usr/local/include `pkg-config --cflags icu-uc icu-i18n re2`
PKG_LIBS   = -L/usr/local/lib -Wl,-rpath,/usr/local/lib `pkg-config --libs icu-uc icu-i18n re2`

PKG_CFLAGS_DBG = -I/usr/local/include `pkg-config --cflags icu-uc icu-i18n re2`
PKG_LIBS_DBG   = -L/usr/local/lib -Wl,-rpath,/usr/local/lib `pkg-config --libs icu-uc icu-i18n re2`

# source directory
SRC_DIR = ./cpp

# object directory
OBJ_DIR = ./obj
OBJ_DIR_DBG = ./obj/debug

# exe file name
TARGET = main
TARGET_DBG = main_debug

# source files to make
SRCS = $(notdir $(wildcard $(SRC_DIR)/*.cpp))
OBJS = $(SRCS:.cpp=.o)
OBJECTS = $(patsubst %.o,$(OBJ_DIR)/%.o,$(OBJS))
DEPS = $(OBJECTS:.o=.d)
OBJECTS_DBG = $(patsubst %.o,$(OBJ_DIR_DBG)/%.o,$(OBJS))
DEPS_DBG = $(OBJECTS_DBG:.o=.d)

all: $(TARGET)

# 编译时只使用 CFLAGS
$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cpp
	@mkdir -p $(@D)
	$(CC) $(CXXFLAGS) $(PKG_CFLAGS) -c $< -o $@ -MD

# 链接时使用 LIBS
$(TARGET) : $(OBJECTS)
	$(CC) $(CXXFLAGS) $(OBJECTS) -o $(TARGET) $(PKG_LIBS)

debug: $(TARGET_DBG)

$(OBJ_DIR_DBG)/%.o : $(SRC_DIR)/%.cpp
	@mkdir -p $(@D)
	$(CC) $(CXXFLAGS_DBG) $(PKG_CFLAGS_DBG) -c $< -o $@ -MD

$(TARGET_DBG) : $(OBJECTS_DBG)
	$(CC) $(CXXFLAGS_DBG) $(OBJECTS_DBG) -o $(TARGET_DBG) $(PKG_LIBS_DBG)

.PHONY: clean all

clean:
	rm -f $(OBJECTS) $(DEPS) $(TARGET) $(OBJECTS_DBG) $(DEPS_DBG) $(TARGET_DBG)

-include $(DEPS)