#---------------AMREX---------------#
PRECISION  = DOUBLE
PROFILE    = FALSE
DEBUG      = FALSE
DIM        = 2
COMP	   = gnu
USE_MPI    = TRUE
#MPI_THREAD_MULTIPLE = TRUE
AMREX_USE_OMP = FALSE
USE_OMP    = FALSE
#USE_OMP_OFFLOAD=TRUE
USE_CUDA   = FALSE
WARN_ALL   = TRUE

AMREX_HOME ?= ../Library/amrex

include $(AMREX_HOME)/Tools/GNUMake/Make.defs
include $(AMREX_HOME)/Src/Base/Make.package
include $(AMREX_HOME)/Src/AmrCore/Make.package
include $(AMREX_HOME)/Src/Amr/Make.package
include $(AMREX_HOME)/Src/Boundary/Make.package

#---------------EIGEN---------------#
$(info Current directory: $(shell pwd))

# Set EIGEN_HOME to point to the  location of Eigen
EIGEN_HOME ?= ../Library/eigen-3.4.0

# Add the path to the Eigen header files to the include directories
CXXFLAGS += -I$(EIGEN_HOME)#/Eigen #if uncomment this, then import using <Dense> and not <Eigen/Dense>

$(info EIGEN_HOME = $(EIGEN_HOME))
$(info CXXFLAGS = $(CXXFLAGS))

INCLUDE_DIRS += $(EIGEN_HOME)/Eigen
$(info INCLUDE_DIRS = $(INCLUDE_DIRS))

LIB_DIRS += $(EIGEN_HOME)/build
LIBS += -leigen

#---------------INDICATORS---------------#
# Set INDICATORS_HOME to point to the location of the indicators library
INDICATORS_HOME ?= ../Library/indicators

# Add the path to the indicators header files to the include directories
# The headers are in the 'include' subdirectory of the repository
CXXFLAGS += -I$(INDICATORS_HOME)/include

# Add -pthread for std::thread support, which indicators may use
LIBS += -lpthread

# --- Final Setup ---
include ../Source/Make.package
VPATH_LOCATIONS  += ../Source
INCLUDE_LOCATIONS += ../Source

all: $(executable) 
	@echo SUCCESS
	
include $(AMREX_HOME)/Tools/GNUMake/Make.rules

