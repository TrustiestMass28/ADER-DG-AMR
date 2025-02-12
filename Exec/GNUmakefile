PRECISION  = DOUBLE
PROFILE    = FALSE
DEBUG      = FALSE
DIM        = 2
COMP	   = gnu
USE_MPI    = TRUE
AMREX_USE_OMP = FALSE
USE_OMP    = FALSE
#MPI_THREAD_MULTIPLE = TRUE
#USE_OMP_OFFLOAD=TRUE
USE_CUDA   = FALSE
WARN_ALL   = TRUE

AMREX_HOME ?= ../Library/amrex

include $(AMREX_HOME)/Tools/GNUMake/Make.defs
include ../Source/Make.package
VPATH_LOCATIONS  += ../Source
INCLUDE_LOCATIONS += ../Source

include $(AMREX_HOME)/Src/Base/Make.package
include $(AMREX_HOME)/Src/AmrCore/Make.package
include $(AMREX_HOME)/Src/Amr/Make.package
include $(AMREX_HOME)/Src/Boundary/Make.package

# Add the path to the Eigen header files to the include directories
EIGEN_HOME ?= ../Library/eigen 

INCLUDE_DIRS += $(EIGEN_HOME)/Eigen

LIB_DIRS += $(EIGEN_HOME)/build
LIBS += -leigen



all: $(executable) 
	@echo SUCCESS
	
include $(AMREX_HOME)/Tools/GNUMake/Make.rules




