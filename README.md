#---------------SETUP---------------#
Folder: /ADER-DG-AMR/Library/

Eigen 3.4.0
	wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip
	unzip eigen-3.4.0.zip
	rm eigen-3.4.0.zip
AMREX
	git clone https://github.com/AMReX-Codes/amrex.git


#---------------COMPILE---------------#
Folder: /ADER-DG-AMR/Exec/

open GNUmakefile and set desired flags for (DIM,USE_MPI,...)

normal		:	make
parallel	:	make -j NCPU

#---------------RUN---------------#
Folder: /ADER-DG-AMR/Exec/

./main2d.gnu.ex	
mpiexec -n 4 ./main2d.gnu.MPI.ex 