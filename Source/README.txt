Compiling and running the code
------------------------------
-go to /Exec folder
-open GNUmakefile and set desired flags for (DIM,USE_MPI)
-open terminal inside /Exec folder
-compile:	make
-run:		./main2d.gnu.ex	
 		mpiexec -n 4 ./main2d.gnu.MPI.ex 
 		
