INCLUDES= -I ../lbfgs -I/usr/local/cuda-5.5/include/

LIBPATH=-L/usr/local/cuda-5.5/lib64

LIBS= -lcublas  -lcudart 

# the compiler to use.
CC=g++

make:	
	${CC} sample.cpp ../lbfgs/lbfgs.c $(INCLUDES) $(LIBPATH) $(LIBS) -o program
	

clean:
	rm -f program






