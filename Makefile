# Makefile for executable scc
# *****************************************************
# Parameters to control Makefile operation

CC = nvcc
CFLAGS = -arch=compute_20 -code=sm_20 -O3 -w -rdc=true -lineinfo
CUB_DIR=$(HOME)/cub-1.1.1
# ****************************************************
# Entries to bring the executable up to date

all: scc scc_w scc_2_w

scc: scc.cu scc.h load.o tarjan.o timing.o scc_routines.o reach_q.o reach_t.o reach_l.o reach_d.o reach_b.o
	$(CC) $(CFLAGS) -o scc scc.cu load.o tarjan.o timing.o scc_routines.o reach_q.o reach_t.o reach_b.o reach_l.o reach_d.o -I $(CUB_DIR)

scc_w: scc.cu scc.h load.o tarjan.o timing.o scc_routines.o reach_q.o reach_t.o reach_l.o reach_d.o wcc.o reach_b.o
	$(CC) $(CFLAGS) -DWCC -o scc_w scc.cu load.o tarjan.o timing.o scc_routines.o reach_q.o reach_t.o reach_b.o reach_l.o reach_d.o wcc.o -I $(CUB_DIR)

scc_2_w: scc.cu scc.h load.o tarjan.o timing.o scc_routines.o reach_q.o reach_t.o reach_l.o reach_d.o wcc.o reach_b.o
	$(CC) $(CFLAGS) -DWCC -DTRIM_2 -o scc_2_w scc.cu load.o tarjan.o timing.o scc_routines.o reach_q.o reach_t.o reach_b.o reach_l.o reach_d.o wcc.o -I $(CUB_DIR)

scc.o: scc.cu scc.h load.h tarjan.h GPUerrchk.h timing.h scc_routines.h reach_q.h reach_l.h reach_d.h worklistc.h	
	$(CC) $(CFLAGS) -c scc.cu -I $(CUB_DIR)

load.o: load.cpp load.h
	$(CC) $(CFLAGS) -c load.cpp
	
timing.o: timing.cpp timing.h
	$(CC) $(CFLAGS) -c timing.cpp

tarjan.o: tarjan.cpp tarjan.h timing.h
	$(CC) $(CFLAGS) -c tarjan.cpp
	
scc_routines.o: scc_routines.cu scc_routines.h GPUerrchk.h worklistc.h
	$(CC) $(CFLAGS) -c scc_routines.cu -I $(CUB_DIR)

reach_q.o: reach_q.cu reach_q.h GPUerrchk.h
	$(CC) $(CFLAGS) -c reach_q.cu

reach_b.o: reach_b.cu reach_b.h GPUerrchk.h
	$(CC) $(CFLAGS) -c reach_b.cu -I $(CUB_DIR)
	
reach_t.o: reach_t.cu reach_t.h GPUerrchk.h
	$(CC) $(CFLAGS) -c reach_t.cu -I $(CUB_DIR)
	
reach_l.o: reach_l.cu reach_l.h GPUerrchk.h worklistc.h
	$(CC) $(CFLAGS) -c reach_l.cu -I $(CUB_DIR)

reach_d.o: reach_d.cu reach_d.h GPUerrchk.h worklistc.h
	$(CC) $(CFLAGS) -c reach_d.cu -I $(CUB_DIR)

wcc.o: wcc.cu wcc.h worklistc.h
	$(CC) $(CFLAGS) -c wcc.cu -I $(CUB_DIR)

clean:
	rm *.o scc scc_w scc_2_w
