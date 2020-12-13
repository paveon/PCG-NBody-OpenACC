#/**
# * @file      Makefile
# *
# * @author    Ondřej Pavela xpavel34
# *            Faculty of Information Technology
# *            Brno University of Technology
# *            xpavel34@stud.fit.vutbr.cz
# *
# * @brief     PCG Assignment 2
# *            N-Body simulation in ACC - Makefile
# *
# * @version   2021
# *
# * @date      11 November  2020, 11:22 (created)
# * @date      11 November  2020, 11:37 (revised)
# *
# */

N=4096
DT=0.01f
STEPS=500
WRITE_INTESITY=20

INPUT=../sampledata/sampleInput.h5
OUTPUT=step1Output.h5

INCLUDE=../commons
LIBS=-lhdf5 -lm

FLAGS=  

.PHONY: all clean run profile


CC=pgc++
CFLAGS=-ta:tesla:pinned  -Minfo=accel
#CFLAGS=-ta:multicore -Minfo=accel
OPT=-O3 -Mcuda=fastmath


all: nbody

nbody: nbody.cpp main.cpp nbody.h
	${CC} ${CFLAGS} ${OPT} -I${INCLUDE} nbody.cpp main.cpp ../commons/h5Helper.cpp ${LIBS} -o nbody

clean:
	rm -f *.o nbody

run:
	./nbody ${N} ${DT} ${STEPS} ${WRITE_INTESITY} $(INPUT) $(OUTPUT)

profile:
	nvprof \
		--devices 0 \
		--metrics flops_sp \
		--metrics flops_sp_add \
		--metrics flops_sp_mul \
		--metrics flops_sp_fma \
		--metrics flops_sp_special \
		--metrics flop_sp_efficiency \
		--metrics sm_efficiency \
		--metrics alu_fu_utilization \
		--metrics gld_transactions \
		--metrics l1_cache_global_hit_rate \
		--metrics shared_load_transactions \
		./nbody ${N} ${DT} 1  0  $(INPUT) $(OUTPUT) 


my_profile:
	nvprof \
		--devices 0 \
		--metrics flops_sp \
		--metrics flops_sp_add \
		--metrics flops_sp_mul \
		--metrics flops_sp_fma \
		--metrics flops_sp_special \
		--metrics flop_sp_efficiency \
		--metrics sm_efficiency \
		--metrics achieved_occupancy \
		--metrics gld_transactions \
		--metrics gst_transactions \
		--metrics stall_memory_throttle \
		--metrics stall_memory_dependency \
		--metrics warp_execution_efficiency \
		./nbody ${N} ${DT} 1  0  $(INPUT) $(OUTPUT) 