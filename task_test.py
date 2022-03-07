#!/bin/env python

from mpi4py import MPI
import task_queue
import time

comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()

class DummyTask:
    def __init__(self, task_nr):
        self.task_nr = task_nr
    def __call__(self):
        print("This is task %d on rank %d" % (self.task_nr, comm_rank))
        time.sleep(0.1)
        if self.task_nr==0:
            time.sleep(5.0)

if __name__ == "__main__":

    nr_queues = comm_size
    tasks = [[] for _ in range(nr_queues)]
    for i in range(nr_queues):
        tasks[i] = [DummyTask(100*i+j) for j in range(5)]
    
    task_queue.execute_tasks(tasks, args=(), comm_master=MPI.COMM_WORLD,
                             comm_workers=MPI.COMM_SELF, queue_per_rank=True)

    comm.barrier()
    if comm_rank == 0:
        print("Finished")
