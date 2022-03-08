#!/bin/env python

import time
import threading
import time
from mpi4py import MPI
import collections
import numpy as np

REQUEST_TASK_TAG=1
ASSIGN_TASK_TAG=2

def sleepy_recv(comm, tag):
    """
    Wait for a message without keeping a core spinning so that we leave
    the core available to run jobs and release the GIL. Checks for
    incoming messages at exponentially increasing intervals starting
    at 1.0e-8s up to a limit of ~1s. Sleeps between checks.
    """
    request = comm.irecv(tag=tag)
    delay = 1.0e-8
    while True:
        completed, message = request.test()
        if completed:
            return message
        if delay < 1.0:
            delay *= 2.0
        time.sleep(delay)

def distribute_tasks(tasks, comm):
    """
    Listen for and respond to requests for tasks to do
    """
    comm_size = comm.Get_size()
    next_task = 0
    nr_tasks = len(tasks)
    nr_done = 0
    while nr_done < comm_size:
        request_src = sleepy_recv(comm, REQUEST_TASK_TAG)
        if next_task < nr_tasks:
            comm.send(tasks[next_task], request_src, tag=ASSIGN_TASK_TAG)
            next_task += 1
        else:
            comm.send(None, request_src, tag=ASSIGN_TASK_TAG)
            nr_done += 1

def distribute_tasks_with_queue_per_rank(tasks, comm):
    """
    Listen for and respond to requests for tasks to do.
    In this case tasks is a sequence of comm_size task lists.
    Each rank will preferentially do tasks from it's own
    task list, but will do other tasks if it runs out.
    """
    comm_size = comm.Get_size()
    next_task = 0
    nr_tasks = sum([len(t) for t in tasks])
    tasks = [collections.deque(t) for t in tasks]
    nr_done = 0
    while nr_done < comm_size:
        request_src = sleepy_recv(comm, REQUEST_TASK_TAG)
        if next_task < nr_tasks:
            # If we have no tasks left for this rank, steal some!
            if len(tasks[request_src]) == 0:

                # Take the second half of the largest remaining task queue.
                #i = np.argmax([len(t) for t in tasks])
                #nr_steal = max(len(tasks[i])//2, 1)
                #for _ in range(nr_steal):
                #    tasks[request_src].appendleft(tasks[i].pop())

                # Or just take one task from the longest queue
                i = np.argmax([len(t) for t in tasks])
                tasks[request_src].append(tasks[i].popleft())
                
            # Get the next task for this rank
            task = tasks[request_src].popleft()
            # Send back the task
            comm.send(task, request_src, tag=ASSIGN_TASK_TAG)
            next_task += 1
        else:
            comm.send(None, request_src, tag=ASSIGN_TASK_TAG)
            nr_done += 1
            #print("Number of ranks done with all tasks = %d" % nr_done)
    #print("All tasks done.")

def execute_tasks(tasks, args, comm_all, comm_master, comm_workers,
                  queue_per_rank=False, return_timing=False):
    """
    Execute the tasks in tasks, which should be a sequence of
    callables which each return a result. Task objects are
    communicated over MPI so they must be pickleable. Each task is
    called as task(*args). The tasks argument is only significant
    on rank 0 of comm_master. The args argument is used on all
    ranks and should be used to pass comm_workers into the code
    executing each task.

    MPI ranks are split into groups of workers. The first rank in
    each worker group belongs to comm_master too. The comm_master
    communicator is used to issue tasks to each group of workers.
    Each task is run in parallel by all of the MPI ranks in one
    group of workers.

    The intended use of this to assign tasks to compute nodes and
    have all of the ranks on a node cooperate on a single task.

    Use comm_master=MPI.COMM_WORLD and comm_workers=MPI.COMM_SELF
    to run each task on a single MPI rank instead.

    On each MPI rank a list of results of the tasks which that rank
    participated in is returned. Ordering of the results is likely
    to be unpredictable!
    """

    timing = {}

    # Clone communicators to prevent message confusion:
    # In particular, tasks are likely to be using comm_workers internally.
    if comm_master != MPI.COMM_NULL:
        comm_master_local = comm_master.Dup()
    else:
        comm_master_local = MPI.COMM_NULL
    comm_workers_local = comm_workers.Dup()
    comm_all_local = comm_all.Dup()

    # Start the clock
    comm_all.barrier()
    overall_t0 = time.time()
    tasks_elapsed_time = 0.0

    # Get ranks in communicators
    master_rank = -1 if comm_master_local==MPI.COMM_NULL else comm_master_local.Get_rank()
    worker_rank = comm_workers_local.Get_rank()

    # First rank in comm_master starts a thread to hand out tasks
    if master_rank == 0:
        if queue_per_rank:
            task_queue_thread = threading.Thread(target=distribute_tasks_with_queue_per_rank,
                                                 args=(tasks, comm_master_local))
        else:
            task_queue_thread = threading.Thread(target=distribute_tasks,
                                                 args=(tasks, comm_master_local))
        task_queue_thread.start()

    # Request and run tasks until there are none left
    result = []
    while True:

        # The first rank in each group of workers requests a task and broadcasts it to the other workers.
        # If the task has a get_worker_task() method we broadcast whatever it returns to the other MPI ranks
        # and execute it. This allows having a different task object on the first rank, e.g. with extra data
        # which doesn't need to be duplicated to all ranks.
        if worker_rank == 0:
            comm_master_local.send(master_rank, 0, tag=REQUEST_TASK_TAG)
            task = comm_master_local.recv(tag=ASSIGN_TASK_TAG)
            if(hasattr(task, "get_worker_task")):
                worker_task = task.get_worker_task()
            else:
                worker_task = task
        else:
            worker_task = None
        worker_task = comm_workers_local.bcast(worker_task)
        if worker_rank != 0:
            task = worker_task

        # All workers in the group execute the task as a collective operation
        if task is not None:
            comm_workers_local.barrier()
            task_t0 = time.time()
            result.append(task(*args))
            comm_workers_local.barrier()
            task_t1 = time.time()
            tasks_elapsed_time += (task_t1-task_t0)
        else:
            break

    # Wait for task distributing thread to finish
    if master_rank == 0:
        task_queue_thread.join()

    # Stop the clock
    comm_all_local.barrier()
    overall_t1 = time.time()

    # Compute dead time
    time_total = comm_all_local.allreduce(overall_t1-overall_t0)
    time_tasks = comm_all_local.allreduce(tasks_elapsed_time)
    dead_fraction = (time_total - time_tasks) / time_total

    # Report total task time
    timing["elapsed"] = overall_t1 - overall_t0
    timing["dead_time_fraction"] = dead_fraction

    # Free local communicators
    if comm_master_local != MPI.COMM_NULL:
        comm_master_local.Free()
    comm_workers_local.Free()
    comm_all_local.Free()

    if return_timing:
        return result, timing
    else:
        return result
