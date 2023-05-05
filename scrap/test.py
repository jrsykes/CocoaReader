import os

delta_lst = [0.8, 1, 1.2, 1.4, 1.6, 1.8, 2]

#get slurm job number
job_id = os.environ["SLURM_ARRAY_TASK_ID"]
print(job_id)
job_id = int(job_id[-1:])

print(job_id)

print(delta_lst[job_id])