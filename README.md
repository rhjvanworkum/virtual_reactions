# virtual_reactions

plan:
List of todo's:
    - code for querying Chembl compounds on similarity as well

- recalculate hessian at each step
- play with stepsize a bit


sbatch --cpus-per-task=8 --ntasks=1 --output=job_%A.out test.sh


mpirun -np 8 -> uses 8 tasks, cpus-per-task than indicates how many cores are allocated to each task