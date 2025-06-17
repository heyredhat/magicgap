#!/bin/bash
ARGS_FILE="extremize_magic_args.txt"
NUM_TASKS=$(wc -l < "$ARGS_FILE")
ARRAY_MAX=$((NUM_TASKS - 1))
SLURM_SCRIPT=$(mktemp /tmp/run_extreme_magic_array.XXXXXX.sh)

cat <<EOF > "$SLURM_SCRIPT"
#!/bin/bash
#SBATCH --job-name=magicgap
#SBATCH --array=0-${ARRAY_MAX}
#SBATCH -n 100
#SBATCH --output=/hpcstor6/scratch01/m/matthew.weiss001/out_%A_%a.txt
#SBATCH --error=/hpcstor6/scratch01/m/matthew.weiss001/err_%A_%a.txt

module load anaconda3-2020.07-gcc-10.2.0-z5oxtnq

ARGS=\$(sed -n "\$((SLURM_ARRAY_TASK_ID + 1))p" $ARGS_FILE)
srun python extremize_magic.py \$ARGS
EOF

sbatch "$SLURM_SCRIPT"
rm "$SLURM_SCRIPT"
