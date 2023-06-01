# sbatch -N 1 --ntasks-per-node=8 --gres=gpu:2 --partition zen3_0512_a100x2 --qos zen3_0512_a100x2 --output=job_%A.out scripts/submit_vr_training.sh

# sbatch -N 1 --ntasks-per-node=16 --gres=gpu:2 --partition zen2_0256_a40x2 --qos zen2_0256_a40x2 --output=job_%A.out scripts/submit_vr_training.sh


with open('test.xyz', 'r') as f:
    lines = f.readlines()
    lines = [line for line in lines if len(line) > 1]

with open('test.xyz', 'w') as f:
    f.writelines(lines)