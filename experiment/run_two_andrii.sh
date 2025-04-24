source ../.modula/bin/activate
python generate_sweeps_andrii.py

python train.py --job_idx 0 &
python train.py --job_idx 1 &
wait