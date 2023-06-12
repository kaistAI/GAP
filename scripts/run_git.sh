export CUDA_VISIBLE_DEVICES=0

for i in {0..99}
do
    echo "$i"
    python run.py --config configs/git.json --index "$i"
done