# Syncing command

# rsync -ravzP --exclude outputs/ --exclude .env/ unnatj@grogu.ri.cmu.edu:/home/jsingla/pql /private/home/unnatjain/js/

# Running our method

cd ~/pql
conda activate pql
export LD_LIBRARY_PATH=$(conda info --base)/envs/pql/lib:$LD_LIBRARY_PATH
python submit_it_fb.py --num_envs=24576 \
    --env=AllegroHand \
    --seed 42  \
    --batch_size=49152 \
    --ngpus=2 \
    --partition=prod1 \
    --wandb-entity=yifansu