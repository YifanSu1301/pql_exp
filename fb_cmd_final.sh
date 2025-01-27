# Syncing commands

rsync -ravzP --exclude outputs/ --exclude .env/ unnatj@grogu.ri.cmu.edu:/home/jsingla/pql /private/home/unnatjain/js/

# Running our method

cd /private/home/unnatjain/js/pql
conda activate pql
export LD_LIBRARY_PATH=$(conda info --base)/envs/pql/lib:$LD_LIBRARY_PATH

for i in 16 32 48
do
    for task in regrasping reorientation throw
    do
        python submit_it_fb.py --num_envs=24576 \
            --env=AllegroKuka \
            --task=$task \
            --seed $i  \
            --batch_size=49152 \
            --ngpus=2 \
            --wandb-entity=jsingla \
            --use_volta32
    done
done

# Allegro and Shadow Hand

for i in 16 32 48
do
    for env in AllegroHand ShadowHand
    do
        python submit_it_fb.py --num_envs=24576 \
            --env=$env \
            --seed $i  \
            --batch_size=49152 \
            --ngpus=2 \
            --wandb-entity=jsingla \
            --use_volta32
    done
done