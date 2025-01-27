# Syncing commands

rsync -ravzPh --exclude outputs/ --exclude .env/ unnatj@grogu.ri.cmu.edu:/home/jsingla/pql /private/home/unnatjain/js/
export WANDB_HTTP_TIMEOUT=6000
# Running our method

cd /private/home/unnatjain/js/pql
conda activate pql
export LD_LIBRARY_PATH=$(conda info --base)/envs/pql/lib:$LD_LIBRARY_PATH

task=regrasping
for artifact in   
do
    python submit_it_fb.py --num_envs=24576 \
        --env=AllegroKuka \
        --task=$task \
        --batch_size=49152 \
        --ngpus=2 \
        --wandb-entity=jsingla \
        --use_volta32 \
        --artifact=jsingla/dexpbt_allegro_kuka_regrapsing/$artifact
done

task=reorientation
for artifact in   
do
    python submit_it_fb.py --num_envs=24576 \
        --env=AllegroKuka \
        --task=$task \
        --batch_size=49152 \
        --ngpus=2 \
        --wandb-entity=jsingla \
        --use_volta32 \
        --artifact=jsingla/dexpbt_allegro_kuka_reorientation/$artifact
done

task=throw
for artifact in   
do
    python submit_it_fb.py --num_envs=24576 \
        --env=AllegroKuka \
        --task=$task \
        --batch_size=49152 \
        --ngpus=2 \
        --wandb-entity=jsingla \
        --use_volta32 \
        --artifact=jsingla/dexpbt_allegro_kuka_throw/$artifact
done


# Allegro and Shadow Hand

env=AllegroHand
for artifact in   
do
    python submit_it_fb.py --num_envs=24576 \
        --env=$env \
        --batch_size=49152 \
        --ngpus=2 \
        --wandb-entity=jsingla \
        --use_volta32 \
        --artifact=jsingla/dexpbt_allegro_hand/$artifact
done

env=ShadowHand
for artifact in uid_00_PQL_16_29-01_04h30m46s:v292 uid_00_PQL_48_29-01_04h30m41s:v155
do
    python submit_it_fb.py --num_envs=24576 \
        --env=$env \
        --batch_size=49152 \
        --ngpus=2 \
        --wandb-entity=jsingla \
        --use_volta32 \
        --artifact=jsingla/dexpbt_shadow_hand/$artifact
done