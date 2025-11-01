@echo off
REM Get arguments
set ndim=%1
set horizon=%2
set seed=%3

REM Main
@REM python trainer.py --seed %seed% --agent tb --ndim %ndim% --horizon %horizon% --batch_size 64 --logger wandb --plot
python trainer.py --agent tb --ndim 2 --horizon 8 --logger wandb --plot --qd_on --run_name "tb-qd"
python trainer.py --agent tb --ndim 2 --horizon 16 --logger wandb --plot --qd_on --run_name "tb-qd"
python trainer.py --agent tb --ndim 2 --horizon 32 --logger wandb --plot --qd_on --run_name "tb-qd"
python trainer.py --agent tb --ndim 2 --horizon 64 --logger wandb --plot --qd_on --run_name "tb-qd"
python trainer.py --agent tb --ndim 2 --horizon 128 --logger wandb --plot --qd_on --run_name "tb-qd"
python trainer.py --agent tb --ndim 2 --horizon 256 --logger wandb --plot --qd_on --run_name "tb-qd"
python trainer.py --agent tb --ndim 4 --horizon 8 --logger wandb --plot --qd_on --run_name "tb-qd"
python trainer.py --agent tb --ndim 4 --horizon 16 --logger wandb --plot --qd_on --run_name "tb-qd"
python trainer.py --agent tb --ndim 4 --horizon 32 --logger wandb --plot --qd_on --run_name "tb-qd"
python trainer.py --agent tb --ndim 4 --horizon 64 --logger wandb --plot --qd_on --run_name "tb-qd"
python trainer.py --agent teacher --ndim 2 --horizon 8 --logger wandb --plot 
python trainer.py --agent teacher --ndim 2 --horizon 16 --logger wandb --plot
python trainer.py --agent teacher --ndim 2 --horizon 32 --logger wandb --plot
python trainer.py --agent teacher --ndim 2 --horizon 64 --logger wandb --plot
python trainer.py --agent teacher --ndim 2 --horizon 128 --logger wandb --plot
python trainer.py --agent teacher --ndim 2 --horizon 256 --logger wandb --plot
python trainer.py --agent teacher --ndim 4 --horizon 8 --logger wandb --plot
python trainer.py --agent teacher --ndim 4 --horizon 16 --logger wandb --plot
python trainer.py --agent teacher --ndim 4 --horizon 32 --logger wandb --plot
python trainer.py --agent teacher --ndim 4 --horizon 64 --logger wandb --plot


@REM # ndim=$1
@REM # horizon=$2
@REM # seed=$3

@REM # # Main
@REM # python trainer.py --seed $seed --agent tb --ndim $ndim --horizon $horizon --logger wandb --plot &
@REM #python trainer.py --seed $seed --agent tb --eps 0.01 --run_name "eps0.01" --ndim $ndim --horizon $horizon --logger wandb --plot &
@REM #python trainer.py --seed $seed --agent gafntb --ri_scale 0.01 --run_name "riscale0.01" --ndim $ndim --horizon $horizon --logger wandb --plot &
@REM #python trainer.py --seed $seed --agent teacher --ndim $ndim --horizon $horizon --logger wandb --plot &
@REM #python trainer.py --seed $seed --agent tb --use_buffer --buffer_pri reward --run_name "PRT" --ndim $ndim --horizon $horizon --logger wandb --plot &
@REM #python trainer.py --seed $seed --agent tb --use_buffer --buffer_pri teacher_reward --run_name "PER" --ndim $ndim --horizon $horizon --logger wandb --plot &
@REM #python trainer.py --seed $seed --agent teacher --use_buffer --buffer_pri reward --run_name "PRT" --ndim $ndim --horizon $horizon --logger wandb --plot &
@REM #python trainer.py --seed $seed --agent teacher --use_buffer --buffer_pri teacher_reward --run_name "PER" --ndim $ndim --horizon $horizon --logger wandb --plot &
@REM #
@REM #wait
@REM #
@REM ## Local search experiments
@REM #python trainer.py --seed $seed --agent tb --ls --run_name "ls" --ndim $ndim --horizon $horizon --logger wandb --plot &
@REM #python trainer.py --seed $seed --agent teacher --ls --run_name "ls" --ndim $ndim --horizon $horizon --logger wandb --plot &
@REM #python trainer.py --seed $seed --agent tb --ls --use_buffer --buffer_pri teacher_reward --run_name "ls_PER" --ndim $ndim --horizon $horizon --logger wandb --plot &
@REM #python trainer.py --seed $seed --agent teacher --ls --use_buffer --buffer_pri teacher_reward --run_name "ls_PER" --ndim $ndim --horizon $horizon --logger wandb --plot &
@REM #
@REM #wait
@REM #
@REM ## Detailed balance experimentss
@REM #python trainer_db.py --seed $seed --agent db --ndim $ndim --horizon $horizon --logger wandb --plot &
@REM #python trainer_db.py --seed $seed --agent db --use_buffer --buffer_pri teacher_reward --run_name "PER" --ndim $ndim --horizon $horizon --logger wandb --plot &
@REM #python trainer_db.py --seed $seed --agent teacher_db --ndim $ndim --horizon $horizon --logger wandb --plot &
@REM #python trainer_db.py --seed $seed --agent teacher_db --use_buffer --buffer_pri teacher_reward --run_name "PER" --ndim $ndim --horizon $horizon --logger wandb --plot &
@REM #
@REM #wait
