#!/bin/bash
set -x

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

cleanup() {
    set +x
    echo "Cleaning Progress..."
    kill -9 $server_pid > /dev/null 2>&1 || true
    pkill -9 -P $server_pid > /dev/null 2>&1 || true
    pkill -9 -f "ray::" > /dev/null 2>&1 || true
    ray stop --force > /dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

model_name="Qwen/Qwen3-VL-8B-Instruct"
train_data=[$(pwd)/data/my/train_6_448_448.parquet]
val_data=[$(pwd)/data/my/train_6_448_448.parquet]

rl_alg=grpo
n_gpus_per_node=8
n_nodes=1
n=8
total_training_steps=140
batch_size=16
ppo_mini_batch_size=8
val_batch_size=32
max_prompt_length=2176
max_action_length=4096
max_response_length=6144
max_obs_length=2048
ppo_max_token_len_per_gpu=$(expr $max_prompt_length + $max_response_length)
temperature=1.0
top_p=1.0
enable_agent=True
strategy="fsdp2"
action_stop_tokens="</call>,</keyframe>"
max_turns=6
kl_loss_coef=0.0
kl_coef=0.0
entropy_coeff=0
kl_loss_type=low_var_kl
lr=${LR:-}
reward_manager=my
ppo_micro_batch_size_per_gpu=1
log_prob_micro_batch_size_per_gpu=1
tensor_model_parallel_size=2
gpu_memory_utilization=0.4
disable_mm_cache=True
do_offload=False
use_dynamic_bsz=True
ulysses_sequence_parallel_size=1
fsdp_size=-1
additional_eos_token_ids=[151645]
mask_observations=True
enable_mtrl=True
workers_per_tool=8

sdpo_alpha=0.5
sdpo_topk=100
sdpo_teacher_update_rate=0.05
sdpo_feedback_max_chars=4096
sdpo_max_reprompt_len=${SDPO_MAX_REPROMPT_LEN:-10240}
sdpo_reprompt_truncation=${SDPO_REPROMPT_TRUNCATION:-right}
loss_mode=${LOSS_MODE:-rlsd_paper}
rlsd_clip_eps=${RLSD_CLIP_EPS:-0.2}
rlsd_lambda_init=${RLSD_LAMBDA_INIT:-0.5}
rlsd_lambda_start_step=${RLSD_LAMBDA_START_STEP:-1}
rlsd_lambda_decay_steps=${RLSD_LAMBDA_DECAY_STEPS:-50}
rlsd_teacher_update_mode=${RLSD_TEACHER_UPDATE_MODE:-sync}
rlsd_teacher_sync_interval=${RLSD_TEACHER_SYNC_INTERVAL:-10}
rlsd_teacher_update_rate=${RLSD_TEACHER_UPDATE_RATE:-$sdpo_teacher_update_rate}
rlsd_policy_clip_eps=${RLSD_POLICY_CLIP_EPS:-0.2}
rlsd_vanilla_on_nonnegative_adv=${RLSD_VANILLA_ON_NONNEGATIVE_ADV:-False}
rlsd_normalize_weight_by_seq_mean=${RLSD_NORMALIZE_WEIGHT_BY_SEQ_MEAN:-False}
if [ "$loss_mode" != "vanilla" ] && [ "$loss_mode" != "sdpo" ] && [ "$loss_mode" != "rlsd_paper" ]; then
    echo "LOSS_MODE must be vanilla, sdpo, or rlsd_paper, got $loss_mode"
    exit 1
fi
if [ -z "$lr" ]; then
    if [ "$loss_mode" = "sdpo" ]; then
        lr=1e-5
    else
        lr=1e-6
    fi
fi

model_pretty_name=$(echo $model_name | tr '/' '_' | tr '[:upper:]' '[:lower:]')
timestamp=$(date +"%Y%m%d_%H%M%S")
run_name="${timestamp}-${reward_manager}-agent-${model_pretty_name}-n${n}-b${batch_size}-${ppo_mini_batch_size}-t${temperature}-${loss_mode}"
export VERL_RUN_ID=$run_name
export NCCL_DEBUG=INFO
export VLLM_USE_V1=1
rollout_mode='async'
export WANDB_RUN_ID=$(echo -n "$run_name" | md5sum | cut -c1-8)
export WANDB_RESUME="allow"

# temp file for action tokens as verl cannot pass special strs as params
action_stop_tokens_file="$(pwd)$(mktemp)"
mkdir -p $(dirname $action_stop_tokens_file)
echo -e -n "$action_stop_tokens" | tee $action_stop_tokens_file
echo "action_stop_tokens_file=$action_stop_tokens_file"

host=$(hostname -i | awk '{print $1}')
port=$(shuf -i 30000-31000 -n 1)
tool_server_url=http://$host:$port/get_observation
python -m verl_tool.servers.serve --host $host --port $port --tool_type "vlm_tool,keyframe" --workers_per_tool $workers_per_tool &
server_pid=$!
echo "Server (pid=$server_pid) started at $tool_server_url"

PYTHONUNBUFFERED=1 python3 -m verl_tool.trainer.main_ppo \
    algorithm.adv_estimator=$rl_alg \
    +algorithm.filter_groups.enable=True \
    +algorithm.filter_groups.metric='seq_final_reward' \
    +algorithm.filter_groups.max_num_gen_batches=0 \
    algorithm.use_kl_in_reward=False \
    data.train_files=$train_data \
    data.val_files=$val_data \
    data.train_batch_size=$batch_size \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.filter_overlong_prompts=True \
    data.truncation='right' \
    reward_model.reward_manager=$reward_manager \
    reward_model.launch_reward_fn_async=True \
    actor_rollout_ref.model.path=$model_name \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.actor.optim.lr=$lr \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.checkpoint.save_contents=['model','optimizer','extra','hf_model'] \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_gpu \
    actor_rollout_ref.actor.use_dynamic_bsz=$use_dynamic_bsz \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.policy_loss.loss_mode=$loss_mode \
    actor_rollout_ref.actor.self_distillation.teacher_prompt_mode=reprompt \
    actor_rollout_ref.actor.self_distillation.include_environment_feedback=True \
    actor_rollout_ref.actor.self_distillation.environment_feedback_max_chars=$sdpo_feedback_max_chars \
    actor_rollout_ref.actor.self_distillation.max_reprompt_len=$sdpo_max_reprompt_len \
    actor_rollout_ref.actor.self_distillation.reprompt_truncation=$sdpo_reprompt_truncation \
    actor_rollout_ref.actor.self_distillation.full_logit_distillation=True \
    actor_rollout_ref.actor.self_distillation.distillation_topk=$sdpo_topk \
    actor_rollout_ref.actor.self_distillation.alpha=$sdpo_alpha \
    actor_rollout_ref.actor.self_distillation.teacher_regularization=ema \
    actor_rollout_ref.actor.self_distillation.teacher_update_rate=$sdpo_teacher_update_rate \
    actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=True \
    actor_rollout_ref.actor.self_distillation.success_reward_threshold=3.5 \
    actor_rollout_ref.actor.self_distillation.rlsd_clip_eps=$rlsd_clip_eps \
    actor_rollout_ref.actor.self_distillation.rlsd_lambda_init=$rlsd_lambda_init \
    actor_rollout_ref.actor.self_distillation.rlsd_lambda_start_step=$rlsd_lambda_start_step \
    actor_rollout_ref.actor.self_distillation.rlsd_lambda_decay_steps=$rlsd_lambda_decay_steps \
    actor_rollout_ref.actor.self_distillation.rlsd_teacher_update_mode=$rlsd_teacher_update_mode \
    actor_rollout_ref.actor.self_distillation.rlsd_teacher_sync_interval=$rlsd_teacher_sync_interval \
    actor_rollout_ref.actor.self_distillation.rlsd_teacher_update_rate=$rlsd_teacher_update_rate \
    actor_rollout_ref.actor.self_distillation.lambda_coef=$rlsd_lambda_init \
    actor_rollout_ref.actor.self_distillation.rlsd_vanilla_on_nonnegative_adv=$rlsd_vanilla_on_nonnegative_adv \
    actor_rollout_ref.actor.self_distillation.rlsd_normalize_weight_by_seq_mean=$rlsd_normalize_weight_by_seq_mean \
    actor_rollout_ref.actor.self_distillation.policy_clip_eps=$rlsd_policy_clip_eps \
    actor_rollout_ref.actor.strategy=$strategy \
    actor_rollout_ref.actor.kl_loss_coef=$kl_loss_coef \
    actor_rollout_ref.actor.kl_loss_type=$kl_loss_type \
    actor_rollout_ref.actor.entropy_coeff=$entropy_coeff \
    actor_rollout_ref.actor.fsdp_config.param_offload=$do_offload \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$do_offload \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=$fsdp_size \
    actor_rollout_ref.actor.clip_ratio_high=0.3 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.loss_agg_mode='token-mean' \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$ulysses_sequence_parallel_size \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$ppo_max_token_len_per_gpu \
    actor_rollout_ref.agent.enable_agent=$enable_agent \
    actor_rollout_ref.agent.tool_server_url=$tool_server_url \
    actor_rollout_ref.agent.max_prompt_length=$max_prompt_length \
    actor_rollout_ref.agent.max_response_length=$max_response_length \
    actor_rollout_ref.agent.max_start_length=$max_prompt_length \
    actor_rollout_ref.agent.max_obs_length=$max_obs_length \
    actor_rollout_ref.agent.max_turns=$max_turns \
    actor_rollout_ref.agent.additional_eos_token_ids=$additional_eos_token_ids \
    actor_rollout_ref.agent.mask_observations=$mask_observations \
    actor_rollout_ref.agent.action_stop_tokens=$action_stop_tokens_file \
    actor_rollout_ref.agent.enable_mtrl=$enable_mtrl \
    actor_rollout_ref.agent.max_action_length=$max_action_length \
    actor_rollout_ref.agent.mask_overlong_loss=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$tensor_model_parallel_size \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$log_prob_micro_batch_size_per_gpu \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=$gpu_memory_utilization \
    actor_rollout_ref.rollout.temperature=$temperature \
    actor_rollout_ref.rollout.top_p=$top_p \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.n=$n \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=$use_dynamic_bsz \
    actor_rollout_ref.rollout.max_num_seqs=128 \
    actor_rollout_ref.rollout.mode=$rollout_mode \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=$disable_mm_cache \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=$use_dynamic_bsz \
    actor_rollout_ref.ref.fsdp_config.param_offload=$do_offload \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$log_prob_micro_batch_size_per_gpu \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=$ulysses_sequence_parallel_size \
    critic.optim.lr=1e-5 \
    critic.strategy=$strategy \
    critic.model.path=$model_name \
    critic.model.fsdp_config.fsdp_size=$fsdp_size \
    critic.ppo_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_gpu \
    critic.ulysses_sequence_parallel_size=$ulysses_sequence_parallel_size \
    algorithm.kl_ctrl.kl_coef=$kl_coef \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$reward_manager \
    trainer.experiment_name=$run_name \
    trainer.val_before_train=False \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=$n_gpus_per_node \
    trainer.resume_mode=auto \
    +trainer.wandb_resume=auto \
    +trainer.wandb_id=$WANDB_RUN_ID \
    trainer.rollout_data_dir=$(pwd)/verl_step_records/$run_name \
    trainer.nnodes=$n_nodes \
    +trainer.remove_previous_ckpt_in_save=True \
    trainer.save_freq=20 \
    trainer.test_freq=0 \
    trainer.total_training_steps=$total_training_steps

pkill -P -9 $server_pid
kill -9 $server_pid
pkill -f "ray::"
ray stop --force || true
