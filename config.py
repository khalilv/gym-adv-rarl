# ENV = "SwimmerAdv-v1"
# MAX_ADVERSARY_STRENGTH = 3
# SEED = 0
# INITIAL_OBSERVATION_STEPS = 25e3
# MAX_REPLAY_BUFFER_SIZE = 1e8
# BATCH_SIZE = 512
# DISCOUNT = 0.99
# TAU = 0.005
# POLICY_NOISE = 0.2
# EXPLORE_NOISE = 0.2
# NOISE_CLIP = 0.4
# POLICY_FREQUENCY = 2
# REWARD_THRESH = 360
# HIDDEN_LAYER_DIM = 256
# N_TRAINING = 6000
# EVAL_EPISODES = 10
# RARL_LOOPS = 100
# MAX_STEPS_PER_EPISODE = 600
# NUM_EXPERIMENTS = 5

ENV = 'HalfCheetahAdv-v1'
MAX_ADVERSARY_STRENGTH = 3
SEED = 0
INITIAL_OBSERVATION_STEPS = 25e3
MAX_REPLAY_BUFFER_SIZE = 1e8
BATCH_SIZE = 512
DISCOUNT = 0.99
TAU = 0.005
POLICY_NOISE = 0.2
EXPLORE_NOISE = 0.2
NOISE_CLIP = 0.4
POLICY_FREQUENCY = 2
REWARD_THRESH = 6000
HIDDEN_LAYER_DIM = 256
N_TRAINING = 5000
EVAL_EPISODES = 10
RARL_LOOPS = 100
MAX_STEPS_PER_EPISODE = 500
NUM_EXPERIMENTS = 5
SAVE_DIR = './halfcheetah_data/'