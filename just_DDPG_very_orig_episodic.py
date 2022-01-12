import gym
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import time
import heapq

import highway_env
from gym.wrappers import Monitor
from IPython import display as ipythondisplay
from pyvirtualdisplay import Display
from gym.wrappers import Monitor
from pathlib import Path
import base64

# display = Display(visible=0, size=(1400, 900))
# display.start()


def record_videos(env, path="videos"):
    monitor = Monitor(env, path, force=True, video_callable=lambda episode: True)

    # Capture intermediate frames
    env.unwrapped.set_monitor(monitor)

    return monitor


def show_videos(path="videos"):
    html = []
    for mp4 in Path(path).glob("*.mp4"):
        video_b64 = base64.b64encode(mp4.read_bytes())
        html.append('''<video alt="{}" autoplay
                      loop controls style="height: 400px;">
                      <source src="data:video/mp4;base64,{}" type="video/mp4" />
                 </video>'''.format(mp4, video_b64.decode('ascii')))
    ipythondisplay.display(ipythondisplay.HTML(data="<br>".join(html)))


env = gym.make('highway-v0')
env = record_videos(env)
env.configure({
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 3,
        "features": ["presence", "x", "y", "vx", "vy"],
        "features_range": {
            "x": [-100, 100],
            "y": [-100, 100],
            "vx": [-20, 20],
            "vy": [-20, 20]
        },
        "normalize": False,
        "absolute": False
    },
    "action": {
        "type": "ContinuousAction"
        # "longitudinal": False
    },
    "lanes_count": 1,
    "vehicles_count": 1,
})

entry, features, agent_num, action_dim, angle_bound = 3, 5 + 3, 1, 1, 45
state_size = (entry, features)
action_size = (action_dim,)

swarm_state_size = (agent_num, entry, features)
swarm_action_size = (agent_num, action_dim)
memory_depth = 4
memory_state_size = (memory_depth, entry, features)
memory_action_size = (memory_depth, action_dim)

action_upper_bound = 0.01
action_lower_bound = -0.01
angle_upper_bound = angle_bound
angle_lower_bound = -angle_bound

obstacle_safe_range = 20
vev_safe_range = 40

obs_num = 0

"""
To implement better exploration by the Actor network, we use noisy perturbations,
specifically
an **Ornstein-Uhlenbeck process** for generating noise, as described in the paper.
It samples noise from a correlated normal distribution.
"""


class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = (
                self.x_prev
                + self.theta * (self.mean - self.x_prev) * self.dt
                + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)


class Buffer:
    def __init__(self, batch_size=10):
        # Number of "experiences" to store at max
        # self.buffer_capacity = buffer_capacity
        # Num of tuples to train on.
        self.batch_size = batch_size

        # Its tells us num of times record() was called.
        self.buffer_counter = 0

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element
        self.state_buffer = []
        self.action_buffer = []
        self.reward_buffer = []
        self.next_state_buffer = []
        self.done_buffer = []

        self.state_buffer_episodic = []
        self.action_buffer_episodic = []
        self.reward_buffer_episodic = []
        self.next_state_buffer_episodic = []
        self.done_buffer_episodic = []

    # Eager execution is turned on by default in TensorFlow 2. Decorating with tf.function allows
    # TensorFlow to build a static graph out of the logic and computations in our function.
    # This provides a large speed up for blocks of code that contain many small TensorFlow operations such as this one.
    @tf.function
    def update_critic(
            self, state_batch, action_batch, reward_batch, next_state_batch, done_batch
    ):
        # Training and updating Actor & Critic networks.
        # See Pseudo Code.
        with tf.GradientTape() as tape:
            target_actions = target_actor(next_state_batch, training=True)

            target_critic_value = target_critic(
                [next_state_batch, target_actions], training=True
            )
            done_batch_not = tf.math.logical_not(done_batch)
            done_batch_not = tf.cast(done_batch_not, dtype=tf.float32)
            target_critic_value *= done_batch_not

            y = reward_batch + gamma * target_critic_value

            critic_value = critic([state_batch, action_batch], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, critic.trainable_variables)
        critic_optimizer.apply_gradients(
            zip(critic_grad, critic.trainable_variables)
        )

    @tf.function
    def update_actor(
            self, state_batch, action_batch, reward_batch, next_state_batch,
    ):
        with tf.GradientTape() as tape:
            actions = actor(state_batch, training=True)
            critic_value = critic([state_batch, actions], training=True)
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, actor.trainable_variables)
        actor_optimizer.apply_gradients(
            zip(actor_grad, actor.trainable_variables)
        )

    # Takes (s,a,r,s') obervation tuple as input
    def record(self, obs_tuple):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        # index = self.buffer_counter % self.buffer_capacity

        self.state_buffer.append(obs_tuple[0])
        self.action_buffer.append(obs_tuple[1])
        self.reward_buffer.append(obs_tuple[2])
        self.next_state_buffer.append(obs_tuple[3])
        self.done_buffer.append(obs_tuple[4])

        # self.buffer_counter += 1

        # return index

    def record_episodic(self):
        self.state_buffer_episodic.append(self.state_buffer)
        self.action_buffer_episodic.append(self.action_buffer)
        self.reward_buffer_episodic.append(self.reward_buffer)
        self.next_state_buffer_episodic.append(self.next_state_buffer)
        self.done_buffer_episodic.append(self.done_buffer)

        self.buffer_counter += 1

        self.state_buffer = []
        self.action_buffer = []
        self.reward_buffer = []
        self.next_state_buffer = []
        self.done_buffer = []

    # We compute the loss and update parameters
    def learn(self, actor_flag):
        # Get sampling range
        record_range = self.buffer_counter
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size)
        # batch_indices = [-1]
        # batch_indices0 = np.array(heapq.nlargest(np.int(self.batch_size/2), range(len(self.ep_reward_buffer)),
        #                                          self.ep_reward_buffer.take))
        # batch_indices1 = np.random.choice(record_range, np.int(self.batch_size/2))
        # batch_indices = np.concatenate([batch_indices0, batch_indices1])

        # Sampling batches
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = [], [], [], [], []
        batch_len = len(batch_indices)
        for i in range(batch_len):
            index_temp = batch_indices[i]
            state_batch.append(self.state_buffer_episodic[index_temp])
            action_batch.append(self.action_buffer_episodic[index_temp])
            reward_batch.append(self.reward_buffer_episodic[index_temp])
            next_state_batch.append(self.next_state_buffer_episodic[index_temp])
            done_batch.append(self.done_buffer_episodic[index_temp])

        # Convert to tensors
        state_batch = tf.convert_to_tensor(np.vstack(state_batch))
        action_batch = tf.convert_to_tensor(np.vstack(action_batch))
        reward_batch = tf.expand_dims(tf.convert_to_tensor(np.hstack(reward_batch)), axis=-1)
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(np.vstack(next_state_batch))
        done_batch = tf.expand_dims(tf.convert_to_tensor(np.hstack(done_batch)), axis=-1)

        self.update_critic(state_batch, action_batch, reward_batch, next_state_batch, done_batch)
        if actor_flag:
            self.update_actor(state_batch, action_batch, reward_batch, next_state_batch)


# This update target parameters slowly
# Based on rate `tau`, which is much less than one.
@tf.function
def update_target(target_weights, weights, tau):
    for (a, b) in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))


"""
Here we define the Actor and Critic networks. These are basic Dense models
with `ReLU` activation.

Note: We need the initialization for last layer of the Actor to be between
`-0.003` and `0.003` as this prevents us from getting `1` or `-1` output values in
the initial stages, which would squash our gradients to zero,
as we use the `tanh` activation.
"""


def get_actor():
    # Initialize weights between -3e-3 and 3-e3
    last_init = tf.random_uniform_initializer(minval=-0.00003, maxval=0.00003)

    inputs = layers.Input(shape=state_size)
    out = layers.Flatten()(inputs)
    out = layers.Dense(256, activation="relu")(out)
    out = layers.Dense(256, activation="relu")(out)
    outputs = layers.Dense(1, activation="tanh", kernel_initializer=last_init)(out)

    # outputs = layers.Dense(1, activation="linear", kernel_initializer=last_init)(out)
    # outputs = tf.clip_by_value(outputs, clip_value_min=action_lower_bound, clip_value_max=action_upper_bound)

    # Our upper bound is 2.0 for Pendulum.
    outputs = outputs * action_upper_bound
    model = tf.keras.Model(inputs, outputs)

    return model


def get_critic():
    # State as input
    state_input = layers.Input(shape=state_size)
    state_out = layers.Flatten()(state_input)
    state_out = layers.Dense(16, activation="relu")(state_out)
    state_out = layers.Dense(32, activation="relu")(state_out)

    # Action as input
    action_input = layers.Input(shape=action_size)
    action_out = layers.Dense(32, activation="relu")(action_input)

    # Both are passed through seperate layer before concatenating
    concat = layers.Concatenate()([state_out, action_out])

    out = layers.Dense(256, activation="relu")(concat)
    out = layers.Dense(256, activation="relu")(out)
    outputs = layers.Dense(1)(out)

    # Outputs single value for give state-action
    model = tf.keras.Model([state_input, action_input], outputs)

    return model


"""
`policy()` returns an action sampled from our Actor network plus some noise for
exploration.
"""


def policy(state, ou_noise, decay_step):
    explore_probability = epsilon_min + (epsilon - epsilon_min) * np.exp(
        -epsilon_decay * decay_step)
    sampled_actions = tf.squeeze(actor(state))
    # legal_noise = ou_noise()
    # legal_noise = np.random.normal(loc=0, scale=0.1, size=sampled_actions.shape)
    # legal_noise = np.clip(legal_noise, action_lower_bound, action_upper_bound)
    legal_noise = np.random.uniform(low=action_lower_bound*1, high=action_upper_bound*1,
                                    size=sampled_actions.shape)
    # if explore_probability > np.random.rand():
    #     noise_actions = sampled_actions * 1 + legal_noise * 1
    #     print('Noise action')
    # else:
    #     noise_actions = sampled_actions
    #     print('-----> Policy action')

    # We make sure action is within bounds
    noise_actions = sampled_actions * (1-explore_probability) + legal_noise * explore_probability
    print('explore_probability: {}'.format(explore_probability))
    legal_action = np.clip(noise_actions, action_lower_bound, action_upper_bound)

    return [legal_action]


"""
## Training hyperparameters
"""

std_dev = 0.01
ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))

actor = get_actor()
critic = get_critic()

target_actor = get_actor()
target_critic = get_critic()

# Making the weights equal initially
target_actor.set_weights(actor.get_weights())
target_critic.set_weights(critic.get_weights())

# Learning rate for actor-critic models
critic_lr = 1e-3
actor_lr = 1e-5

critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
actor_optimizer = tf.keras.optimizers.Adam(actor_lr)

total_episodes = 1000
# Discount factor for future rewards
gamma = 0.99
# Used to update target networks
tau = 0.001

buffer = Buffer(10)

"""
Now we implement our main training loop, and iterate over episodes.
We sample actions using `policy()` and train with `learn()` at each time step,
along with updating the Target networks at a rate `tau`.
"""

# To store reward history of each episode
ep_reward_list = []
# To store average reward history of last few episodes
avg_reward_list = []

epsilon_min = 0.01
epsilon = 1
epsilon_decay = 3e-04

decay_step = 0
# Takes about 4 min to train
plt.figure()
plt.xlabel("Episode")
plt.ylabel("Avg. Epsiodic Reward")
# plt.axis([0, total_episodes, 0, 100])
for ep in range(total_episodes):

    prev_state = env.reset()
    lane = np.zeros([entry, 2])  # vx_init, vy_init
    lane[0][0], lane[0][1] = 1, 0
    origin = np.zeros([entry, 1])  # y_init
    origin[0] = prev_state[0][2]
    prev_state = np.concatenate([prev_state, lane, origin], axis=1)
    episodic_reward = 0
    step = 0
    off_road_penalty_pre, vel_reward_pre = 0, 0

    done = False

    while not done:
        # Uncomment this to see the Actor in action
        # But not in a python notebook.
        env.render()

        tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)

        action = policy(tf_prev_state, ou_noise, decay_step)
        # Recieve state and reward from environment.
        state, reward, done, info = env.step([0, action[0]])
        state = np.concatenate([state, lane, origin], axis=1)
        # vx, vy = state[0][3] * state[0][5], state[0][4] * state[0][6]
        vx, vy = state[0][3] / (np.sqrt(state[0][3] ** 2 + state[0][4] ** 2)), \
                 state[0][4] / (np.sqrt(state[0][3] ** 2 + state[0][4] ** 2))

        vx_init, vy_init = state[0][5], state[0][6]
        y = state[0][2]
        y_init = state[0][7]
        vel_reward = vx * vx_init + vy * vy_init
        off_road_penalty = np.abs(y_init - y)
        if info['crashed']:
            collide_penalty = 10
        else:
            collide_penalty = 0
        reward = 1 - np.tanh(off_road_penalty * .1) - collide_penalty
        # if off_road_penalty <= 1:
        #     reward = 1 - collide_penalty
        # else:
        #     reward = 0 - collide_penalty
        # reward = +vel_reward - np.tanh(off_road_penalty * .1) - collide_penalty
        # reward = +vel_reward * (1 - np.tanh(off_road_penalty * .1)) - collide_penalty
        # reward = +vel_reward - np.tanh(off_road_penalty * 1) - collide_penalty
        # reward = +1 - np.tanh(off_road_penalty * .1) - collide_penalty

        # if off_road_penalty == 0:
        #     reward0 = 1
        # else:
        #     if off_road_penalty < off_road_penalty_pre:
        #         reward0 = 1
        #     else:
        #         reward0 = 0

        # if vel_reward_pre > vel_reward:
        #     reward1 = 1
        # elif vel_reward_pre < vel_reward:
        #     reward1 = -1
        # else:
        #     reward1 = 0
        # reward1 = 0
        # reward = reward0 + reward1 - collide_penalty

        off_road_penalty_pre, vel_reward_pre = off_road_penalty, vel_reward

        buffer.record((prev_state, action, reward, state, done))
        episodic_reward += reward

        update_target(target_actor.variables, actor.variables, tau)
        update_target(target_critic.variables, critic.variables, tau)

        # End this episode when `done` is True
        # if done:
        #     break

        prev_state = state

        print("Episode {} step {} time {}".format(ep, step, time.ctime()))
        print("action: ")
        print(action)
        print("reward: ")
        print(reward)

        step += 1
        decay_step += 1

    buffer.record_episodic()

    if ep >= 10:
        if ep % 3 == 0:
            actor_flag = True
        else:
            actor_flag = False
        buffer.learn(actor_flag)

    ep_reward_list.append(episodic_reward)

    # Mean of last 40 episodes
    avg_reward = np.mean(ep_reward_list[-40:])
    # print("Episode * {} * action: {} Avg Reward is ==> {}".format(ep, action, avg_reward))
    avg_reward_list.append(avg_reward)

    plt.plot(np.arange(0, ep+1), ep_reward_list, 'r')
    plt.plot(np.arange(0, ep+1), avg_reward_list, 'b')
    plt.pause(0.05)

# Plotting graph
# Episodes versus Avg. Rewards
plt.plot(avg_reward_list)
plt.xlabel("Episode")
plt.ylabel("Avg. Epsiodic Reward")
plt.show()
env.close()
show_videos()

"""
If training proceeds correctly, the average episodic reward will increase with time.

Feel free to try different learning rates, `tau` values, and architectures for the
Actor and Critic networks.

The Inverted Pendulum problem has low complexity, but DDPG work great on many other
problems.

Another great environment to try this on is `LunarLandingContinuous-v2`, but it will take
more episodes to obtain good results.
"""

# Save the weights
actor.save_weights("pendulum_actor.h5")
critic.save_weights("pendulum_critic.h5")

target_actor.save_weights("pendulum_target_actor.h5")
target_critic.save_weights("pendulum_target_critic.h5")

"""
Before Training:

![before_img](https://i.imgur.com/ox6b9rC.gif)
"""

"""
After 100 episodes:

![after_img](https://i.imgur.com/eEH8Cz6.gif)
"""
