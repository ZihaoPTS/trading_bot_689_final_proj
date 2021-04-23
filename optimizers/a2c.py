import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Softmax, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model

def actor_loss():
    def loss(advantage, predicted_output):
        """
            The policy gradient loss function.
            Note that you are required to define the Loss^PG
            which should be the integral of the policy gradient
            The "returns" is the one-hot encoded (return - baseline) value for each action a_t
            ('0' for unchosen actions).

            args:
                advantage: advantage of each action a_t (one-hot encoded).
                predicted_output: Predicted actions (action probabilities).

            Use:
                K.log: Element-wise log.
                K.sum: Sum of a tensor.
        """
        ################################
        #   YOUR IMPLEMENTATION HERE   #
        ################################
        log_probs = K.log(predicted_output)
        losses = -K.sum(advantage * log_probs, -1)
        # losses = K.print_tensor(losses, 'actor loss:')
        return losses

    return loss

def critic_loss():
    def loss(advantage, predicted_output):
        """
            The integral of the critic gradient

            args:
                advantage: advantage of each action a_t (one-hot encoded).
                predicted_output: Predicted state value.

            Use:
                K.sum: Sum of a tensor.
        """
        ################################
        #   YOUR IMPLEMENTATION HERE   #
        ################################
        # diffs = K.sum((advantage + predicted_output) - predicted_output, -1)
        # losses = K.mean(K.square(diffs))
        # losses = K.mean(K.sum(K.square(advantage), 1)) * predicted_output
        # losses = K.sum(0 * predicted_output + advantage)
        losses = -K.sum(advantage * predicted_output, -1)
        # losses = K.print_tensor(losses, 'critic loss:')
        return losses

    return loss

class A2C():
    def __init__(self, env, lr=0.001, layer_nodes=[64, 64]):
        self.lr = lr
        self.layer_nodes = layer_nodes
        self.env = env
        self.nA = self.env.action_space.n
        self.state_size = (self.env.observation_space.shape[0],)
        self.actor_critic = self.build_actor_critic()
        self.batch_size = 30
        self.gamma = 0.99
        self.episodes_run = 0

    def build_actor_critic(self):
        state = Input(shape=self.state_size, name='state')
        current_layer = state
        for nodes_for_layer in self.layer_nodes[:-1]:
            current_layer = Dense(nodes_for_layer, activation='relu')(current_layer)

        # Actor and critic heads have a seperated final fully connected layer
        current_actor_layer = Dense(self.layer_nodes[-1], activation='tanh')(current_layer)
        current_actor_layer = Dense(self.nA, activation='tanh')(current_actor_layer)
        current_critic_layer = Dense(self.layer_nodes[-1], activation='relu')(current_layer)

        policy = Softmax(name='policy')(current_actor_layer)
        state_value = Dense(1, activation='linear', name='state_value')(current_critic_layer)

        model = Model(inputs=[state], outputs=[policy, state_value])
        model.compile(optimizer=Adam(lr=self.lr),
                      loss={'policy': actor_loss(),
                            'state_value': critic_loss()},
                      loss_weights={'policy': 1.0, 'state_value': 1.0})
        return model

    def get_policy(self, state):
        return self.actor_critic(np.array([state]))[0][0].numpy()

    def choose_action(self, state):
        return np.random.choice(self.nA, p=self.get_policy(state))

    def generate_episode(self, render=False):
        """ Generates an episode and returns the trajectory """
        total_r = 0

        s = self.env.reset()
        done = False
        episode = []
        while not done:
            a = self.choose_action(s)
            s_prime, r, done, _ = self.env.step(a)
            episode.append((s, a, r))

            s = s_prime
            total_r += r

            if render:
                self.env.render()

        self.env.close()
        self.episodes_run += 1
        print(f'Reward from episode {self.episodes_run}: {total_r}')

        return episode

    def train_on_episode(self, render=False):
        """ Trains on a single episode """
        episode = self.generate_episode(render=render)

        # Estimate state values
        x_train = np.array([s for s, _, _ in episode])
        state_values = [prediction[0] for prediction in self.actor_critic.predict(x_train)[1]]

        # Go through episode determining action encodings and advantages
        deltas = np.empty((len(episode), self.nA))
        for i, ((s, a, r), v_s) in enumerate(zip(episode, state_values)):
            v_s_prime = state_values[i + 1] if i < len(episode) - 1 else 0
            advantage = r + self.gamma * v_s_prime - v_s

            # delta: advantage for taken action, 0 for other actions
            delta = np.zeros(self.env.action_space.n)
            delta[a] = advantage
            deltas[i] = delta

        y_train = {
            'policy': deltas,
            'state_value': deltas,
        }

        # Update actor critic
        self.actor_critic.train_on_batch(x=x_train, y=y_train)

    def train(self, n, *_, render_every=None):
        """ Trains on n episodes """
        for i in range(1, n + 1):
            render = render_every and i % render_every == 0
            self.train_on_episode(render=render)

    def get_model(self):
        """ Returns the actor-critic network for serialization or other external use """
        return self.actor_critic