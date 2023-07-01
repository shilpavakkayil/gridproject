import numpy as np
import matplotlib.pyplot as plt

BOARD_ROWS = 5
BOARD_COLS = 5
WIN_STATE = (4, 4)
START = (1, 0)
DETERMINISTIC = False
JUMP_STATE = (3, 3)
JUMP_START_STATE = (1, 3)
EPISODE_MAX = 100
LEARNING_RATE = 0.2
DECAY_GAMA = 0.9
EXP_RATE = 0.3
USER_CHOICE = '1'

class State:
    def __init__(self, state=START):
        self.board = np.zeros([BOARD_ROWS, BOARD_COLS])
        self.board[1, 1] = -1
        self.state = state
        self.isEnd = False
        self.determine = DETERMINISTIC

    def giveReward(self, prevstate):
        if self.state == WIN_STATE:
            return 10
        elif self.state == JUMP_STATE and prevstate == JUMP_START_STATE:
            return 5
        else:
            return -1

    def isEndFunc(self):
        if (self.state == WIN_STATE) :
            self.isEnd = True

    def _chooseActionProb(self, action):
        if action == "north":
            return np.random.choice(["north", "west", "east"], p=[0.8, 0.1, 0.1])
        if action == "south":
            return np.random.choice(["south", "west", "east"], p=[0.8, 0.1, 0.1])
        if action == "west":
            return np.random.choice(["west", "north", "south"], p=[0.8, 0.1, 0.1])
        if action == "east":
            return np.random.choice(["east", "north", "south"], p=[0.8, 0.1, 0.1])

    def nxtPosition(self, action):
        """
        action: north, south, west, east
        -------------
        0 | 1 | 2| 3|4
        1 |
        2 |
        3 |
        4 |
        return next position on board
        """
        if self.determine:
            if action == "north":
                nxtState = (self.state[0] - 1, self.state[1])
            elif action == "south":
                if self.state == JUMP_START_STATE:
                    nxtState = JUMP_STATE
                else:
                    nxtState = (self.state[0] + 1, self.state[1])
            elif action == "west":
                nxtState = (self.state[0], self.state[1] - 1)
            else:
                nxtState = (self.state[0], self.state[1] + 1)
            self.determine = False
        else:
            # non-deterministic
            action = self._chooseActionProb(action)
            self.determine = True
            nxtState = self.nxtPosition(action)

        # if next state is legal
        if (nxtState[0] >= 0) and (nxtState[0] <= 4):
            if (nxtState[1] >= 0) and (nxtState[1] <= 4):
                if nxtState != (3, 2) and nxtState != (2, 2) and nxtState !=(2, 3) and nxtState!=(2, 4):
                    return nxtState
        return self.state

    def showBoard(self):
        self.board[self.state] = 1
        for i in range(0, BOARD_ROWS):
            print('-----------------')
            out = '| '
            for j in range(0, BOARD_COLS):
                if self.board[i, j] == 1:
                    token = '*'
                if self.board[i, j] == -1:
                    token = 'z'
                if self.board[i, j] == 0:
                    token = '0'
                out += token + ' | '
            print(out)
        print('-----------------')


class Agent:

    def __init__(self):
        self.states = []  # record position and action taken at the position
        self.actions = ["north", "south", "west", "east"]
        self.State = State()
        self.isEnd = self.State.isEnd
        self.lr = LEARNING_RATE
        self.exp_rate = EXP_RATE
        self.decay_gamma = DECAY_GAMA
        self.episode_count = 0
        self.cum_reward_list = np.zeros(EPISODE_MAX)
        self.episode_array = np.zeros(EPISODE_MAX)

        # initial Q values
        self.Q_values = {}
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                self.Q_values[(i, j)] = {}
                for a in self.actions:
                    self.Q_values[(i, j)][a] = 0  # Q value is a dict of dict

    def chooseAction(self):
        # choose action with most expected value
        mx_nxt_reward = 0
        action = ""
        # exploration
        if np.random.uniform(0, 1) <= self.exp_rate:
            action = np.random.choice(self.actions)
        else:
            # greedy action
            current_position = self.State.state
            action_dict = self.Q_values[current_position]
            action = max(action_dict, key=action_dict.get)
        return action

    def takeAction(self, action):
        position = self.State.nxtPosition(action)
        # update State
        return State(state=position)

    def reset(self):
        self.states = []
        self.State = State()
        self.isEnd = self.State.isEnd
    def updateQvalue(self, cur_state, action, next_state, reward):
        current_q_value = self.Q_values[cur_state][action]
        q_val_array = self.Q_values[next_state]
        max_q_value = max(q_val_array.values())
        # if cur_state == WIN_STATE:
        #     for a in self.actions:
        #         self.Q_values[cur_state][a] = 0
        # else:
        current_q_value = current_q_value + self.lr*(reward + self.decay_gamma * max_q_value - current_q_value)
        self.Q_values[cur_state][action] = round(current_q_value,3)
    # display Q values in a grid
    def displaygrid(self):
        print("The Q-table")
        print("-----------------------------------------")
        print("\t\t|N\t\t|S\t\t|W\t\t|E\t\t|")
        print("-----------------------------------------")
        for rowdict in self.Q_values:
            row = self.Q_values[rowdict]
            print(rowdict, end='\t|')
            for coldict in row:
                value = row[coldict]
                if value == 0:
                    print(value, end ='\t\t|')
                else:
                    print(value, end = '\t|')
            print("\n-----------------------------------------")
    def displaygridstatevalues(self):

        #print("-----------------------------------------")
        #q_val_array = self.Q_values[next_state]
        #max_q_value = max(q_val_array.values())
        print("The state value table")
        print("-----------------------------------------")
        for i in range(BOARD_ROWS):
            state_val_array = "|"
            for j in range(BOARD_COLS):
                if max(self.Q_values[(i, j)].values()) == 0:
                    state_val_array += str(max(self.Q_values[(i, j)].values())) + '\t\t|'
                else:
                    state_val_array += str(max(self.Q_values[(i, j)].values())) + '\t|'
            print(state_val_array)
            print("-----------------------------------------")
    def play(self, rounds=10):
        i = 0
        while i < rounds:
            # checking if end state is reached or one episode is completed
            if self.State.isEnd:
                sum_of_rewards = 0.0
                reward = self.State.giveReward(None)
                #reward = 0
                self.states.append([(self.State.state), reward])
                for a in self.actions:
                    self.updateQvalue(self.State.state, a, self.State.state, reward)
                for s in self.states:
                    sum_of_rewards += s[1]
                print(f'COMPLETED EPISODE NO. :{self.episode_count + 1}')
                print("---------------------")
                print(f'Cumulative reward in Episode {self.episode_count + 1} :{sum_of_rewards}')
                self.cum_reward_list[self.episode_count] = sum_of_rewards
                #print("Game End Reward", reward)
                #self.displaygrid()
                #self.displaygridstatevalues()
                print("*======================================================*")
                # checks if the average cumulative reward is greater than 10 over 30 consecutive episodes
                self.episode_array[self.episode_count] = self.episode_count + 1
                if USER_CHOICE == '2':
                    if self.episode_count >= 29 and np.mean(self.cum_reward_list[self.episode_count - 29: self.episode_count + 1]) > 10:
                        starting_episode = self.episode_count - 29 + 1
                        end_episode = self.episode_count + 1
                        print(f' The training completed after 30 consecutive episodes starting from episode {starting_episode} to {end_episode} and average cumulative reward is {round(np.mean( self.cum_reward_list[self.episode_count -29 : self.episode_count + 1]),2)} >10')
                        break

                self.episode_count += 1
                self.reset()
                i += 1
            else:
                # the present state of the environment
                prevstate = self.State.state
                # choosing the action randomly or using greedy policy
                action = self.chooseAction()

                print("current position {} action {}".format(self.State.state, action))
                # as a result of action the enviroment is changed to next state
                self.State = self.takeAction(action)
                # the reward is given by the enviroment
                reward = self.State.giveReward(prevstate)
                #print(f'the reward {reward}')
                # the current state and reward are appended to the list of states in the episode
                self.states.append([(prevstate), reward])
                # updating the the Q value for the current state and action
                self.updateQvalue(prevstate, action, self.State.state, reward)

                # mark is end
                self.State.isEndFunc()
                print("nxt state", self.State.state)
                print(f"The reward is:{reward}")
                print("---------------------")
                self.isEnd = self.State.isEnd


if __name__ == "__main__":
    print("Choose an option from below")
    print("Enter 1 to Train the agent for 100 episodes")
    print("Enter 2 to halt the program when cumulative reward is greater than 10 over 30 consecutive episodes")
    print("Enter the choice:")
    ch = input()
    if ch == '1':
        print("The agent will be trained for 100 episodes")
    elif ch == '2':
        USER_CHOICE = ch
        print("The training will be halted when the cumulative reward accross 30 consecutive episodes is greater than 10")
    else:
        print("Wrong Choice")


    ag = Agent()
    print("initial Q-values ... \n")
    print(ag.Q_values)
    ag.displaygrid()


    ag.play(EPISODE_MAX)
    print("latest Q-values ... \n")
    print(ag.Q_values)
    ag.displaygrid()
    ag.displaygridstatevalues()

    new_arr_reward = []
    new_arr_episode = ag.episode_array[np.where(ag.episode_array != 0)]
    i = 0

    while i < len(new_arr_episode):
        new_arr_reward.append(ag.cum_reward_list[i])
        i += 1

    #print(len(new_arr_reward))
    #print(len(new_arr_episode))
    if len(new_arr_episode) == len(new_arr_reward):
        plt.plot(new_arr_episode, new_arr_reward)
        plt.xlabel('Episode')
        plt.ylabel('Cumulative Reward')
        plt.title('Cumulative Reward Accross Episodes')
        plt.show()
    else:
        pass