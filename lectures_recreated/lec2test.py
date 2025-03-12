
import numpy as np # Linear Algebra
import matplotlib.pyplot as plt # Plotting
import copy

### TO-DO: Define the state space. We want to define each cell with a tuple (i,j), and we want to save all cells/states in a list called 'states'.
### ANSWER: Insert code here
states=[]
(rows,cols)=(4,4)
for i in range(rows):
    for j in range(cols):
        s=(i,j)
        states.append(s)
#print(states)
### END of ANSWER
(rows, cols) = (8, 8)
states = [(i, j) for i in range(rows) for j in range(cols)]
# Helper function that returns true when a cell is lava
'''def is_lava(s):
    return (s == (1, 1) or s == (1, 2))

# Helper function that returns true when a cell is terminal
def is_terminal(s):
    return (s == (3, 3))'''
def is_lava(s):
    temp = True if s in [(3, 1), (3, 2), (3, 3), (5, 5), (5, 6), (5, 7)] else False
    return temp;

def is_terminal(s):
    return (s == (6, 6));

### TO-DO: Define the action space. This is a list of all possible actions. Save it in a variable named 'actions'.
### ANSWER: Insert code here
a1,a2,a3,a4="Move Up","Move Left","Move Right","Move Down"
actions=[a1,a2,a3,a4]
#print(actions)
### END of ANSWER

# Let's define the transition table
transition = {}
# We initial all to zeros
for s in states:
    for a in actions:
        for sp in states:
            transition[(s, a, sp)] = 0.
### TO-DO: Fill all actions
### ANSWER: Insert code here
for s in states:
    for a in actions:
        sp=s
        if is_terminal(s):
            sp=s
        elif is_lava(s):
            sp=s
        elif a==a1: #up
            if s[0]>0: sp=(s[0]-1,s[1])
        elif a==a2: #left
            if s[1]>0: sp=(s[0],s[1]-1)
        elif a==a3: #right
            if s[1]< (cols-1): sp=(s[0],s[1]+1)
        elif a==a4: #down
            if s[0]< (rows-1): sp=(s[0]+1,s[1])
        transition[(s, a, sp)] = 1.
#print(transition)
### END of ANSWER

# Let's define the reward table/function
reward = {}
for s in states:
    ### TO-DO: Fill with values
    ### ANSWER: Insert code here
    if is_lava(s):
        reward[s]=-10.
    elif is_terminal(s):
        reward[s]=50.
    else:
        reward[s]=-1.
    ### END of ANSWER

def return_next_state(s,a):
    sp=s
    if is_terminal(s):
        sp=s
    elif is_lava(s):
        sp=s
    elif a==a1: #up
        if s[0]>0: sp=(s[0]-1,s[1])
    elif a==a2: #left
        if s[1]>0: sp=(s[0],s[1]-1)
    elif a==a3: #right
        if s[1]<cols-1: sp=(s[0],s[1]+1)
    elif a==a4: #down
        if s[0]<rows-1: sp=(s[0]+1,s[1])
    return sp

# Uniform policy
policy = {}
# Let's create a uniform random policy
for s in states:
    for a in actions:
        policy[(s, a)] = 1. / float(len(actions))

def mc_prediction(policy, num_episodes=10000, gamma=0.9):
    returns_sum = {s: 0 for s in states}  # Total return for each state
    returns_count = {s: 0 for s in states}  # Number of times a state is visited
    V = {s: 0.0 for s in states}  # Estimated value function

    for _ in range(num_episodes):
        episode = []
        s = (0, 0)  # Start state

        # Generate an episode
        while True:
            action_probs = [policy[(s, a)] for a in actions]
            a = np.random.choice(actions, p=action_probs)
            sp = return_next_state(s, a)  # Move first!

            episode.append((s, a, reward[s]))  # Store transition with reward from current state

            # Stop if the new state is terminal or lava (but after storing the reward)
            if is_terminal(sp) or is_lava(sp):
                action_probs = [policy[(s, a)] for a in actions]
                a = np.random.choice(actions, p=action_probs)
                episode.append((sp, a, reward[sp]))  # Store lava/terminal reward before breaking
                break  

            s = sp  # Move to next state

        # Monte Carlo value update (Every-Visit MC)
        G = 0
        for t in reversed(range(len(episode))):
            s, a, r = episode[t]
            G = gamma * G + r  # Compute return Gt
            
            # Every-Visit update
            returns_sum[s] += G  
            returns_count[s] += 1
            V[s] = returns_sum[s] / returns_count[s]  # Compute mean return

    return V


# Let's plot the value function
def plot_value_function(V):
    # Visualize Value function
    _, ax = plt.subplots()
    img = np.zeros((rows, cols))
    for state in states:
        img[state] = V[state]
    im = ax.imshow(img)
    ax.figure.colorbar(im, ax=ax)
    plt.show()


policy = {}
for s in states:
    for a in actions:
        policy[(s, a)] = 1. / float(len(actions)) # Uniform initial policy

#plot_value_function(mc_prediction(policy))


def td_learning(policy, num_episodes=10000, alpha=0.1, gamma=0.9):
    V = {s: 0.0 for s in states}  # Initialize value function

    for _ in range(num_episodes):
        s = (0, 0)  # Start state

        while True:
            # Choose action based on policy
            action_probs = [policy[(s, a)] for a in actions]
            a = np.random.choice(actions, p=action_probs)
            sp = return_next_state(s, a)  # Move to next state
            r = reward[s]  # Reward comes from the current state
            
            # If next state is terminal or lava, stop the episode after updating
            if is_terminal(s) or is_lava(s):
                V[s] = V[s] + alpha * (r - V[s])  # No future rewards
                break  
            else:
                V[s] = V[s] + alpha * (r + gamma * V[sp] - V[s])  # Standard TD update
            
            s = sp  # Move to next state

    return V

# Compute value function using MC
V_mc = mc_prediction(policy, num_episodes=10000, gamma=0.9)

# Compute value function using TD(0)
V_td = td_learning(policy, num_episodes=10000, alpha=0.1, gamma=0.9)

# Plot both results
print("Monte Carlo Value Function:")
plot_value_function(V_mc)

print("Temporal Difference (TD) Value Function:")
plot_value_function(V_td)
import numpy as np
import matplotlib.pyplot as plt

def epsilon_greedy(Q, s, epsilon=0.1):
    """Selects an action using ε-greedy policy"""
    if np.random.rand() < epsilon:
        return np.random.choice(actions)  # Explore
    else:
        return max(actions, key=lambda a: Q[(s, a)])  # Exploit

def sarsa(num_episodes=10000, alpha=0.1, gamma=0.9, epsilon=0.1):
    """SARSA On-Policy TD Control"""
    Q = {(s, a): 0.0 for s in states for a in actions}  # Initialize Q-values

    for _ in range(num_episodes):
        s = (0, 0)  # Start state
        a = epsilon_greedy(Q, s, epsilon)  # Select initial action
        
        while True:  # Until terminal/lava state
            sp = return_next_state(s, a)  # Next state
            r = reward[s]  # Immediate reward

           

            # SARSA Update
            #Q[(s, a)] = Q[(s, a)] + alpha * (r + gamma * Q[(sp, a_prime)] - Q[(s, a)])

            #s, a = sp, a_prime  # Move to next state-action pair

            if is_terminal(s) or is_lava(s):
                Q[(s, a)] = Q[(s, a)] + alpha * (r - Q[(s, a)])
                break   
            else:
                a_prime = epsilon_greedy(Q, sp, epsilon)  # Next action (on-policy)
                Q[(s, a)] = Q[(s, a)] + alpha * (r + gamma * Q[(sp, a_prime)] - Q[(s, a)])
                s, a = sp, a_prime  # Move to next state-action pair
    return Q

# Run SARSA
Q_sarsa = sarsa()

# Convert Q-values to a state-value function for visualization
V_sarsa = {s: max(Q_sarsa[(s, a)] for a in actions) for s in states}

# Plot the state-value function learned by SARSA
print("SARSA Value Function:")
plot_value_function(V_sarsa)


def q_learning(num_episodes=10000, alpha=0.1, gamma=0.9, epsilon=0.1):
    """Q-Learning Off-Policy TD Control"""
    Q = {(s, a): 0.0 for s in states for a in actions}  # Initialize Q-values

    for _ in range(num_episodes):
        s = (0, 0)  # Start state
        
        while True:
            a = epsilon_greedy(Q, s, epsilon)  # Select action using ε-greedy
            sp = return_next_state(s, a)  # Get next state
            r = reward[s]  # Immediate reward

            # If we reach a terminal or lava state, apply special update and break
            if is_terminal(s) or is_lava(s):
                Q[(s, a)] = Q[(s, a)] + alpha * (r - Q[(s, a)])  # No future rewards
                break  

            # Q-Learning Update Rule (Off-Policy)
            max_Q_next = max(Q[(sp, z)] for z in actions)  # Get max Q-value for next state
            Q[(s, a)] = Q[(s, a)] + alpha * (r + gamma * max_Q_next - Q[(s, a)])

            s = sp  # Move to next state

    return Q

# Run Q-Learning
Q_qlearning = q_learning()

# Convert Q-values to a state-value function for visualization
V_qlearning = {s: max(Q_qlearning[(s, a)] for a in actions) for s in states}

# Plot the state-value function learned by Q-Learning
print("Q-Learning Value Function:")
plot_value_function(V_qlearning)


def expected_sarsa(num_episodes=10000, alpha=0.1, gamma=0.9, epsilon=0.1):
    """Expected SARSA - Off-Policy Learning"""
    Q = {(s, a): 0.0 for s in states for a in actions}  # Initialize Q-values

    for _ in range(num_episodes):
        s = (0, 0)  # Start state
        
        while True:
            a = epsilon_greedy(Q, s, epsilon)  # Select action using ε-greedy
            sp = return_next_state(s, a)  # Get next state
            r = reward[s]  # Immediate reward

            # Ensure terminal/lava states are properly updated
            if is_terminal(s) or is_lava(s):
                Q[(s, a)] = Q[(s, a)] + alpha * (r - Q[(s, a)])  # No future rewards
                break  

            # Compute the expected Q-value over the policy in state s'
            expected_Q = sum(
                (epsilon / len(actions) + (1 - epsilon) * (1 if a_prime == epsilon_greedy(Q, sp, epsilon=0) else 0))
                * Q[(sp, a_prime)]
                for a_prime in actions
            )

            # Expected SARSA update rule
            Q[(s, a)] = Q[(s, a)] + alpha * (r + gamma * expected_Q - Q[(s, a)])

            s = sp  # Move to next state

    return Q

# Run Expected SARSA
Q_expected_sarsa = expected_sarsa()

# Convert Q-values to a state-value function for visualization
V_expected_sarsa = {s: max(Q_expected_sarsa[(s, a)] for a in actions) for s in states}

# Plot the state-value function learned by Expected SARSA
print("Expected SARSA Value Function:")
plot_value_function(V_expected_sarsa)