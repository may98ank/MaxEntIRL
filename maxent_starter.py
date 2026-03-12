import numpy as np
import numpy.random as rand
from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import math

        
def build_trans_mat_gridworld():
  # 5x5 gridworld laid out like:
  # 0  1  2  3  4
  # 5  6  7  8  9 
  # 9  10 11 12 13
  # 14 15 16 17 18
  # 20 21 22 23 24
  # where 24 is a goal state that always transitions to a 
  # special zero-reward terminal state (25) with no available actions
  trans_mat = np.zeros((26,4,26))
  
  # NOTE: the following iterations only happen for states 0-23.
  # This means terminal state 25 has zero probability to transition to any state, 
  # even itself, making it terminal, and state 24 is handled specially below.
  
  # Action 0 = down
  for s in range(24):
    if s < 20:
      trans_mat[s,0,s+5] = 1
    else:
      trans_mat[s,0,s] = 1
      
  # Action 1 = up
  for s in range(24):
    if s >= 5:
      trans_mat[s,1,s-5] = 1
    else:
      trans_mat[s,1,s] = 1
      
  # Action 2 = left
  for s in range(24):
    if s%5 > 0:
      trans_mat[s,2,s-1] = 1
    else:
      trans_mat[s,2,s] = 1
      
 # Action 3 = right
  for s in range(24):
    if s%5 < 4:
      trans_mat[s,3,s+1] = 1
    else:
      trans_mat[s,3,s] = 1

  # Finally, goal state always goes to zero reward terminal state
  for a in range(4):
    trans_mat[24,a,25] = 1  
      
  return trans_mat


def build_state_features_gridworld():
  # There are 4 features and only one is active at any given state, represented 1-hot vector at each state, with the layout as follows:
  # 0 0 0 0 0
  # 0 1 1 1 1
  # 0 0 2 0 0
  # 0 0 0 0 0 
  # 0 0 0 0 4
  # And the special terminal state (25) has all zero state features.

  sf = np.zeros((26,4))  
  sf[0,0] = 1
  sf[1,0] = 1
  sf[2,0] = 1
  sf[3,0] = 1
  sf[4,0] = 1
  sf[5,0] = 1
  sf[6,1] = 1
  sf[7,1] = 1
  sf[8,1] = 1
  sf[9,1] = 1
  sf[10,0] = 1
  sf[11,0] = 1
  sf[12,2] = 1
  sf[13,0] = 1
  sf[14,0] = 1
  sf[15,0] = 1
  sf[16,0] = 1
  sf[17,0] = 1
  sf[18,0] = 1
  sf[19,0] = 1
  sf[20,0] = 1
  sf[21,0] = 1
  sf[22,0] = 1
  sf[23,0] = 1
  sf[24,3] = 1
  return sf


           
def calcMaxEntPolicy(trans_mat, horizon, r_weights, state_features, term_index):
  """
  Implement steps 1-3 of Algorithm 1 in Ziebart et al.
  
  For a given reward function and horizon, calculate the MaxEnt policy that gives equal weight to equal reward trajectories
  
  trans_mat: an S x A x S' array of transition probabilites from state s to s' if action a is taken
  horizon: the finite time horizon (int) of the problem for calculating state frequencies
  r_weights: a size F array of the weights of the current reward function to evaluate
  state_features: an S x F array that lists F feature values for each state in S
  term_index: the index of the special terminal state
  
  return: an S x A policy in which each entry is the probability of taking action a in state s
  """
  n_states = np.shape(trans_mat)[0]
  n_actions = np.shape(trans_mat)[1]
  R = state_features @ r_weights
  Z = np.zeros(n_states)
  Z[term_index] = 1.0
  for _ in range(horizon):
    Z_new = np.zeros(n_states)
    for s in range(n_states):
      if s == term_index:
        Z_new[s] = 1.0
        continue
      for a in range(n_actions):
        next_partition = np.dot(trans_mat[s, a, :], Z)
        Z_new[s] += np.exp(R[s]) * next_partition
    Z_new[term_index] = 1.0
    Z = Z_new
  Z_action = np.zeros((n_states, n_actions))
  for s in range(n_states):
    for a in range(n_actions):
      Z_action[s, a] = np.exp(R[s]) * np.dot(trans_mat[s, a, :], Z)
  policy = np.zeros((n_states, n_actions))
  for s in range(n_states):
    if s == term_index:
      policy[s, :] = 0.0
      continue
    Z_s = np.sum(Z_action[s, :])
    if Z_s > 0:
      policy[s, :] = Z_action[s, :] / Z_s
    else:
      policy[s, :] = 1.0 / n_actions
    policy[s, :] /= policy[s, :].sum()
  return policy


def get_Z_values(trans_mat, horizon, r_weights, state_features, term_index):
  n_states = trans_mat.shape[0]
  n_actions = trans_mat.shape[1]
  R = state_features @ r_weights
  Z = np.zeros(n_states)
  Z[term_index] = 1.0
  for _ in range(horizon):
    Z_new = np.zeros(n_states)
    for s in range(n_states):
      if s == term_index:
        Z_new[s] = 1.0
        continue
      for a in range(n_actions):
        Z_new[s] += np.exp(R[s]) * np.dot(trans_mat[s, a, :], Z)
    Z_new[term_index] = 1.0
    Z = Z_new
  Z_action = np.zeros((n_states, n_actions))
  for s in range(n_states):
    for a in range(n_actions):
      Z_action[s, a] = np.exp(R[s]) * np.dot(trans_mat[s, a, :], Z)
  return Z, Z_action


def calcExpectedStateFreq(trans_mat, horizon, start_dist, policy):
  """
  Implement steps 4-6 of Algorithm 1 in Ziebart et al.
  
  Given a MaxEnt policy, begin with the start state distribution and propagate forward to find the expected state frequencies over the horizon
  
  trans_mat: an S x A x S' array of transition probabilites from state s to s' if action a is taken
  horizon: the finite time horizon (int) of the problem for calculating state frequencies
  start_dist: a size S array of starting start probabilities - must sum to 1
  policy: an S x A array array of probabilities of taking action a when in state s
  
  return: a size S array of expected state visitation frequencies
  """
  n_states = np.shape(trans_mat)[0]
  n_actions = np.shape(trans_mat)[1]
  D = np.zeros((n_states, horizon + 1))
  D[:, 0] = start_dist
  for t in range(horizon):
    for s_i in range(n_states):
      D[s_i, t + 1] = 0.0
      for s_k in range(n_states):
        for a in range(n_actions):
          D[s_i, t + 1] += D[s_k, t] * policy[s_k, a] * trans_mat[s_k, a, s_i]
  state_freq = np.sum(D, axis=1)
  return state_freq


def maxEntIRL(trans_mat, state_features, demos, seed_weights, n_epochs, horizon, learning_rate, term_index, return_history=False):
  """
  Implement the outer loop of MaxEnt IRL that takes gradient steps in weight space
  
  Compute a MaxEnt reward function from demonstration trajectories
  
  trans_mat: an S x A x S' array that describes transition probabilities from state s to s' if action a is taken
  state_features: an S x F array that lists F feature values for each state in S
  demos: a list of lists containing D demos of varying lengths, where each demo is series of states (ints)
  seed_weights: a size F array of starting reward weights
  n_epochs: how many times (int) to perform gradient descent steps
  horizon: the finite time horizon (int) of the problem for calculating state frequencies
  learning_rate: a multiplicative factor (float) that determines gradient step size
  term_index: the index of the special terminal state
  
  return: a size F array of reward weights
  """
  n_states = trans_mat.shape[0]
  n_features = state_features.shape[1]
  r_weights = np.copy(seed_weights)
  f_tilde = np.zeros(n_features)
  for demo in demos:
    for s in demo:
      f_tilde += state_features[s, :]
  f_tilde /= len(demos)
  start_dist = np.zeros(n_states)
  for demo in demos:
    start_dist[demo[0]] += 1.0
  start_dist /= len(demos)
  history = []
  for _ in range(n_epochs):
    policy = calcMaxEntPolicy(trans_mat, horizon, r_weights, state_features, term_index)
    state_freq = calcExpectedStateFreq(trans_mat, horizon, start_dist, policy)
    expected_features = state_freq @ state_features
    gradient = f_tilde - expected_features
    if return_history:
      history.append({
        'gradient_norm': np.linalg.norm(gradient),
        'r_weights': np.copy(r_weights),
        'expected_features': np.copy(expected_features)
      })
    r_weights += learning_rate * gradient
  if return_history:
    return r_weights, history
  return r_weights


def log_likelihood_demos(trans_mat, state_features, demos, r_weights, horizon, term_index):
  policy = calcMaxEntPolicy(trans_mat, horizon, r_weights, state_features, term_index)
  total_log_prob = 0.0
  for demo in demos:
    for t in range(len(demo) - 1):
      s_t, s_next = demo[t], demo[t + 1]
      a_t = np.argmax(trans_mat[s_t, :, s_next])
      if policy[s_t, a_t] <= 0:
        total_log_prob -= 1e10
        break
      total_log_prob += np.log(policy[s_t, a_t] + 1e-12)
  return total_log_prob / len(demos)


def numerical_gradient_check(trans_mat, state_features, demos, r_weights, horizon, term_index, eps=1e-5):
  n_features = state_features.shape[1]
  f_tilde = np.zeros(n_features)
  for demo in demos:
    for s in demo:
      f_tilde += state_features[s, :]
  f_tilde /= len(demos)
  start_dist = np.zeros(trans_mat.shape[0])
  for demo in demos:
    start_dist[demo[0]] += 1.0
  start_dist /= len(demos)
  policy = calcMaxEntPolicy(trans_mat, horizon, r_weights, state_features, term_index)
  state_freq = calcExpectedStateFreq(trans_mat, horizon, start_dist, policy)
  expected_features = state_freq @ state_features
  analytical_grad = f_tilde - expected_features
  numerical_grad = np.zeros(n_features)
  for i in range(n_features):
    r_plus = np.copy(r_weights)
    r_plus[i] += eps
    r_minus = np.copy(r_weights)
    r_minus[i] -= eps
    numerical_grad[i] = (log_likelihood_demos(trans_mat, state_features, demos, r_plus, horizon, term_index)
                         - log_likelihood_demos(trans_mat, state_features, demos, r_minus, horizon, term_index)) / (2 * eps)
  return analytical_grad, numerical_grad, np.max(np.abs(analytical_grad - numerical_grad))


def reward_weights_to_grid(r_weights, state_features, grid_shape=(5, 5), n_states=25):
  reward_fxn = np.array([np.dot(r_weights, state_features[s_i]) for s_i in range(n_states)])
  return np.reshape(reward_fxn, grid_shape)


def plot_reward_surface(reward_grid, title='Learned reward function', save_path=None, show=True):
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  X = np.arange(0, reward_grid.shape[1], 1)
  Y = np.arange(0, reward_grid.shape[0], 1)
  X, Y = np.meshgrid(X, Y)
  surf = ax.plot_surface(X, Y, reward_grid, rstride=1, cstride=1, cmap=cm.coolwarm,
                         linewidth=0, antialiased=False)
  ax.set_xlabel('x')
  ax.set_ylabel('y')
  ax.set_title(title)
  fig.colorbar(surf, shrink=0.5, aspect=5)
  if save_path:
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
  if show:
    plt.show()
  plt.close()


def plot_reward_heatmap(reward_grid, save_path=None):
  fig, ax = plt.subplots()
  im = ax.imshow(reward_grid, cmap='RdYlBu_r', aspect='equal')
  for i in range(reward_grid.shape[0]):
    for j in range(reward_grid.shape[1]):
      ax.text(j, i, '{:.2f}'.format(reward_grid[i, j]), ha='center', va='center', fontsize=8)
  ax.set_xticks(range(5))
  ax.set_yticks(range(5))
  ax.set_xlabel('column')
  ax.set_ylabel('row')
  plt.colorbar(im, ax=ax, label='reward')
  plt.tight_layout()
  if save_path:
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
  plt.close()


def plot_gradient_norms_curves(lr_results_list, n_epochs, save_path=None):
  fig, ax = plt.subplots()
  for lr, grad_norms in lr_results_list:
    ax.plot(range(1, n_epochs + 1), grad_norms, label='lr={}'.format(lr))
  ax.set_xlabel('epoch')
  ax.set_ylabel('gradient norm')
  ax.legend()
  ax.set_title('Gradient norm vs epoch')
  if save_path:
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
  plt.close()


def plot_Z_s_bars(Z_s, save_path=None):
  fig, ax = plt.subplots()
  states = np.arange(len(Z_s))
  ax.bar(states, Z_s, color='steelblue', edgecolor='gray')
  ax.axhline(1, color='gray', linestyle='--', label='terminal Z=1')
  ax.set_xlabel('state')
  ax.set_ylabel('Z_s')
  ax.set_title('Partition function Z_s by state')
  ax.legend()
  if save_path:
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
  plt.close()


def plot_feature_curves(history, f_tilde, save_path=None):
  n_epochs = len(history)
  n_features = len(f_tilde)
  fig, ax = plt.subplots()
  colors = ['C0', 'C1', 'C2', 'C3']
  epochs = range(1, n_epochs + 1)
  for k in range(n_features):
    vals = [h['expected_features'][k] for h in history]
    color = colors[k % len(colors)]
    ax.plot(epochs, vals, label='feature {}'.format(k), color=color)
    ax.hlines(f_tilde[k], 1, n_epochs, colors=color, linestyles='--', linewidth=1)
  ax.set_xlabel('epoch')
  ax.set_ylabel('expected feature count')
  ax.set_title('Feature expectations vs empirical counts')
  ax.legend()
  if save_path:
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
  plt.close()


def run_experiment(trans_mat, state_features, demos, n_epochs, horizon, learning_rate,
                   seed_weights=None, term_index=25, save_plot_path=None, experiment_name=None, show=True):
  if seed_weights is None:
    seed_weights = np.zeros(state_features.shape[1])
  r_weights = maxEntIRL(trans_mat, state_features, demos, seed_weights, n_epochs, horizon,
                        learning_rate, term_index)
  reward_grid = reward_weights_to_grid(r_weights, state_features)
  title = experiment_name or f'Reward (lr={learning_rate}, epochs={n_epochs}, H={horizon})'
  plot_reward_surface(reward_grid, title=title, save_path=save_plot_path, show=show)
  return r_weights, reward_grid


if __name__ == '__main__':
  import os
  os.makedirs('plots', exist_ok=True)
  trans_mat = build_trans_mat_gridworld()
  state_features = build_state_features_gridworld() 
  demos = [[4,9,14,19,24,25],[3,8,13,18,19,24,25],[2,1,0,5,10,15,20,21,22,23,24,25],[1,0,5,10,11,16,17,22,23,24,25]]
  term_index = 25
  seed_weights = np.zeros(4)
  n_epochs = 25
  horizon = 15

  summary_lines = []

  learning_rates = [0.01, 0.05, 0.1, 0.5, 1.0]
  lr_results = []
  lr_curves = []
  for lr in learning_rates:
    r_weights, history = maxEntIRL(trans_mat, state_features, demos, seed_weights, n_epochs, horizon, lr, term_index, return_history=True)
    grad_norms = [h['gradient_norm'] for h in history]
    reward_grid = reward_weights_to_grid(r_weights, state_features)
    plot_reward_surface(reward_grid, title='lr={}, epochs={}'.format(lr, n_epochs),
                        save_path='plots/reward_lr{}.png'.format(lr), show=False)
    np.savetxt('plots/gradient_norms_lr{}.txt'.format(lr), grad_norms, fmt='%.6f')
    lr_results.append((lr, r_weights, grad_norms[0], grad_norms[-1]))
    lr_curves.append((lr, grad_norms))
  plot_gradient_norms_curves(lr_curves, n_epochs, save_path='plots/gradient_norms_curves.png')

  print("\nLearning rate comparison (epochs={}, horizon={})".format(n_epochs, horizon))
  print("lr    | grad_norm(epoch1) | grad_norm(epoch{}) | final weights".format(n_epochs))
  for lr, w, g0, gN in lr_results:
    print("{:.2f}  | {:>17.4f} | {:>17.4f} | {}".format(lr, g0, gN, np.round(w, 3)))
  summary_lines.append("Q1: Learning rates tried: {}. Gradient norm decreases over epochs for all.".format(learning_rates))
  summary_lines.append("  Use table above to pick lr (e.g. 0.1): final weights and gradient decay.")

  n_epochs_100 = 100
  horizon_15 = 15
  lrs_q2 = [0.01, 0.1, 0.5]
  q2_results = []
  for lr in lrs_q2:
    r_weights_100 = maxEntIRL(trans_mat, state_features, demos, seed_weights, n_epochs_100, horizon_15, lr, term_index)
    reward_grid_100 = reward_weights_to_grid(r_weights_100, state_features)
    plot_reward_surface(reward_grid_100, title='100ep, H=15, lr={}'.format(lr),
                        save_path='plots/reward_100ep_lr{}.png'.format(lr), show=False)
    np.savetxt('plots/reward_grid_100ep_lr{}.csv'.format(lr), reward_grid_100, delimiter=',', fmt='%.6f')
    r12 = float(reward_grid_100.ravel()[12])
    r24 = float(reward_grid_100.ravel()[24])
    q2_results.append((lr, r_weights_100, reward_grid_100, r12, r24))
  r_weights_100 = q2_results[1][1]
  reward_grid_100 = q2_results[1][2]

  print("\n100 epochs, horizon=15 (multiple lr)")
  print("lr    | state 12 reward | state 24 (goal) reward |  plot")
  for lr, w, grid, r12, r24 in q2_results:
    print("{:.2f}  | {:>15.4f} | {:>22.4f} | reward_100ep_lr{}.png".format(lr, r12, r24, lr))
  print("Reward grid (5x5) for lr=0.1 (main run):")
  for row in range(5):
    print("  " + " ".join("{:7.2f}".format(reward_grid_100[row, col]) for col in range(5)))
  plot_reward_heatmap(reward_grid_100, save_path='plots/reward_100ep_lr0.1_heatmap.png')
  summary_lines.append("Q2: 100ep runs for lr in {}. Use reward_100ep_lr0.1.png for writeup. State 12 and 24 in table.".format(lrs_q2))

  r_weights_100, history_100 = maxEntIRL(trans_mat, state_features, demos, seed_weights, n_epochs_100, horizon_15, 0.1, term_index, return_history=True)
  n_features = state_features.shape[1]
  f_tilde = np.zeros(n_features)
  for demo in demos:
    for s in demo:
      f_tilde += state_features[s, :]
  f_tilde /= len(demos)
  plot_feature_curves(history_100, f_tilde, save_path='plots/feature_curves_100ep_lr0.1.png')

  theta_points = [np.array([-1.0, -2.0, -0.5, 0.3]), np.zeros(4), np.array([0.5, 0.5, -0.5, 0.2])]
  print("\nGradient check (analytical vs numerical)")
  for i, theta0 in enumerate(theta_points):
    ana, num, diff = numerical_gradient_check(trans_mat, state_features, demos, theta0, horizon, term_index, eps=1e-5)
    np.savetxt('plots/analytical_grad_{}.txt'.format(i), ana, fmt='%.6f')
    np.savetxt('plots/numerical_grad_{}.txt'.format(i), num, fmt='%.6f')
    print("theta[{}] = {}  -> max |ana - num| = {:.2e}".format(i, np.round(theta0, 2), diff))
  summary_lines.append("Q3: Gradient check at {} theta points. Files analytical_grad_*.txt, numerical_grad_*.txt.".format(len(theta_points)))

  horizons_z = [5, 15]
  print("\nPartition functions Z_s, Z_action")
  for H in horizons_z:
    Z_s, Z_action = get_Z_values(trans_mat, H, r_weights_100, state_features, term_index)
    np.savetxt('plots/Z_s_H{}.txt'.format(H), Z_s, fmt='%.6f')
    np.savetxt('plots/Z_action_H{}.txt'.format(H), Z_action, fmt='%.6f')
    print("horizon={}: Z_s[0]={:.4f}, Z_s[12]={:.4f}, Z_s[24]={:.4f}, Z_s[25]={:.4f}".format(H, Z_s[0], Z_s[12], Z_s[24], Z_s[25]))
  Z_s, Z_action = get_Z_values(trans_mat, horizon_15, r_weights_100, state_features, term_index)
  plot_Z_s_bars(Z_s, save_path='plots/Z_s_bars.png')
  print("Terminal Z_s[25]=1 by definition (empty trajectory, return 0, exp(0)=1).")
  summary_lines.append("Q5/Q6: Z_s saved for horizons {}. Terminal Z_s[25]=1.".format(horizons_z))

  with open('plots/summary.txt', 'w') as f:
    f.write("\n".join(summary_lines))
  print("\nSummary written to plots/summary.txt")



