from collections import deque
import numpy as np
import torch

def train_dqn(epochs, agent, env, gamma, tau, eps_decay, eps_min, train=True):
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    
    # reset the environment
    env_info = env.reset(train_mode=train)[brain_name]
    # number of actions
    action_size = brain.vector_action_space_size
    # examine the state space 
    state = env_info.vector_observations[0]
    state_size = len(state)
    
    eps = 1.0 if train else 0.0
    scores_window = deque(maxlen=100)
    scores = []

    for epoch in np.arange(1, epochs + 1):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]
        score = 0.

        done = False
        while not done:                
            action = agent.act(state, eps)
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]
            
            if train:
                agent.step(state, action, reward, next_state, done, gamma, tau)

            state = next_state
            score += reward
     
        if train:
            agent.scheduler.step()
            eps = max(eps * eps_decay, eps_min)

        elif epoch % 20 == 0:
            print("Epoch %d ended... Final score %.2f" % (epoch, float(score)))

        scores_window.append(score)
        scores.append(score)
        

        if epoch % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(epoch, np.mean(scores_window)))
        if (np.mean(scores_window)>=13) and train:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(epoch-100, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
        
    return scores