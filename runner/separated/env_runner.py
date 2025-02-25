import time
import os
import numpy as np
import torch
from pathlib import Path
from runner.separated.base_runner import Runner
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict


def _t2n(x):
    return x.detach().cpu().numpy()


class EnvRunner(Runner):
    def __init__(self, config):
        super(EnvRunner, self).__init__(config)

    def average_dicts(self,reward_dict_list):
        keys = reward_dict_list[0].keys()
        num_dicts = self.n_rollout_threads

        sum_dict = {key: 0 for key in keys}
        for d in reward_dict_list:
            for key in keys:
                sum_dict[key] += d[key]

        average_result = {key: sum_dict[key] / num_dicts for key in keys}
        # print(average_result)
        return average_result
    
    def run(self, args):
        reward_dir = str(self.run_dir / "reward")
        if not os.path.exists(reward_dir):
            os.makedirs(reward_dir)
        reward_files = []
        for i in range(self.robots_num):
            reward_files.append(os.path.join(reward_dir, f"reward{i}.txt"))

        self.warmup()

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                for agent_id in range(self.robots_num):
                    self.trainer[agent_id].policy.lr_decay(episode, episodes)

            episode_rewards = np.zeros([self.n_rollout_threads,self.robots_num,1])
            episode_rewards_info = []
            for step in range(self.episode_length):
                # Sample actions
                (
                    values,
                    actions,
                    action_log_probs,
                    rnn_states,
                    rnn_states_critic,
                    actions_env,
                ) = self.collect(step)

                # Obser reward and next obs
                obs, share_obs, rewards, dones, reward_info = self.envs.step(actions)
                # print(rewards)
                episode_rewards += rewards
                step_rewards_info = self.average_dicts(reward_info)
                episode_rewards_info.append(step_rewards_info)

                data = (obs,share_obs,rewards,dones,reward_info,values,actions,action_log_probs,rnn_states,rnn_states_critic,)

                # insert data into buffer
                self.insert(data)

            average_episode_rewards_info = self.average_dicts(episode_rewards_info)
            # compute return and update network
            self.compute()
            train_infos = self.train()
            for i in range(self.robots_num):
                with open(reward_files[i], "a") as f:
                    rewards_str = ', '.join(map(str, episode_rewards[:,i,:]))
                    line = f"{episode+1}, {rewards_str.replace('[', '').replace(']', '')}\n"
                    f.write(line)

            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads

            # save model
            if episode % self.save_interval == 0 or episode == episodes - 1:
                self.save()

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print(average_episode_rewards_info)
                print(
                    "Algo {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n".format(
                        self.algorithm_name,
                        episode+1,
                        episodes,
                        total_num_steps,
                        self.num_env_steps,
                        int(total_num_steps / (end - start)),
                    ))
                
                self.log_train(train_infos, total_num_steps)
            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    def warmup(self):
        # reset env
        obs,share_obs = self.envs.reset()  # shape = [env_num, robot_num, 1+human_num, obs_dim]

        for agent_id in range(self.robots_num):
            self.buffer[agent_id].share_obs[0] = share_obs.copy()
            self.buffer[agent_id].robot_obs[0] = obs[:,agent_id,0].copy()
            self.buffer[agent_id].humans_obs[0] = obs[:,agent_id,1:,:self.human_obs_dim].copy()

    @torch.no_grad()
    def collect(self, step):
        values = []
        actions = []
        temp_actions_env = []
        action_log_probs = []
        rnn_states = []
        rnn_states_critic = []

        for agent_id in range(self.robots_num):
            self.trainer[agent_id].prep_rollout()
            value, action, action_log_prob, rnn_state, rnn_state_critic = self.trainer[agent_id].policy.get_actions(
                self.buffer[agent_id].share_obs[step],
                self.buffer[agent_id].robot_obs[step],
                self.buffer[agent_id].humans_obs[step],
                self.buffer[agent_id].rnn_states[step],
                self.buffer[agent_id].rnn_states_critic[step],
                self.buffer[agent_id].masks[step],
            )
            # [agents, envs, dim]
            values.append(_t2n(value))
            action = _t2n(action)

            # rearrange action
            if self.envs.action_space[agent_id].__class__.__name__ == "MultiDiscrete":
                for i in range(self.envs.action_space[agent_id].shape):
                    uc_action_env = np.eye(self.envs.action_space[agent_id].high[i] + 1)[action[:, i]]
                    if i == 0:
                        action_env = uc_action_env
                    else:
                        action_env = np.concatenate((action_env, uc_action_env), axis=1)
            elif self.envs.action_space[agent_id].__class__.__name__ == "Discrete":
                action_env = np.squeeze(np.eye(self.envs.action_space[agent_id].n)[action], 1)
            else:
                action_env = action

            actions.append(action)
            temp_actions_env.append(action_env)
            action_log_probs.append(_t2n(action_log_prob))
            rnn_states.append(_t2n(rnn_state))
            rnn_states_critic.append(_t2n(rnn_state_critic))

        # [envs, agents, dim]
        actions_env = []
        for i in range(self.n_rollout_threads):
            one_hot_action_env = []
            for temp_action_env in temp_actions_env:
                one_hot_action_env.append(temp_action_env[i])
            actions_env.append(one_hot_action_env)

        values = np.array(values).transpose(1, 0, 2)
        actions = np.array(actions).transpose(1, 0, 2)
        action_log_probs = np.array(action_log_probs).transpose(1, 0, 2)
        rnn_states = np.array(rnn_states).transpose(1, 0, 2, 3)
        rnn_states_critic = np.array(rnn_states_critic).transpose(1, 0, 2, 3)

        return (
            values,
            actions,
            action_log_probs,
            rnn_states,
            rnn_states_critic,
            actions_env,
        )

    def insert(self, data):
        (obs,
         share_obs,
            rewards,
            dones,
            reward_info,
            values,
            actions,
            action_log_probs,
            rnn_states,
            rnn_states_critic,) = data

        rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size),dtype=np.float32)   
        rnn_states_critic[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size),dtype=np.float32)
        masks = np.ones((self.n_rollout_threads, self.robots_num, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        for agent_id in range(self.robots_num):
            self.buffer[agent_id].insert(
                share_obs,
                obs[:, agent_id, 0],
                obs[:, agent_id, 1:, :self.human_obs_dim],
                rnn_states[:, agent_id],
                rnn_states_critic[:, agent_id],
                actions[:, agent_id],
                action_log_probs[:, agent_id],
                values[:, agent_id],
                rewards[:, agent_id],
                masks[:, agent_id],)


    @torch.no_grad()
    def render(self, mode = 'vedio', visualize = False, method='ppo'):
        success_times = 0
        all_episodes_rewards_info = []
        for episode in range(self.all_args.render_episodes):
            self.data_dir = str(self.run_dir / f"episode{episode+1}")
            if not os.path.exists(self.data_dir):
                os.makedirs(self.data_dir)
            agent_file = []
            for i in range(self.robots_num + self.humans_num):
                if i < self.robots_num:
                    agent_file.append(os.path.join(self.data_dir, f"robot{i+1}.txt"))
                else:
                    agent_file.append(os.path.join(self.data_dir, f"human{i-self.robots_num+1}.txt"))

            print('episode:',episode + 1)
            # print('result saved in:',self.data_dir)
            episode_rewards = []
            obs,share_obs = self.envs.reset()

            rnn_states = np.zeros((self.n_rollout_threads,self.robots_num,self.recurrent_N,self.hidden_size,),dtype=np.float32,)
            masks = np.ones((self.n_rollout_threads, self.robots_num, 1), dtype=np.float32)
            episode_rewards_info = []
            decision_time = 0
            for step in range(self.episode_length):
                # print("step",step)
                actions = []
                if method == 'ppo':
                    # write robot obs
                    for i,o in enumerate(share_obs[0:self.robots_num]):
                        with open(agent_file[i], "a") as f:
                            line = f"{step+1}, {', '.join(map(str, o))}\n"
                            f.write(line)
                    # write human obs
                    for i,o in enumerate(np.array(share_obs[self.robots_num:])):
                        with open(agent_file[i+self.robots_num], "a") as f:
                            line = f"{step+1}, {', '.join(map(str, o))}\n"
                            f.write(line)

                    time1 = time.time()
                    for agent_id in range(self.robots_num):
                        self.trainer[agent_id].prep_rollout()
                        #获取action
                        
                        action, rnn_state = self.trainer[agent_id].policy.act(
                            obs[:, agent_id, 0],      # robot_obs
                            obs[:, agent_id, 1:, :self.human_obs_dim],       # humans_obs
                            rnn_states[:, agent_id],
                            masks[:, agent_id],
                            deterministic=True,)
                        action = action.detach().cpu().numpy()
                        action = action[0]
                        actions.append(action)
                        rnn_states[:, agent_id] = _t2n(rnn_state)
                    # Obser reward and next obs

                    obs, share_obs, rewards, dones, reward_info = self.envs.step(actions)  #obs[i][0:1]
                    decision_time += time.time() - time1
                    steps_rewards_info = self.average_dicts(reward_info) # average for multi env
                    episode_rewards_info.append(steps_rewards_info)

                    rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size),dtype=np.float32,)
                    masks = np.ones((self.n_rollout_threads, self.robots_num, 1), dtype=np.float32)
                    masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

                    episode_rewards.append(rewards)

                elif method == 'orca':
                    obs = obs[0]
                    for i,o in enumerate(obs):
                        with open(agent_file[i], "a") as f:
                            line = f"{step+1}, {', '.join(map(str, o))}\n"
                            f.write(line)
                    
                    actions = self.orca_policy.predict(obs)  # (robot_num+human_num * 6)
                    obs = self.envs.step(actions)

                elif method == 'apf':
                    obs = obs[0]
                    actions = self.apf_policy.act(obs)
                    obs = self.envs.step(actions)

            average_decision_time = decision_time / self.episode_length
            print("average decision time:",average_decision_time)
            episode_success,total_used_time = self.envs.render(mode=mode,visualize=visualize)
            average_episode_rewards_info = self.average_dicts(episode_rewards_info) # average for all step in one episode
            print('one episode rewards:', average_episode_rewards_info)
            all_episodes_rewards_info.append(average_episode_rewards_info)
            
            if episode_success == True:
                success_times += 1
            success_rate = success_times / (episode + 1)
            average_used_time = total_used_time / (success_times) if success_times > 0 else 0
            print(f'success rate is:{success_rate}')
            print(f'average used time is:{average_used_time}\n')

            # if method == 'ppo':
            #     episode_rewards = np.array(episode_rewards)
            #     for agent_id in range(self.robots_num):
            #         average_episode_rewards = np.mean(np.sum(episode_rewards[:, :, agent_id], axis=0))
            #         print("eval average episode rewards of agent%i: " % agent_id + str(average_episode_rewards))
        average_all_episodes_rewards_info = self.average_dicts(all_episodes_rewards_info)
        print(f'average all episodes rewards:{average_all_episodes_rewards_info}')
        

        
