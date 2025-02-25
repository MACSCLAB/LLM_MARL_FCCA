import numpy as np
import torch
from multiprocessing import Process, Pipe
from abc import ABC, abstractmethod


class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize(序列化) contents (otherwise multiprocessing tries to use pickle)
    """

    def __init__(self, x):
        self.x = x

    def __getstate__(self):   #序列化
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):    #反序列化
        import pickle
        self.x = pickle.loads(ob)

#公用的矢量环境，定义的是一个抽象类，只有写完其中的方法才能被实例化
class ShareVecEnv(ABC):
    """
    An abstract asynchronous, vectorized environment.
    Used to batch data from multiple copies of an environment, so that
    each observation becomes an batch of observations, and expected action is a batch of actions to
    be applied per-environment.
    """
    closed = False
    viewer = None

    metadata = {
        'render.modes': ['human', 'rgb_array']
    }

    def __init__(self, num_envs, robot_observation_space, human_observation_space, share_observation_space, action_space):
        self.num_envs = num_envs
        self.robot_observation_space = robot_observation_space
        self.human_observation_space = human_observation_space
        self.share_observation_space = share_observation_space
        self.action_space = action_space

    @abstractmethod
    def reset(self):
        """
        Reset all the environments and return an array of
        observations, or a dict of observation arrays.

        If step_async is still doing work, that work will
        be cancelled and step_wait() should not be called
        until step_async() is invoked again.
        """
        pass

    @abstractmethod
    def step_async(self, actions):
        """
        Tell all the environments to start taking a step
        with the given actions.
        Call step_wait() to get the results of the step.

        You should not call this if a step_async run is
        already pending.
        """
        pass

    @abstractmethod
    def step_wait(self):
        """
        Wait for the step taken with step_async().

        Returns (obs, rews, dones, reward_infos):
         - obs: an array of observations, or a dict of
                arrays of observations.
         - rews: an array of rewards
         - dones: an array of "episode done" booleans
         - reward_infos: a sequence of reward_info objects
        """
        pass

    def close_extras(self):
        """
        Clean up the  extra resources, beyond what's in this base class.
        Only runs when not self.closed.
        """
        pass

    def close(self):
        if self.closed:
            return
        if self.viewer is not None:
            self.viewer.close()
        self.close_extras()
        self.closed = True

    def step(self, actions):
        """
        Step the environments synchronously.

        This is available for backwards compatibility.
        """
        self.step_async(actions)
        return self.step_wait()

    def render(self, mode='human'):
        # imgs = self.get_images()
        # bigimg = tile_images(imgs)
        # if mode == 'human':
        #     self.get_viewer().imshow(bigimg)
        #     return self.get_viewer().isopen
        # elif mode == 'rgb_array':
        #     return bigimg
        # else:
        #     raise NotImplementedError
        pass

    def get_images(self):
        """
        Return RGB images from each environment
        """
        raise NotImplementedError

    # @property
    # def unwrapped(self):
    #     if isinstance(self, VecEnvWrapper):
    #         return self.venv.unwrapped
    #     else:
    #         return self

    def get_viewer(self):
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.SimpleImageViewer()
        return self.viewer


def worker(remote, parent_remote, env_fn_wrapper):
    #remote是在工作的环境
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()    #在主进程接收数据
        if cmd == 'step':
            ob, share_obs, reward, done, reward_info = env.step(data)
            if 'bool' in done.__class__.__name__:
                if done:
                    obs,share_obs = env.reset()
            else:
                if np.all(done):
                    # print(done)
                    obs,share_obs = env.reset()
                    # print('reset')

            remote.send((ob, share_obs, reward, done, reward_info))
        elif cmd == 'reset':
            obs,share_obs = env.reset()
            remote.send((obs,share_obs))
        elif cmd == 'render':
            if data == "rgb_array":
                fr = env.render(mode=data)
                remote.send(fr)
            elif data == "human":
                env.render(mode=data)
        elif cmd == 'close':    #暂时没用
            env.close()
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((env.robot_observation_space, env.human_observation_space, env.share_observation_space, env.action_space))
        else:
            raise NotImplementedError


class SubprocVecEnv(ShareVecEnv):   #多线程环境，一般用于训练
    def __init__(self, env_fns, spaces=None):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))      
                   for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)] 
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        robot_observation_space, human_observation_space, share_observation_space, action_space = self.remotes[0].recv()
        ShareVecEnv.__init__(self, len(env_fns), robot_observation_space, human_observation_space, share_observation_space, action_space)

    def step_async(self, actions):
        #把action发送到各个线程的环境
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True  #等待各个线程的环境处理

    def step_wait(self):
        #处理完成后接收结果
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, share_obs, rews, dones, reward_infos = zip(*results)
        return np.stack(obs), np.stack(share_obs), np.stack(rews), np.stack(dones), reward_infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        results = [remote.recv() for remote in self.remotes]
        obs, share_obs = zip(*results)
        return np.stack(obs),np.stack(share_obs)

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()   #等待子进程结束并堵塞主进程
        self.closed = True

    def step(self, actions):    #和抽象类的step是一样的，写在这里方便看
        self.step_async(actions)
        return self.step_wait()
  

class DummyVecEnv(ShareVecEnv):     #单线程环境，一般用于验证和测试
    def __init__(self, env_fns, args):
        self.envs = [fn() for fn in env_fns]
        self.env = self.envs[0]
        ShareVecEnv.__init__(self, len(env_fns), self.env.robot_observation_space, self.env.human_observation_space, self.env.share_observation_space, self.env.action_space)
        self.actions = None
        self.method = args.method

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        if self.method == 'ppo':
            results = self.envs[0].step(self.actions)
            obs, share_obs, rews, dones, reward_infos = map(np.array, zip(results))   #map用于将函数变成指定形式
            self.actions = None
            return obs, share_obs, rews, dones, reward_infos
        elif self.method == 'orca' or self.method == 'apf':
            obs = self.envs[0].step(self.actions)
            self.actions = None
            return obs

    def reset(self):
        obs, share_obs = self.env.reset() # to ensure that the shape even with the env above
        obs = np.expand_dims(obs,axis=0)
        share_obs = np.expand_dims(obs,axis=0)
        return obs, share_obs

    def close(self):
        for env in self.envs:
            env.close()
    
    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def render(self, mode="vedio", visualize=False):
        episode_success,total_used_time = self.envs[0].render(mode=mode,visualize=visualize)
        return episode_success,total_used_time
