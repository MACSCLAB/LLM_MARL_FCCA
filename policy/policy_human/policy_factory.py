from policy.policy_human.orca import ORCA
from policy.policy_human.lstm_lux import Lstm_RL_LUX
# from policy.policy_human.orca_robot import ORCA_ROBOT
from policy.policy_human.multi_rl import Multi_RL

def none_policy():
    return None

policy_factory = dict()
policy_factory['orca'] = ORCA
# policy_factory['orca_robot'] = ORCA_ROBOT
policy_factory['none'] = none_policy
policy_factory['lstm_rl'] = Lstm_RL_LUX
policy_factory['multi_rl'] = Multi_RL
