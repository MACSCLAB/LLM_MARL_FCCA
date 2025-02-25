import sys
import os

DEPLOY_DIR = os.getcwd()
LLM_MARL_ROOT_DIR = f"{DEPLOY_DIR}/../../../.."
# os.system(f"export PYTHONPATH={LLM_MARL_ROOT_DIR}:$PYTHONPATH")
# os.system("echo $PYTHONPATH")
# print(f"Command exited with status {exit_status}")
# sys.path.append('../../../..')
os.environ["PYTHONPATH"] = f"../../../..:{os.environ.get('PYTHONPATH', '')}"

print(os.environ["PYTHONPATH"])

from runner.separated.base_runner import Runner