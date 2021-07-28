from pypdevs.DEVS import *
from pypdevs.infinity import INFINITY
import gym



# 环境部分
class Gridworld(AtomicDEVS):
    def __init__(self):
        AtomicDEVS.__init__(self, "Gridworld")
        self.state = {}
        self.time_step = 1.0  # 时间步长都取1
        self.inport = self.addInPort("input")
        self.outport = self.addOutPort("output")
        self.isbegin = False

        # 创建gym环境
        self.env = gym.make('CartPole-v0')

    def timeAdvance(self):
        if self.isbegin:
            return self.time_step
        else:
            return INFINITY

    def outputFnc(self):
        return {self.outport: {"content":self.state, "protocol":"inform"}}

    def extTransition(self, inputs):
        agt_state = inputs[self.inport]  # 找到inputs里的对应的字典端口，必是字典格式
        # print(agt_state)
        self.action = agt_state["content"]["action"]
        self.isbegin = True
        if self.action == "initialize": # 区分第一次
            self.state = {"state":self.env.reset()}
        else:
            state, reward, done, _ = self.env.step(self.action)
            self.state = {"state": state, "reward": reward, "done": done}
        return self.state

    def intTransition(self):
        self.isbegin = False
        return self.state
