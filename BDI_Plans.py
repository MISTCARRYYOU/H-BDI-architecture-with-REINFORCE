from pypdevs.DEVS import *
from pypdevs.infinity import INFINITY
from pypdevs.simulator import Simulator
import random
import agent_BDI_REINFORCE_paper.BDI_Data as data
import torch


# plan 部分
class Plan1(AtomicDEVS):
    def __init__(self):
        AtomicDEVS.__init__(self, "Plan1")
        self.outport_dispatch = self.addOutPort("outport_dispatch")  # 增加端口
        self.inport_dispatch = self.addInPort("inport_dispatch")  # 增加端口

        self.outport_interaction = self.addOutPort("outport_interaction")  # 增加端口
        self.inport_interaction = self.addInPort("inport_interaction")  # 增加端口

        self.isbegin = False
        self.choose_output = "interaction"  # interaction或者dispatch
        self.state = None
        self.plan_return = False  # 永动机模式
        self.name = "plan1"
        self.time_step = 0
        # 记录 episode 数据  R A S
        self.states = []
        self.rewards = []
        self.actions = []
        self.done = False
        self.requirements = {}

    def timeAdvance(self):  # ta 要的是返回值
        if self.isbegin:
            return 1.0
        else:
            return INFINITY

    # 输出
    def outputFnc(self):  # ta 要的是返回值
        if self.choose_output == "interaction":
            return {self.outport_interaction: {"requirements":self.requirements, "planID":self.name}}  # self.state 为当前选择的位置
        elif self.choose_output == "dispatch":
            return {self.outport_dispatch: {"plan_return":self.plan_return, "perception":self.perceptions}}  # {"plan_return":, "perception":}

    def intTransition(self):  # ta 要的是返回值
        self.isbegin = False
        return self.state

    # 外部事件转移函数
    def extTransition(self, inputs):
        # 这两个没有公共的部分
        try:  # 收到dispatch的输入
            # 输入：{"planID":self.current_plan, "parameters":self.current_parameter}
            self.inputs_dispatch = inputs[self.inport_dispatch]
            self.choose_output = "interaction"
            self.time_step = 0  # 时间步长归零
            self.episode_reward = 0  # 与时间步长同时清零
            self.states = []
            self.rewards = []
            self.actions = []
            self.policy = self.inputs_dispatch["parameters"]

            self.requirements["action"] = "initialize"  # 代表初始化动作

            if self.name == self.inputs_dispatch["planID"]: # 证明这次发送针对它
                self.isbegin = True
            else:
                self.isbegin = False
            return self.state

        except KeyError:  # 收到interaction的输入

            # 接受来自交互模块的内容
            self.inputs_interaction = inputs[self.inport_interaction]  # {planID":..,"perception":[“state”:,"reward:","done":]}
            # print(self.inputs_interaction)
            self.perception = self.inputs_interaction["perception"]

            if self.time_step == 100 or self.done:
                # print(66666)
                self.choose_output = "dispatch"
                self.perceptions = {}
                self.perceptions["states"] = self.states
                self.perceptions["actions"] = self.actions
                self.perceptions["rewards"] = self.rewards
                self.perceptions["episode_reward"] = self.episode_reward
                self.done = False
            else:
                # print(self.time_step)
                self.choose_output = "interaction"
                self.state = self.perception["state"]
                state = torch.FloatTensor(self.state).unsqueeze(0).to(data.device)
                action = int(self.policy.select_action(state))  # 根据状态选择动作，在线学习的本质在这一句话里！！！！！！！！！！
                if self.time_step != 0:  # 代表接受初始化的消息
                    # 拆分来自交互模块的内容
                    # 分两种情况获取信息，第一次和其他
                    # if self.time_step == 0: # 第一次需要只获取state
                    #     state = self.perception["state"]
                    #     self.requirements["action"] = "initialize" # 代表初始化动作
                    # else:
                    self.states.append(self.state)

                    # print(state)  tensor([[-0.0372, -0.3707,  0.0408,  0.5346]], device='cuda:0')
                    self.actions.append(action)
                    # 获取交互的信息
                    # print(self.perception)

                    self.state = self.perception["state"]

                    reward = self.perception["reward"]
                    self.done = self.perception["done"]  # 交互部分！！！！！
                    self.episode_reward += reward  # 蒙特卡罗式的奖励收集
                    self.rewards.append(reward)
                self.time_step += 1  # 完成了一步交互
                self.requirements["action"] = action
            # 不用管这个部分
            if self.name == self.inputs_interaction["planID"]:  # 证明这次发送针对它
                self.isbegin = True
            else:
                self.isbegin = False

            return self.state



