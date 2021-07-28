from pypdevs.DEVS import *
from pypdevs.infinity import INFINITY
import agent_BDI_REINFORCE_paper.BDI_Data as data
from torch.distributions import Bernoulli
import timeit

import torch


# Goal部分
class Goal(AtomicDEVS):
    def __init__(self):
        AtomicDEVS.__init__(self, "Goal")
        # 初始化从(0,0)位置开始
        self.inputs = {"plans_return":[False, False, False, False], "perception":{'ways': [(1, 0), (0, 1)], 'target1': [], 'target2': []}}
        # goal的端口只与intention进行交互
        self.outport = self.addOutPort("outport")  # 增加端口
        self.inport = self.addInPort("inport")  # 增加端口
        self.isbegin = True
        with open("history_.txt", "w", encoding="utf-8") as f:
            pass
        # 状态state
        self.state = (0, 0)
        # policy π goal只负责更新policy，经过一个goal更新一次policy
        self.policy = data.PolicyNetwork().to(data.device)
        self.optim = data.optim(self.policy.parameters(), lr=1e-4)
        # torch.save(self.policy.state_dict(), 'policy_parameters.pkl') # 初始必须存储一下策略，以给plan公
        # 计数
        self.episode = 0

    def timeAdvance(self):  # ta 要的是返回值
        if self.episode == 5000:  # 500次的迭代
            return INFINITY
        if self.isbegin:
            return 1.0
        else:
            return INFINITY

    # 输出
    def outputFnc(self):  # ta 要的是返回值
        try:
            plans, parameters = self._decide_plan()
        except KeyError:
            parameters = [self.policy]  # 网络初始化
            plans = ["plan1"]
        self.state = parameters
        self.isbegin = False
        if plans == "over":
            return {}  # 将程序锁死
        # 开始正常输出 plans parameters state
        # print({"plans":plans, "parameters":parameters, "state": self.state})
        return {self.outport:{"plans":plans, "parameters":self.state}}

    def intTransition(self):  # ta 要的是返回值
        return self.state

    # 外部事件转移函数
    def extTransition(self, inputs):
        # 获取intention的输出，从一时刻起，而不是零时刻
        # with open("time_advance.txt", "a", encoding="utf-8") as f:
        #     print("goal")
        self.episode += 1  # 走了一幕
        self.inputs = inputs[self.inport]
        # print(self.inputs)
        self.isbegin = True
        return self.state

    # 返回一个列表
    def _decide_plan(self):
        # print("目前正在进行：", self.episode)
        # print(timeit.default_timer())
        # 判断智能体是否已经完成了自己的目标
        is_shutdown = self._is_all_ture(self.inputs["plans_return"])
        if is_shutdown:
            return "over", ""
        # 开始选择计划
        self.deal_plan_list = []  # 用于存储输出的plan
        self.plan_parameters = []  # 用于存储plan相关的参数

        perception = self.inputs["perception"]

        # 获取perception中的参数
        rewards = perception["rewards"]
        actions = perception["actions"]
        states = perception["states"]
        episode_reward = perception["episode_reward"]
        # print(len(rewards))
        # 更新网络参数
        R = 0.
        # 参与损失计算， 参考 《Reinforcement Learning: An Introduction》 13.3 章节
        _gamma = torch.zeros((len(rewards), 1)).to(data.device)
        # 根据折扣，计算累计回报 G_t
        for i in reversed(range(len(rewards))):
            R = data.gamma * R + rewards[i]
            rewards[i] = R
            _gamma[i] = torch.FloatTensor([data.gamma]).to(data.device).pow(i)
        # 将 list 转成 tensor
        states_tensor = torch.FloatTensor(states).to(data.device)
        actions_tensor = torch.FloatTensor(actions).unsqueeze(1).to(data.device)
        rewards_tensor = torch.FloatTensor(rewards).unsqueeze(1).to(data.device)
        # 前向计算
        # print(len(states))
        # print(len(actions))
        # print(len(rewards))
        prob = self.policy(states_tensor)
        b = Bernoulli(prob)
        # log_prob 是分布提供的方法，可以计算 action 在分布中的对数概率
        log_prob = b.log_prob(actions_tensor)
        # REINFORCE 损失计算
        loss = - log_prob * rewards_tensor * _gamma
        loss = loss.mean()
        # 反向传播和参数更新
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        if self.episode % 10 == 0:  # 10步一打印，10步一存储
            print('Episode:{}, episode reward is {}'.format(self.episode, episode_reward))
            with open("history.txt", "a", encoding="utf-8") as f:
                print(self.episode, episode_reward, loss.cpu(), file=f)

            # torch.save(self.policy.state_dict(), 'policy_parameters.pkl')  # 初始必须存储一下策略，以给plan公用
        # 传递一下策略试一下
        self.deal_plan_list.append("plan1")  # 在这个问题上目前只规划了一个plan试一下
        self.plan_parameters.append(self.policy)
        return self.deal_plan_list, self.plan_parameters

    # 判断plan的返回值是不是为空,一个为假就返回假
    def _is_all_ture(self, bool_list):
        temp = True
        for eve in bool_list:
            if eve == False:
                temp = False
                break
        if temp == True:
            return True
        else:
            return False

