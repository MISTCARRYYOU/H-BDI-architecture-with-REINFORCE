from pypdevs.DEVS import *
from pypdevs.infinity import INFINITY
from pypdevs.simulator import Simulator
import agent_BDI_REINFORCE_paper.BDI_Plans as Plans
from agent_BDI_REINFORCE_paper.BDI_Goal import Goal
from agent_BDI_REINFORCE_paper.BDI_Interaction import Interaction
from agent_BDI_REINFORCE_paper.BDI_Dispatch import Dispatch
from agent_BDI_REINFORCE_paper.BDI_Environment import Gridworld
import timeit

# 环境依然是grid world背景
# 智能体依然是每一个步长只能感知到其位置上下左右四个位置的地方
# 依然设置那两个障碍物，一个是想去的，一个是不想去的
#         |   plan                 |   触发条件   goal              |
#         |  1 朝着奖励障碍物走策略   |  有奖励障碍物                  |
#         |  2 避开惩罚障碍物走策略   |  没有奖励障碍物，但是有惩罚障碍物 |
#         |  3 随机行走策略          |  没有奖励障碍物，也没有惩罚障碍物 |

with open("time_advance.txt", "w", encoding="utf-8") as f:
    pass
# 大的仿真耦合模型
class Simulation(CoupledDEVS):
    def __init__(self):
        CoupledDEVS.__init__(self, "Simulation")
        # 创建子模型
        self.goal = self.addSubModel(Goal())
        self.dispatch = self.addSubModel(Dispatch())
        self.plan1 = self.addSubModel(Plans.Plan1())
        self.interaction = self.addSubModel(Interaction())
        self.gridworld = self.addSubModel(Gridworld())
        # 连接goal与dispatch
        self.connectPorts(self.goal.outport, self.dispatch.inport_goal)
        self.connectPorts(self.dispatch.outport_goal, self.goal.inport)
        # 连接dispatch与plan
        self.connectPorts(self.dispatch.outport_plan, self.plan1.inport_dispatch)
        self.connectPorts(self.plan1.outport_dispatch, self.dispatch.inport_plan)
        # 连接plan与interaction
        self.connectPorts(self.plan1.outport_interaction, self.interaction.inport_plan)
        self.connectPorts(self.interaction.outport_plan, self.plan1.inport_interaction)
        # 连接interaction与environment
        self.connectPorts(self.interaction.outport_agent, self.gridworld.inport)
        self.connectPorts(self.gridworld.outport, self.interaction.inport_agent)


a = timeit.default_timer()
# 3-进行仿真
model = Simulation()
# print(model.generator.state)
sim = Simulator(model)

sim.setClassicDEVS()
sim.setVerbose("log")
# Required to set Classic DEVS, as we simulate in Parallel DEVS otherwise

# sim.setTerminationTime(150)
sim.simulate()
# print(model.generator.state)
b = timeit.default_timer()
print(b-a)  # 时间51s
