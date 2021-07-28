from pypdevs.DEVS import *
from pypdevs.infinity import INFINITY


# 交互部分
# 在grid-world的例子中，这个部分的交互过程显得有些多余，起一个端口整合和归纳的作用，为后续的协议留出基础
# 注意，agent代表的是整个智能体的，其端口应该与环境进行连接
class Interaction(AtomicDEVS):
    def __init__(self):
        AtomicDEVS.__init__(self, "Interaction")

        self.outport_plan = self.addOutPort("outport_plan")  # 增加端口
        self.inport_plan = self.addInPort("inport_plan")  # 增加端口
        self.outport_agent = self.addOutPort("outport_agent")  # 增加端口
        self.inport_agent = self.addInPort("inport_agent")  # 增加端口

        # 变量
        self.state = None
        self.isbegin = False
        self.choose_output = "agent"  # agent和plan两个接口


    def timeAdvance(self):  # ta 要的是返回值
        if self.isbegin:
            return 1.0
        else:
            return INFINITY

    # 输出
    def outputFnc(self):  # ta 要的是返回值
        if self.choose_output == "agent":
            # print(self.inputs_plan)
            return {self.outport_agent: {"content": self.state, "protocol": "inform"}}  # self.state 为当前选择的位置
        elif self.choose_output == "plan":
            # print(self.inputs_agent)
            return {self.outport_plan: {"planID": self.inputs_plan["planID"], "perception": self.state}}  # {"plan_return":, "perception":}

    def intTransition(self):  # ta 要的是返回值
        self.isbegin = False
        return self.state

    # 外部事件转移函数
    # 默认为inform协议什么也不需要做
    def extTransition(self, inputs):
        # 处理两个端口
        try:  # 来自plan的输入
            self.isbegin = True
            # 输入：{"state":self.state, "planID":self.name}
            self.inputs_plan = inputs[self.inport_plan]
            self.choose_output = "agent"
            self.state = {"action":self.inputs_plan["requirements"]["action"]}  # 没有太多实际意义的一个指令
            return self.state
        except KeyError:  # 收到agent的输入
            # print(inputs)
            self.inputs_agent = inputs[self.inport_agent]  # {"perception":[(),(),...]}
            # print(self.inputs_agent)
            self.state = self.inputs_agent["content"]
            self.isbegin = True
            self.choose_output = "plan"
            return self.state
