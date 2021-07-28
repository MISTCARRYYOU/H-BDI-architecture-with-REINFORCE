from pypdevs.DEVS import *
from pypdevs.infinity import INFINITY


# 交互部分
class Dispatch(AtomicDEVS):
    def __init__(self):
        AtomicDEVS.__init__(self, "Dispatch")
        self.outport_goal = self.addOutPort("outport_goal")  # 增加端口
        self.inport_goal = self.addInPort("inport_goal")  # 增加端口
        self.outport_plan = self.addOutPort("outport_plan")  # 增加端口
        self.inport_plan = self.addInPort("inport_plan")  # 增加端口
        self.isbegin = False  # 初始为静止
        self.choose_output = "plan"  # 两个值，plan和goal
        self.state = None
        self.plans_return = []

    def timeAdvance(self):  # ta要的是返回值
        if self.isbegin:
            return 1.0
        else:
            return INFINITY

    # 输出
    def outputFnc(self):  # ta 要的是返回值
        if self.choose_output == "plan":
            return {self.outport_plan: {"planID":self.current_plan, "parameters":self.current_parameter}}
        elif self.choose_output == "goal":
            return {self.outport_goal: self.state}
        else:
            raise EOFError

    def intTransition(self):  # ta 要的是返回值
        self.isbegin = False
        return self.state

    # 外部事件转移函数
    # 接受来自goal的信息：self.outport:{"plans":plans, "parameters":parameters, "state": self.state}
    def extTransition(self, inputs):
        self.isbegin = True  # 无论谁给调度模块来消息，都是开始
        # 下面分别对两种外部转移函数的模式进行处理
        # 1-goal  2-plan
        try: # goal
            self.inputs_goal = inputs[self.inport_goal]  # 是一个包含有三个元素的字典
            self.choose_output = "plan"
            # 下面对于plans里面的每一个计划都进行一个调度过程
            self.choose_plan = 0
            self.plans_return = []
        except KeyError:  # plan
            # print(self.choose_plan)
            # print(self.inputs_goal)
            temp = inputs[self.inport_plan]  # self.state = {"plan_return":, "perception":}
            self.plans_return.append(temp["plan_return"])
            self.state = {"plans_return":self.plans_return, "perception": temp["perception"]}
            # 判断下一步的内容
            if self.choose_plan == len(self.inputs_goal["plans"])-1: # 代表上一个已经完成了最后的plan调度
                self.choose_output = "goal"
                return self.state
            else:
                self.choose_output = "plan"
            self.choose_plan += 1  #开始执行下一个计划
        # 这个部分是公共的部分，意义是指定下一次输出的plan的类型和内容
        # print(self.inputs_goal)
        self.current_plan = self.inputs_goal["plans"][self.choose_plan]  # 一个字符串格式的
        self.current_parameter = self.inputs_goal["parameters"][self.choose_plan]  # [(1,1),(2,2)]
        return self.state
