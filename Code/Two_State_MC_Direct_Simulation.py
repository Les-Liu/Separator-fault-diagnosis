import pandas as pd
import numpy as np
import math
from scipy.stats import weibull_min

"""
    直接蒙特卡洛模拟
"""

def Load_Data(file_name:str):
    """   
    读取寿命服从威布尔分布的单元的尺度参数和形状参数
    :param file_name: 数据所在文件名称 文件为excle类型
    :return: 每个单元的尺度参数和形状参数
    """
    data = pd.read_excel(io="data/" + file_name)
    eta_list = list(data["Scale Parameter"])
    m_list = list(data["Shape Parameter"])

    return eta_list,m_list

def Repair_Rate_Mu(t:float):
    """
    单元修复率函数
    :param t: 单元修复时间，单位h
    :return: 单元修复率
    """
    Mu = 1/t
    return Mu

def Weibull_Time(eta:float,m:float):
    """
    使用威布尔分布对时间进行更新
    :param eta: 尺度参数
    :param m: 形状参数 
    :return: 服从威布尔分布的时间
    """
    time = weibull_min.rvs(m, loc=0, scale=eta, size=1)[0]
    return np.round(time) 

def Exponential_Time(unit_transfer_rate:float):
    """
    使用指数分布对时间进行更新
    :param unit_transfer_rate: 单元转移速率
    :return: 服从指数分布的时间
    """
    time =  - math.log(1-np.random.uniform(0,1))/unit_transfer_rate
    return np.round(time)


def Equal_Ignorance_Abnormals(time_list:list,unit_state_list:list,T:float):
    """
    使用同等无知原则，来完成具体工况的确定
    :param time_list: 时间列表
    :param unit_state_list: 单元状态列表
    :return: 每个单元的状态随时间的变化的情况
    """
    # 1表示阀门泄漏，执行器全关，2表示阀门堵塞，执行器全开
    unit_total_transition_rate = []
    title = []
    for i in range(len(unit_state_list)):
        unit_total_transition_rate.insert(i,[])
        title.insert(i,"单元{0}".format(i+1))
        # 抽取每个单元具体的失效状态
        for j in range(len(unit_state_list[i])):
            if unit_state_list[i][j] != 0:
                # 利用同等无知原则
                Rc = pd.Series(np.random.uniform(0,1))
                result = list(Rc.between(0,0.5))[0]
                if result == True:
                    unit_state_list[i][j] = 1
                elif result == False:
                    unit_state_list[i][j] = 2
            else:
                continue
    
    # 差分时间相，并取整数
    diff_time = np.around(np.diff(time_list)).astype(np.int64)
    # 统计出全时间段内所有节点的状态
    unit_state = np.array(unit_state_list).transpose()
    unit_status_statistics = [] 

    for i in range(unit_state.shape[0]):
        for j in range(diff_time[i]):
            unit_status_statistics.insert(j,list(unit_state[i,:]))

    unit_state_list = pd.DataFrame(data=np.array(unit_state_list).transpose(),columns=title)
    unit_state_list.to_excel("result/Direct Simulation/{0}_mc_unit_state.xlsx".format(T),index=False)
    result = pd.DataFrame(data=np.array(unit_status_statistics),columns=title)
    result.to_excel("result/Direct Simulation/{0}_mc_result.xlsx".format(T),index=False)


if __name__ == "__main__":

    # 读取分离器的失效率和维修率 泊松分布
    separator_data = pd.read_excel(io="data/Separator_Unit.xlsx")
    separator_failure_rate = list(separator_data["Failure rate"])[0]
    separator_repair_rate = list(separator_data["Repair rate"])

    # # 读取阀门的尺度参数和形状参数 威布尔分布
    # valve_eta_list,valve_m_list = Load_Data(file_name="Valve_Unit.xlsx")

    # # 读取执行器的失效率和维修率 泊松分布
    # actuator_data = pd.read_excel(io="data/Actuator_Unit.xlsx")
    # actuator_failure_rate = list(actuator_data["Failure rate"])
    # actuator_repair_rate = list(actuator_data["Repair rate"])
    
    # 获取单元个数
    # unit_number = len(valve_eta_list) + len(separator_data) + len(actuator_data)
    unit_number = len(separator_data)

    # 规定修理时间，单位：h
    repair_time = 2*24

    # 记录每个阀门的维修率
    valve_repair_rate_list = []
    for i in range(len(valve_eta_list)):
        valve_repair_rate_list.insert(i,Repair_Rate_Mu(t=repair_time))

    # 初始化抽样时间
    renew_t = 0.0
    # 更改T值改变抽样总时长
    T = int(input("请输入想要抽取的年份时长:"))
    # 规定抽样时间
    sampling_time = T * (365 * 24)
    # 初始化每个单元的状态，前4个索引服从泊松分布，剩下的服从威布尔分布
    unit_state_list = [[0],[0],[0],[0],[0],[0],[0]]
    # 储存每个单元的状态转移速率
    unit_transition_rate = [[],[],[],[],[],[],[]]
    # 记录状态转移的时间
    time_list = [0]
    # 进行蒙特卡洛抽样
    while True:
        # 储存每个单元：从当前状态，转移到另一个状态所需要的时间
        unit_time_list = []
        for i in range(unit_number):
            if i == 0:
                if unit_state_list[i][-1] == 0:
                    t = Exponential_Time(unit_transfer_rate=separator_failure_rate)
                elif unit_state_list[i][-1] == 1:
                    t = Exponential_Time(unit_transfer_rate=separator_repair_rate[0])
            elif (1 <= i) and (i <= 3):
                if unit_state_list[i][-1] == 0:
                    t = Exponential_Time(unit_transfer_rate=actuator_failure_rate[i-1])
                elif unit_state_list[i][-1] == 1:
                    t = Exponential_Time(unit_transfer_rate=actuator_repair_rate[i-1])
            elif (4 <= i) and (i <= 6):
                if unit_state_list[i][-1] == 0:
                    t = Weibull_Time(eta=valve_eta_list[i-4],m=valve_m_list[i-4])
                elif unit_state_list[i][-1] == 1:
                    t = Exponential_Time(unit_transfer_rate=Repair_Rate_Mu(t=repair_time))
            unit_time_list.insert(i,t)

        # 获取unit_time_list中最小的元素和其对应的索引
        t_value = min(unit_time_list)
        min_index = unit_time_list.index(t_value)

        # 更新时间
        renew_t = renew_t + t_value
        # 储存系统发生转移的时间
        time_list.append(renew_t)

        if renew_t <= sampling_time:
            # 更新每个单元的状态
            for i in range(unit_number):
                if i == min_index:
                    print("State transition occurs in unit {0}.".format(i+1))
                    if unit_state_list[i][-1] == 0:
                        unit_state_list[i].append(1)
                    elif unit_state_list[i][-1] == 1:
                        unit_state_list[i].append(0)
                else:
                    print("Unit {0} has not undergone a transfer.".format(i+1))
                    unit_state_list[i].append(unit_state_list[i][-1])
        else:
            print("Program Finish！！！")
            break

    # 使用同等无知原则，确定失效单元的具体状态，并输出每一个单元状态随时间变化的具体情况
    Equal_Ignorance_Abnormals(time_list=time_list,unit_state_list=unit_state_list,T=T)

    









   
        
    
        


        

        
        




    





