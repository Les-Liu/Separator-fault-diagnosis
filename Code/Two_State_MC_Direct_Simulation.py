import pandas as pd
import numpy as np
import math
from scipy.stats import weibull_min

"""
    Direct Monte Carlo Simulation
"""

def Load_Data(file_name:str):
    """   
    Reading the scale and shape parameters of a unit whose lifetime obeys a Weibull distribution
    :param file_name: Path to the data file 
    :return: Scale parameters and shape parameters for each unit
    """
    data = pd.read_excel(io="data/" + file_name)
    eta_list = list(data["Scale Parameter"])
    m_list = list(data["Shape Parameter"])

    return eta_list,m_list

def Repair_Rate_Mu(t:float):
    """
    Unit repair rate function
    :param t: Unit repair time, unit h
    :return: Unit repair rate
    """
    Mu = 1/t
    return Mu

def Weibull_Time(eta:float,m:float):
    """
    Updating time using the Weibull distribution
    :param eta: Scale parameter
    :param m: Shape parameter 
    :return: Time to obey the Weibull distribution
    """
    time = weibull_min.rvs(m, loc=0, scale=eta, size=1)[0]
    return np.round(time) 

def Exponential_Time(unit_transfer_rate:float):
    """
    Updating time using an exponential distribution
    :param unit_transfer_rate: Unit transfer rate
    :return: Time to obey the exponential distribution
    """
    time =  - math.log(1-np.random.uniform(0,1))/unit_transfer_rate
    return np.round(time)


def Equal_Ignorance_Abnormals(time_list:list,unit_state_list:list,T:float):
    """
    Use the principle of equivalent ignorance to complete the determination of specific working conditions
    :param time_list: Time list
    :param unit_state_list: Unit status list
    :return: Changes in the state of each cell over time
    """
    # 1 means the valve is leaking and the actuator is fully close;
    # 2 means the valve is blocked and the actuator is fully open.
    unit_total_transition_rate = []
    title = []
    for i in range(len(unit_state_list)):
        unit_total_transition_rate.insert(i,[])
        title.insert(i,"单元{0}".format(i+1))
        # Extract the specific failure state of each unit
        for j in range(len(unit_state_list[i])):
            if unit_state_list[i][j] != 0:
                # Utilisation of the principle of equal ignorance
                Rc = pd.Series(np.random.uniform(0,1))
                result = list(Rc.between(0,0.5))[0]
                if result == True:
                    unit_state_list[i][j] = 1
                elif result == False:
                    unit_state_list[i][j] = 2
            else:
                continue
    
    # Differential time phase and integer
    diff_time = np.around(np.diff(time_list)).astype(np.int64)
    # Count the status of all nodes in the full time period
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

    # Read the failure rate and repair rate of the separator. Poisson distribution
    separator_data = pd.read_excel(io="data/Separator_Unit.xlsx")
    separator_failure_rate = list(separator_data["Failure rate"])[0]
    separator_repair_rate = list(separator_data["Repair rate"])

    # Read the scale and shape parameters of the valve. Weibull distribution
    valve_eta_list,valve_m_list = Load_Data(file_name="Valve_Unit.xlsx")

    # Read actuator failure rates and repair rates. Poisson distribution
    actuator_data = pd.read_excel(io="data/Actuator_Unit.xlsx")
    actuator_failure_rate = list(actuator_data["Failure rate"])
    actuator_repair_rate = list(actuator_data["Repair rate"])
    
    # Get the number of units
    unit_number = len(valve_eta_list) + len(separator_data) + len(actuator_data)
    
    # Provision of time for repairs, unit: h
    repair_time = 2*24

    # Record the repair rate for each valve
    valve_repair_rate_list = []
    for i in range(len(valve_eta_list)):
        valve_repair_rate_list.insert(i,Repair_Rate_Mu(t=repair_time))

    # Initialisation sampling time
    renew_t = 0.0
    # Changing the T-value changes the total sampling time
    T = int(input("Please enter the length of the year you wish to extract:"))
    # Defined sampling time
    sampling_time = T * (365 * 24)
    # Initialise the state of each cell, 
    # with the first 4 indices obeying a Poisson distribution and the rest obeying a Weibull distribution
    unit_state_list = [[0],[0],[0],[0],[0],[0],[0]]
    # Stores the rate of state transfer for each cell
    unit_transition_rate = [[],[],[],[],[],[],[]]
    # Record the time of state transfer
    time_list = [0]
    # Conduct Monte Carlo sampling
    while True:
        # Store each unit: the time it takes to move from the current state, to another state
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

        # Get the smallest element in unit_time_list and its corresponding index
        t_value = min(unit_time_list)
        min_index = unit_time_list.index(t_value)

        # Update time
        renew_t = renew_t + t_value
        # Time of transfer of the storage system
        time_list.append(renew_t)

        if renew_t <= sampling_time:
            # Update the status of each unit
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

    # Using the principle of equivalent ignorance, 
    # determine the specific states of the failed units and output the specifics of the change 
    # in state of each unit over time
    Equal_Ignorance_Abnormals(time_list=time_list,unit_state_list=unit_state_list,T=T)

    









   
        
    
        


        

        
        




    





