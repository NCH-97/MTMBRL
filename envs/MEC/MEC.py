import sys,os
from numba.typed.listobject import new_list

sys.path.append('/usr/share/sumo/tools')
import numpy as np
import math
import traci
import re
from gym import spaces
from torch.cuda import seed_all
sys.path.append(
    os.path.abspath(os.path.join(__file__, "../../.."))
)
from envs.MEC.UAV import UAV
from scipy.stats import nakagami
from scipy.stats import gamma
from scipy.special import gamma as GammaFunc
from scipy.special import gammaincc
from torch import distributions as torchd
import torch

def dbm_to_w(dbm):
    return 10 ** (dbm / 10) / 1000

def w_to_dbm(watt):
    return 10 * np.log10(watt * 1000)

class MEC:
    def __init__(self, Max_step=150, Max_vehicales=30, SEED=1):
        #ENV
        self.uav = UAV()
        self.step = 0
        self.Max_step = Max_step
        self.Max_vehicales = Max_vehicales
        #COMP
        self.posibility_task = 0.8
        self.task_mean = 50
        self.task_std = 10
        self.outpop = 0
        #COMM
        self.vehicle_transP_dBm = 20 # 100mW
        self.c_mps = 3e8
        self.f_Hz = 5e9
        self.Gt = self.Gr = 1
        self.m_max = 2
        self.m_min = 0.5
        self.m_func = lambda d,h,a=-0.01,b=0.01: min(max(self.m_min, 2*math.exp(-b*math.sqrt(d**2-h**2))+0.5*math.exp(-2*math.atan(a*h/d))), self.m_max)
         # Nakagami-m fading 参数
         # d: distance  h: UAV height  b: environment parameter
        self.m = 1
        self.min_recive_dBm = -95
        self.noise_dBm = -105
        self.SEED = SEED
        np.random.seed(self.SEED)
        torch.manual_seed(self.SEED)
        self.BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.SUMO_CFG = os.path.abspath(
            os.path.join(self.BASE_DIR, "./TEST/SUMO_TEST/HG/grid.sumocfg")
        )
        # print("Using SUMO config:", self.SUMO_CFG)
        # print("Exists:", os.path.exists(self.SUMO_CFG))
        traci.start(["sumo", "-c", self.SUMO_CFG])
        # print("Starting SUMO simulation with TraCI...")
        

    def _observation_space(self):
        ENV_bound = traci.simulation.getNetBoundary()
        x_min = ENV_bound[0][0]
        x_max = ENV_bound[1][0]
        y_min = ENV_bound[0][1]
        y_max = ENV_bound[1][1]
        UAV_pos_space = spaces.Box(low=np.array([x_min, y_min, 40]), high=np.array([x_max, y_max, 60]))
        cars_state = spaces.Box(low=np.array([0, x_min-1.6, y_min-1.6, -105, 0, 0, 0, 0, 0]*self.Max_vehicales),
                                high=np.array([self.Max_vehicales, x_max+1.6, y_max+1.6, 20, 100, 100, 100, self.Max_step, self.Max_step]*self.Max_vehicales))
        self.obs_space = spaces.Box(
            low=np.concatenate([UAV_pos_space.low, cars_state.low]),
            high=np.concatenate([UAV_pos_space.high, cars_state.high]),
            dtype=np.float32
        )
        print(f"Observation space shape: {self.obs_space.shape}")
        return spaces.Dict({
            "obs": self.obs_space,
        })

    def _action_space(self):
        pos = traci.simulation.getNetBoundary()
        UAV_pos_space = spaces.Box(low=np.array([pos[0][0], pos[0][1], 40]), high=np.array([pos[1][0], pos[1][1], 60]))
        cars_trans_space = spaces.Box(low=np.full(30, 0), high=np.full(30, 1))
        self.action_space = spaces.Box(
            low=np.concatenate([UAV_pos_space.low, cars_trans_space.low]),
            high=np.concatenate([UAV_pos_space.high, cars_trans_space.high])
        )
        return self.action_space
    
    def _communication_fading(self, UAV_position=(0,0,0), car_position=(0,0), vehicle_transP_dBm=20):

        def free_space_path_loss(pt, frequency, distance):
            wavelength = self.c_mps / frequency
            return pt * self.Gt * self.Gr / ((4 * np.pi * distance / wavelength) ** 2)
        
        distance = math.sqrt(
            (UAV_position[0] - car_position[0]) ** 2 +
            (UAV_position[1] - car_position[1]) ** 2 +
            (UAV_position[2] - 0) ** 2
        )

        Pt = dbm_to_w(vehicle_transP_dBm)
        Pr_no_fading = free_space_path_loss(Pt, self.f_Hz, distance)  # 接收功率-LF
        self.m = self.m_func(distance, UAV_position[2])
        cdf_value = gamma.cdf(self.uav.recive_bound_dBm, a=self.m, scale=(Pr_no_fading / self.m)) # CDF
        coverage_probability = 1 - cdf_value
        z = self.m * dbm_to_w(self.uav.recive_bound_dBm) / Pr_no_fading
        Gamma_m_z = GammaFunc(self.m)*gammaincc(self.m, z)
        Gamma_m1_z = GammaFunc(self.m+1)*gammaincc(self.m+1, z)
        E_pr_cond = Pr_no_fading * (Gamma_m1_z / Gamma_m_z) * (1 / self.m)
        Pr_with_fading = E_pr_cond
        # fading = nakagami.rvs(self.m, scale=np.sqrt(1 / self.m), size=100)
        # Pr_with_fading = Pr_no_fading * np.mean(fading) ** 2

        return {
            "Pt_W": Pt,
            "Pt_dBm": w_to_dbm(Pt),
            "Distance": distance,
            "Free_space_path_loss_W": Pr_no_fading,
            "Free_space_path_loss_dBm": w_to_dbm(Pr_no_fading),
            "Nakagami_W": Pr_with_fading,
            "Nakagami_dBm": w_to_dbm(Pr_with_fading),
            "Com_Prob": coverage_probability
        }

    def _comm_step(self, step):
        traci.simulationStep()
        vehicle_id_list = traci.vehicle.getIDList()
        # vehicle action
        for vehicle_id in vehicle_id_list:
            clean_vehicle_id = re.sub(r'[^\d.-]', '', vehicle_id)
            if not clean_vehicle_id:
                clean_vehicle_id = '0'
            clean_vehicle_id = float(clean_vehicle_id)
            position = traci.vehicle.getPosition(vehicle_id)
            clipped_samples = 0
            if np.random.binomial(n=1, p=self.posibility_task, size=1):
               
                comm_recive = self._communication_fading(self.uav.start_position, position, self.vehicle_transP_dBm)
                if np.random.binomial(1, comm_recive["Com_Prob"]):
                    row_num, _ = self.vehicle_comm_list.shape
                    if row_num < self.Max_vehicales:
                        comm_T_normal_samples = np.random.normal(loc=self.task_mean, scale=self.task_std, size=1)
                        comm_T_data = np.clip(comm_T_normal_samples, 0, 100) * 1e6  # M
                        comm_R_normal_samples = np.random.normal(loc=self.task_mean, scale=self.task_std,size=1)
                        comm_R_data = np.clip(comm_R_normal_samples, 0, 100) * 1e6  # M

                        comp_normal_samples = np.random.normal(loc=self.task_mean, scale=self.task_std, size=1)
                        comp_data = np.clip(comp_normal_samples, 0, 100) * 1e6  # M

                        task_deadline_samples = np.random.normal(loc=5, scale=2, size=1)
                        task_deadline_date = np.clip(task_deadline_samples, 1, 10)

                        start_time = step
                        deadline_time = task_deadline_date[0] + step
                        new_line = np.array([clean_vehicle_id, position[0], position[1],
                                    comm_recive["Nakagami_dBm"], comm_T_data[0], comm_R_data[0],
                                    comp_data[0], start_time, deadline_time])
                        self.vehicle_comm_list = np.vstack((self.vehicle_comm_list, new_line))
                    else:
                        self.outpop += 1


        def grow_to_rows(A, target_rows, fill_value=0):
            A_expanded = np.copy(A)
            A_size = np.size(A)
            if A_size:
                current_rows, cols = A.shape
                if current_rows >= target_rows:
                    return A.copy()
                rows_to_add = target_rows - current_rows
                new_rows = np.full((rows_to_add, cols), fill_value)
                A_expanded = np.vstack([A, new_rows])
            return A_expanded

        if self.vehicle_comm_list.size:
            sorted_indices = np.lexsort((self.vehicle_comm_list[:, 3], self.vehicle_comm_list[:, 6]))
            self.vehicle_comm_list = self.vehicle_comm_list[sorted_indices]
        OBS = grow_to_rows(self.vehicle_comm_list, self.Max_vehicales)
        obs = np.concatenate((np.array(self.uav.start_position).flatten(), OBS.flatten()))
        return {"Uav_position": self.uav.start_position,
                "vehicle_comm_list": self.vehicle_comm_list, #OBS.ravel()
                }, obs

    def _count_outline_task(self, step):
        count_outline = 0
        new_list = []
        for row in self.vehicle_comm_list:
            if step > int(float(row[7])):
                count_outline += 1
            else:
                new_list.append(row)
        if count_outline:
            self.vehicle_comm_list = np.array(new_list)
        return count_outline

    def Shannon(self, bandwidth, Remaining_interference_power, recive_power):
        return bandwidth * np.log2(1 + recive_power / (Remaining_interference_power + dbm_to_w(self.noise_dBm)))

    def _trans_time(self):
        trans_time_s = np.array([])
        if self.vehicle_comm_list.size:
            current_trans_comm_list = self.vehicle_comm_list[self.vehicle_comm_list[:, 7] >= self.step]
            if current_trans_comm_list.size:
                sorted_indices = np.argsort(-current_trans_comm_list[:, 3])
                sorted_values = current_trans_comm_list[sorted_indices]
                recive_power_w = np.array([])
                for sorted_value in sorted_values[:,3]:
                    recive_power_w = np.append(recive_power_w, dbm_to_w(sorted_value))
                Sum_recive_power_w = np.sum(recive_power_w)
                for index in range(len(recive_power_w)):
                    Remaining_interference_power = Sum_recive_power_w - recive_power_w[index]
                    Sum_recive_power_w = Remaining_interference_power
                    C = self.Shannon(self.f_Hz, Remaining_interference_power, recive_power_w[index])
                    trans_time_s = np.append(trans_time_s, sorted_values[index, 5]/C)
        return trans_time_s

    def _act(self, a):
        
        a = np.array(a[0][3:])
        sorted_indices = np.argsort(-a)
        sorted_values = a[sorted_indices]

        UAV_trans_power = self.uav.trans_power_dBm
        a_sum = a.sum()
        index = 0
        comp_time_s = np.array([])
        comp_power_J = 0
        trans_time_s = np.array([])
        trans_power_J = 0
        remaining_power = UAV_trans_power

        for power_weight in sorted_values:
            if power_weight:
                trans_power = (power_weight / a_sum) * UAV_trans_power
                if sorted_indices[index] < self.vehicle_comm_list.size/9:
                    recive_power = self._communication_fading(self.uav.start_position,
                                                              (self.vehicle_comm_list[sorted_indices[index],1],
                                                               self.vehicle_comm_list[sorted_indices[index],2]), trans_power)
                    remaining_power -= trans_power
                    recive_remaining_power = self._communication_fading(self.uav.start_position,
                                                                        (self.vehicle_comm_list[sorted_indices[index],1],
                                                                         self.vehicle_comm_list[sorted_indices[index],2]), remaining_power)
                    C = self.Shannon(self.f_Hz, recive_remaining_power['Nakagami_W'], recive_power['Nakagami_W'])

                    trans_time_s = np.append(trans_time_s,self.vehicle_comm_list[sorted_indices[index],4]/C) # sorted_indices
                    trans_power_J += trans_time_s[-1] * trans_power
                    comp_return = self.uav.comp(self.vehicle_comm_list[sorted_indices[index],6]) # comp_time_s comp_cost_J
                    self.vehicle_comm_list[sorted_indices[index],:] = 0
                    comp_time_s = np.append(comp_time_s,comp_return["comp_time_s"]) # sorted_indices
                    comp_power_J += comp_return["comp_cost_J"]
            index += 1
        if self.vehicle_comm_list.size:
            self.vehicle_comm_list = self.vehicle_comm_list[self.vehicle_comm_list[:,7] != 0]
        
        return {'comp_time_s': comp_time_s, 'comp_cost_J': comp_power_J,'comm_trans_time_s': trans_time_s,'comm_trans_power_J': trans_power_J}

    def _step(self, a):
        a = torch.tensor(self.action_space.low) + 0.5 * (a + 1.0) * (torch.tensor(self.action_space.high) - torch.tensor(self.action_space.low))
        reward_return = 0
        comm_re = self._trans_time()
        next_position = (a[0][0], a[0][1], a[0][2])
        move_re = self.uav.move(next_position)  # UAV 移动
        reward_return = move_re['reward']

        comm_comp_tr = self._act(a)
        time_cost = comm_comp_tr['comp_time_s'].sum() + comm_comp_tr['comm_trans_time_s'].sum()
        energy_cost = comm_comp_tr['comp_cost_J'] + comm_comp_tr['comm_trans_power_J'] + move_re['total_energy_J']
        # print(f"move reward: {reward_return}, success_task: {comm_comp_tr['comp_time_s'].size},comp_time_s: {comm_comp_tr['comp_time_s'].sum()}, "
        #       f"comp_cost_J: {comm_comp_tr['comp_cost_J']}, comm_trans_time_s: {comm_comp_tr['comm_trans_time_s'].sum()}, "
        #       f"comm_trans_power_J: {comm_comp_tr['comm_trans_power_J']}")
        r = reward_return + comm_comp_tr['comp_time_s'].size - time_cost - energy_cost
        d = (self.step >= self.Max_step)
        # traci.simulationStep() # SUMO step
        count_outline = self._count_outline_task(self.step) # 计算超出deadline的任务
        self.step += 1
        info, obs=self._comm_step(self.step) # 生成新任务
        return obs, r, d

    def _reset(self):
        traci.close()
        self.step = 0
        self.uav = UAV()
        self.vehicle_comm_list = np.empty((0, 9))
        traci.start(["sumo", "-c", self.SUMO_CFG])
        # print("Starting SUMO simulation with TraCI...")
        # traci.simulationStep()  # SUMO step
        info, obs=self._comm_step(self.step)  # 生成新任务
        # print(f"Initial vehicles in the simulation: {traci.vehicle.getIDList()}")
        return obs

def main():
    ENV=MEC()
    SP=ENV._observation_space()
    # print(f"observation_space:{SP}")
    acts=ENV._action_space()
    # print(f"action_space:{acts} shape {acts.shape}")
    a = np.ones(acts.shape)
    b = -np.ones(acts.shape)
    # random_actor = torchd.independent.Independent(
    #             torchd.uniform.Uniform(
    #                 torch.tensor(acts.low).repeat(1, 1),
    #                 torch.tensor(acts.high).repeat(1, 1),
    #             ),
    #             1,
    #         )
    random_actor = torchd.independent.Independent(
                torchd.uniform.Uniform(
                    torch.tensor(b).repeat(1, 1),
                    torch.tensor(a).repeat(1, 1),
                ),
                1,
            )
    done = False
    reward = 0
    for _ in range(2):
        obs=ENV._reset()
        # print(f"obs_UAV_pos:{obs[0:2]}\n"
        #   f"obs_com_list:{obs[3:]}")
        while not done:
            a=random_actor.sample()
            # print(f"Action: {a}")
            obs, r, done = ENV._step(a)
            # print(f"Reward: {r}, Done: {done}")
            reward += r
            # print(f"obs_UAV_pos:{obs['Uav_position']}\n"
            #       f"obs_com_list:{obs['vehicle_comm_list']}")
        print(f"Episode Reward: {reward} steps: {ENV.step}")
        reward = 0
        done = False
    traci.close()
    
if __name__ == "__main__":
    main()

# python $SUMO_HOME/tools/randomTrips.py -n grid.net.xml -r grid.rou.xml -b 0 -e 100 --period 5 --binomial 2 --prefix car --validate