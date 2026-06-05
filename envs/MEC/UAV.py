import math
import numpy as np
import sympy as sp
from scipy.integrate import quad
from scipy.stats import nakagami, gamma
from scipy.special import gamma as GammaFunc, gammaincc

def is_symbolically_integrable(expr, var):
    try:
        result = sp.integrate(expr, var)
        return not isinstance(result, sp.Integral)
    except:
        return False

def is_numerically_integrable(f, a, b):
    try:
        result, error = quad(f, a, b)
        return np.isfinite(result)
    except:
        return False

class UAV:
    def __init__(self, UAV_id=0, start_position=(0,0,40)): #() ()
        self.UAV_id = UAV_id
        self.start_position = start_position
        #MOVE
        self.mass_kg = 1.5
        self.propeller_diameter_m = 0.3
        self.frontal_area_m2 = 0.06
        self.drag_coefficient = 1.1
        self.air_density = 1.225
        self.max_velocity_mps = 6
        self.acceleration_mps2 = 3
        self.g = 9.81
        self.each_step_time = 5
        #COMPUTING
        self.compute_ability_flops = 64e9
        self.compute_cost_per_flops_JpFLOPs = 1e-9
        #COMMUNICATION
        self.trans_power_dBm = 20
        self.recive_bound_dBm = -95
        self.c_mps = 3e8
        self.f_Hz = 5e9
        self.Gt = self.Gr =1
        self.m = 1
        self.m_max = 2
        self.m_min = 0.5
        self.m_func = lambda d,h,a=-0.01,b=0.01: min(max(self.m_min, 2*math.exp(-b*math.sqrt(d**2-h**2))+0.5*math.exp(-2*math.atan(a*h/d))), self.m_max)
        self.noise_power_dBm = -105


    def set_start_position(self, position):
        self.start_position = position

    def move(self, end_position = (0, 0, 0)):

        total_distance_m = 0.0
        thrust_N = self.mass_kg * self.g
        rotor_area = 4 * math.pi * (self.propeller_diameter_m / 2) ** 2
        p_hover = (thrust_N ** 1.5) / math.sqrt(2 * self.air_density * rotor_area)

        def P_crusie(v):
            return 0.5 * self.air_density * self.drag_coefficient * self.frontal_area_m2 * v ** 3

        p_cruise=P_crusie(self.max_velocity_mps)
        # Calculate cruise distance if start/end positions provided (3D)
        if self.start_position != end_position:
            total_distance_m = math.sqrt(
                (self.start_position[0] - end_position[0]) ** 2 +
                (self.start_position[1] - end_position[1]) ** 2 +
                (self.start_position[2] - end_position[2]) ** 2
            )
        if total_distance_m > 18:
            def move_point_3d(start, end, distance):
                x1, y1, z1 = start
                x2, y2, z2 = end

                dx = x2 - x1
                dy = y2 - y1
                dz = z2 - z1

                length = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

                if length == 0:
                    return start

                scale = distance / length
                return (
                    x1 + dx * scale,
                    y1 + dy * scale,
                    z1 + dz * scale
                )

            self.start_position = move_point_3d(self.start_position, end_position, 18)
            return {'move_distance_m': 18.0,
                    'move_energy_J': 34.927200000000006,
                    'move_time_s': 3.0,
                    'hover_time_s': 2.0,
                    'hover_energy_J': 135.6410559552138,
                    'total_energy_J': 170.5682559552138,
                    'reward':18-total_distance_m
                    }
        else:
            self.start_position = end_position
            accel_dist = 0.5 * self.max_velocity_mps ** 2 / self.acceleration_mps2
            decel_dist = accel_dist
            max_velocity_mps = self.max_velocity_mps
            if total_distance_m < (accel_dist + decel_dist):
                accel_dist = decel_dist = total_distance_m / 2
                max_velocity_mps = math.sqrt(accel_dist * 2 * self.acceleration_mps2)
            cruise_velocity_mps = max_velocity_mps
            p_acc_cruise = 0
            f_cruise=is_symbolically_integrable(P_crusie, sp.Symbol('v'))
            if f_cruise:
                p_acc_cruise = sp.integrate(P_crusie, (sp.Symbol('v'), 0, cruise_velocity_mps))
            else:
                f_acc_cruise = is_numerically_integrable(P_crusie,0, cruise_velocity_mps)
                if f_acc_cruise:
                    p_acc_cruise=quad(P_crusie,0, cruise_velocity_mps)
                    p_acc_cruise = p_acc_cruise[0]
                else:
                    print("Can't integrable")

            accel_time = cruise_velocity_mps / self.acceleration_mps2
            cruise_distance = total_distance_m - (accel_dist + decel_dist)
            cruise_time = cruise_distance / self.max_velocity_mps if cruise_distance > 0 else 0
            energy_total_J = p_acc_cruise * 2 + p_cruise * cruise_time
            time_total_s = cruise_time + accel_time
            hover_energy_J = (self.each_step_time - time_total_s) * p_hover
            total_energy_J = energy_total_J + hover_energy_J

        return {
            "move_distance_m": total_distance_m,
            "move_energy_J": energy_total_J,
            "move_time_s": time_total_s,
            "hover_time_s" : (self.each_step_time - time_total_s),
            "hover_energy_J": hover_energy_J,
            "total_energy_J": total_energy_J,
            'reward': 0
        }

    def comp(self,comp_data):
        comp_time_s = comp_data/self.compute_ability_flops
        comp_cost_J = comp_time_s * self.compute_cost_per_flops_JpFLOPs
        return {
            "comp_time_s": comp_time_s,
            "comp_cost_J": comp_cost_J
        }

    def comm(self, car_position, car_trans_power_dBm, car_min_recive_dBm=-95):
        print("Communication")
        def dbm_to_w(dbm):
            return 10 ** (dbm / 10) / 1000
        def w_to_dbm(watt):
            return 10 * np.log10(watt * 1000)
        # free space path loss
        def free_space_path_loss(pt,frequency, distance):
            wavelength = self.c_mps / frequency
            return pt * self.Gt * self.Gr /((4 * np.pi * distance / wavelength) ** 2)

        distance = math.sqrt(
                (self.start_position[0] - car_position[0]) ** 2 +
                (self.start_position[1] - car_position[1]) ** 2 +
                (self.start_position[2] - 0) ** 2
            )

        Pt=dbm_to_w(car_trans_power_dBm)
        Pr_no_fading = free_space_path_loss(Pt,self.f_Hz,distance)  # 实际接收功率

        self.m = self.m_func(distance, self.start_position[2])
        cdf_value = gamma.cdf(car_min_recive_dBm, a=self.m, scale=(Pr_no_fading / self.m)) # CDF
        coverage_probability = 1 - cdf_value
        z = self.m * dbm_to_w(car_min_recive_dBm) / Pr_no_fading
        Gamma_m_z = GammaFunc(self.m)*gammaincc(self.m, z)
        Gamma_m1_z = GammaFunc(self.m+1)*gammaincc(self.m+1, z)
        E_pt_cond = Pr_no_fading * (Gamma_m1_z / Gamma_m_z) * (1 / self.m)

        # fading = nakagami.rvs(self.m, scale=np.sqrt(1 / self.m), size=100)
        # Pr_with_fading = Pr_no_fading * np.mean(fading) ** 2

        return {
            "Pt_W": Pt,
            "Pt_dBm": w_to_dbm(Pt),
            "Distance": distance,
            "Nakagami_W": E_pt_cond,
            "Nakagami_dBm": w_to_dbm(E_pt_cond),
            "Coverage_Probability": coverage_probability
        }

def main():
    uav = UAV()
    re = uav.move(end_position = (0, 0, 12))
    print(f"re:{re}")
    comm = uav.comm((0, 100, 0), 20)
    print(f"comm:{comm}")

if __name__ == "__main__":
    main()