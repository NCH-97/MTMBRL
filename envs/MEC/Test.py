import numpy as np
import sys
import UAV
sys.path.append('/usr/share/sumo/tools')
import traci


# vehicle_comm_list = np.array([[3,1,21],[7,1,5],[46,6,1],[6,67,1],[5,8,9]])
# print(vehicle_comm_list,"\n")
# vehicle_comm_list[1,:]=0
# print(vehicle_comm_list)
# filtered_matrix = vehicle_comm_list[vehicle_comm_list[:, 0] != 0]
# print("\n过滤后矩阵:")
# print(filtered_matrix)
# a = vehicle_comm_list[0][3:]
# sorted_indices = np.argsort(-a)
# sorted_values = a[sorted_indices]
# original_to_sorted_index = np.empty_like(sorted_indices)
# original_to_sorted_index[sorted_indices] = np.arange(len(a))
# print(f"vehicle_comm_list:\n{a}")
# print("排序后值：\n", sorted_values)
# print("排序后索引（值来自原数组的哪个位置）：\n", sorted_indices)
#
# for index in range(len(sorted_indices)):
#     print(f"sorted_indices[{index}]:{sorted_indices[index]}")
#     print(f"sorted_values[{index}]:{sorted_values[index]}")
#     print(f"a[{sorted_indices[index]}]:{a[sorted_indices[index]]}")

# traci.start(["sumo-gui", "-c", "./TEST/SUMO_TEST/HG/grid.sumocfg"])
# print("Starting SUMO simulation with TraCI...")
# print(f"SUMO bound {traci.simulation.getNetBoundary()}")
# for step in range(150):
#     traci.simulationStep()
#     vehicle_id_list = traci.vehicle.getIDList()
#     print(f"step: {step}")
#     for vehicle_id in vehicle_id_list:
#         position = traci.vehicle.getPosition(vehicle_id)
#         print(f"{vehicle_id}: {position}")
# traci.close()

# uav = UAV.UAV()
# print(f"uav.start_position:{uav.start_position}")
# print(f"{uav.move((0,18,40))}")