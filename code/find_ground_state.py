# import numpy as np
# import matplotlib.pyplot as plt

# def calculate_energe_change(sustem:np.ndarray;type:str,x:int,y:int)->tuple[np.ndarry,int]:
#     """
#     输入分别是：初始的系统状态，随机选中的格点的类型，横坐标，纵坐标
#     输出是
#     """

# import numpy as np
# import matplotlib.pyplot as plt

# def ising_monte_carlo(N=40, J=1.0, T=5.0, steps=50000):
#     spins = np.random.choice([-1, 1], size=(N, N))
#     beta = 1.0 / T

#     for _ in range(steps):
#         i, j = np.random.randint(0, N, 2)
        
#         if (i+j) % 2 == 1:
#             neighbors = (
#                 spins[(i+1)%N, j] + spins[(i-1)%N, j] +
#                 spins[i, (j+1)%N] + spins[i, (j-1)%N] +
#                 spins[(i+1)%N,(j+1)%N] + spins[(i-1)%N,(j-1)%N])
#         else:
#             neighbors = (
#                 spins[(i+1)%N, j] + spins[(i-1)%N, j] +
#                 spins[i, (j+1)%N] + spins[i, (j-1)%N] +
#                 spins[(i+1)%N,(j-1)%N] + spins[(i-1)%N,(j+1)%N])
#         delta_E = 2 * J * spins[i, j] * neighbors
        
#         if delta_E <= 0 or np.random.rand() < np.exp(-beta * delta_E):
#             spins[i, j] *= -1  

#     return spins

# spins = ising_monte_carlo(N=50, T=2.5, steps=100000)
# plt.imshow(spins, cmap='gray')
# plt.title("Ising Model at T=2.5")
# plt.show()

import numpy as np
import matplotlib.pyplot as plt


# 取格点尺寸大小为N，计算基态
# 通过三维数组【N,N,2]来分别代表A,B两种格点

def ising_monte_carlo(N:int,J=1.0,step:int)->np.ndarray:
    """
    用于计算基态，使用蒙特卡洛方法，
    """
    system = np.random.choice([-1, 1], size=(N, N,2))
    for _ in range(step):
        type_gedian = np.random.randint(0, 2) #0代表A，1代表B
        # 随机选取一个格点
        i, j = np.random.randint(0, N, 2)
        energy_change = calculte_energy_change_in_neighbors(system, i, j,type_gedian)
        w = calculate_change_possibility(energy_change)
        if np.random.rand() < w:
            system[i, j, type_gedian] *= -1
    return system




def calculte_energy_change_in_neighbors(system:np.ndarray,x:int,y:int,type_gedian:int)->int:
    """
    计算邻居的能量变化
    """
    if type_gedian ==0:
        energy_old = system[x, y, 0] * (syetem[x-1,y,0]+system[x+1,y,0]+sysyem[x,y,1]+system[x-1,y,1]+system[x,y-1,1]+system[x-1,y-1,1])
        energy_new = -system[x, y, 0] * (syetem[x-1,y,0]+system[x+1,y,0]+sysyem[x,y,1]+system[x-1,y,1]+system[x,y-1,1]+system[x-1,y-1,1])
        return energy_new - energy_old
    else:
        energy_old = system[x, y, 1] * (syetem[x,y-1,1]+system[x,y+1,1]+sysyem[x,y,0]+system[x+1,y,0]+system[x,y+1,0]+system[x+1,y+1,0])
        energy_new = -system[x, y, 1] * (syetem[x,y-1,1]+system[x,y+1,1]+sysyem[x,y,0]+system[x+1,y,0]+system[x,y+1,0]+system[x+1,y+1,0])
        return energy_new - energy_old

def calculate_change_possibility(energy_change:int)->float:
    """
    计算能量变化的可能性
    """
    if energy_change < 0:
        return 1.0
    else:
        return np.exp(-energy_change)

def visualize_spin(spin: np.ndarray):
    """
    可视化AB格点的自旋状态。
    
    参数:
        spin: np.ndarray, 形状为 (N, N, 2)，第三维度为 0 表示 A 格点，1 表示 B 格点。
    """
    N = spin.shape[0]
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # 遍历 A 和 B 格点
    for i in range(N):
        for j in range(N):
            # A 格点的实际位置
            x_a, y_a = 2 * i, 2 * j
            if spin[i, j, 0] == 1:
                ax.arrow(x_a, y_a, 0, 0.4, head_width=0.2, head_length=0.2, fc='pink', ec='pink')
            else:
                ax.arrow(x_a, y_a, 0, -0.4, head_width=0.2, head_length=0.2, fc='pink', ec='pink')
            
            # B 格点的实际位置
            x_b, y_b = 2 * i + 1, 2 * j + 1
            if spin[i, j, 1] == 1:
                ax.arrow(x_b, y_b, 0, 0.4, head_width=0.2, head_length=0.2, fc='blue', ec='blue')
            else:
                ax.arrow(x_b, y_b, 0, -0.4, head_width=0.2, head_length=0.2, fc='blue', ec='blue')
    
    # 设置网格和显示范围
    ax.set_xlim(-1, 2 * N)
    ax.set_ylim(-1, 2 * N)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    ax.grid(False)
    plt.title("Spin Visualization (A: Pink, B: Blue)")
    plt.show()
    path = "/image/show_ground_state.png"
    plt.savefig(path)

if __name__ == "__main__":
    N = 8
    J = 1.0
    steps = 1000
    spins = ising_monte_carlo(N, J, steps)
    visualize_spin(spins)