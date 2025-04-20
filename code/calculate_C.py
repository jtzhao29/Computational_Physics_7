from find_ground_state import ising_monte_carlo,calculate_energy_change_in_neighbors,calculate_change_possibility,visualize_spin,find_neibor
import numpy as np
import matplotlib.pyplot as plt


def compute_correlation(spin: np.ndarray, N: int, mu: int, nu: int) -> np.ndarray:
    """
    计算关联函数 C^{mu, nu}(r)
    mu, nu = 0 表示 A 子格；1 表示 B 子格
    返回一个形状为 (N, N) 的二维数组
    """
    correlation = np.zeros((N, N))
    count = np.zeros((N, N))  # 用于记录每个位移的统计次数

    for i in range(N):
        for j in range(N):
            s1 = spin[mu, i, j]
            for dx in range(N):
                for dy in range(N):
                    ni = (i + dx) % N
                    nj = (j + dy) % N
                    s2 = spin[nu, ni, nj]
                    correlation[dx, dy] += s1 * s2
                    count[dx, dy] += 1

    return correlation / count


def compute_correlation_average(num:int,N:int, mu: int, nu: int,steps:int) -> np.ndarray:
    corralation_average = correlation = np.zeros((N, N))
    for i in range(num):
        spin =  ising_monte_carlo(N, steps)
        correlation = np.zeros((N, N))
        count = np.zeros((N, N))  # 用于记录每个位移的统计次数

        for i in range(N):
            for j in range(N):
                s1 = spin[mu, i, j]
                for dx in range(N):
                    for dy in range(N):
                        ni = (i + dx) % N
                        nj = (j + dy) % N
                        s2 = spin[nu, ni, nj]
                        correlation[dx, dy] += s1 * s2
                        count[dx, dy] += 1
        corralation_average +=correlation/ count

    return corralation_average/num


def plot_correlation_matrix_average( N: int,num:int,steps:int):
    """
    可视化四种关联函数的热力图
    """
    labels = ['A', 'B']
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    
    for i, mu in enumerate([0, 1]):
        for j, nu in enumerate([0, 1]):
            corr = np.zeros((N,N))
            for _ in range(num):
                corr += compute_correlation_average(num,N,mu,nu,steps)
            corr=corr/num

            ax = axs[i][j]
            im = ax.imshow(corr, cmap='coolwarm', origin='lower')
            ax.set_title(f'C^{labels[mu]}{labels[nu]}(r)', fontsize=14)
            plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.savefig(f'./image/correlation_heatmap_average_size={N}.png')
    plt.close()


def plot_correlation_matrix(spin: np.ndarray, N: int,num:int):
    """
    可视化四种关联函数的热力图
    """
    labels = ['A', 'B']
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    
    for i, mu in enumerate([0, 1]):
        for j, nu in enumerate([0, 1]):
            corr = np.zeros((N,N))
            for _ in range(num):
                corr += compute_correlation(spin, N, mu, nu)
            corr=corr/num

            ax = axs[i][j]
            im = ax.imshow(corr, cmap='coolwarm', origin='lower')
            ax.set_title(f'C^{labels[mu]}{labels[nu]}(r)', fontsize=14)
            plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.savefig(f'./image/correlation_heatmap_size={N}.png')
    plt.close()



if __name__ == "__main__":
    N = 8
    J = 1.0
    steps = 5000
    num = 80
    # spin_average = np.zeros((2,N,N))
    # for i in range(num):
    # spins = ising_monte_carlo(N, steps)
    #     spin_average += spins
    # spin_average=spin_average/num

    # spins = np.array([
    #         [[1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1],
    #          [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1]],
    #         [[-1, -1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1, -1],
    #          [-1, -1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1, -1],
    #          [-1, -1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1, -1]]
    #     ])

    # visualize_spin(spin_average,N)
    plot_correlation_matrix_average( N,num,steps)
