#"""
#@original author: Junghun Shin
#@modified by Hyunsol Park, Wontae Hwang
#"""
import numpy as np
import pandas as pd

Mach=['1', '2', '3', '4', '5', '6', '7', '8', '9']
Mach=np.array(Mach)
AoA = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15']
AoA=np.array(AoA)
NMa = 4
NAoA = 5
noCol = 11
numd = 123
zone1_i = 401 
zone1_j = 81
cuttail = 20	
glayer = 64 	
sim_data_path = "../../../Database/Simdata_airfoil_Steady_FC/"
res_data_path = "../Results/"
Tecplot_header_in = "variables=X, Y, Z, Rho, U, V, W, P, T, Vor, Qcri"
Tecplot_header_out = "variables=X, Y, Rho, U, V, P"

# list of file names
filenames = []
merged = []
Ntime = 0 

for i in (Mach):
	for j in (AoA):
		filenames.append(sim_data_path+"flo_" + str(i) +"_" + str(j) + ".dat")
	Ntime += 1

def is_number(num):
    try:
        float(num)
        return True #num을 float으로 변환할 수 있는 경우
    except ValueError: #num을 float으로 변환할 수 없는 경우
        return False
# --- 1. 定义目标测试集 (与模型脚本完全一致) ---
target_test_cases = [(2, 9), (2, 14), (2, 5),(4, 9), (4, 14), (4, 5),(6, 9), (6, 14), (6, 5),(8, 9), (8, 14), (8, 5)]
num_test = len(target_test_cases) # 结果为 4

# 计算训练集数量：总数 135 (9*15) - 测试集 4 = 131
numd = 131 

# --- 2. 初始化逻辑修改 ---
k = 0   # 训练集索引
kk = 0  # 测试集索引
Nm = 1
Na = 1

for file in filenames:
    snapshot_data = []
    # 这里不再需要定义 Testname = sim_data_path + ...
    
    with open(file) as f:
        lines = f.readlines()
        for line in lines:
            vals = []
            line = line.replace("\n","")
            raw_vals = line.split(" ")
            for c in raw_vals:
                if is_number(c):
                    vals.append(c)
            
            if len(vals) == noCol:
                snapshot_data.append(vals)

        snapshot_data = np.array(snapshot_data, dtype=float)
        
        # 第一次循环时初始化数组
        if k == 0 and kk == 0:
            # 这里的 12 全部改为 num_test (即 4)
            Traindata = np.zeros((numd, snapshot_data.shape[0], snapshot_data.shape[1]))
            Testdata = np.zeros((num_test, snapshot_data.shape[0], snapshot_data.shape[1]))
            Trainlabel = np.zeros((numd, 2))
            Testlabel = np.zeros((num_test, 2))

        # --- 3. 核心匹配逻辑修改 ---
        # 检查当前文件的 (Nm, Na) 是否属于目标测试案例
        if (Nm, Na) in target_test_cases:
            print(f"Match Test Case: flo_{Nm}_{Na}.dat -> Index {kk}")
            Testdata[kk, :, :] = snapshot_data
            Testlabel[kk, 0] = Nm
            Testlabel[kk, 1] = Na
            kk += 1
        else:
            # 属于训练集
            if k < numd:
                Traindata[k, :, :] = snapshot_data
                Trainlabel[k, 0] = Nm
                Trainlabel[k, 1] = Na
                k += 1

        # --- 4. 索引更新逻辑 (必须严格按照 1-15 循环) ---
        if Na % 15 == 0:
            Nm += 1
            Na = 0
        Na += 1
for i in range(2):
	if i == 0:
		array_data = Traindata
		N = array_data.shape[-2]
		save_path = res_data_path+"Staedy_airfoil_cuttail_train.npz"
				
	if i == 1:
		array_data = Testdata
		save_path = res_data_path+"Staedy_airfoil_cuttail_test.npz"

	xc_star = array_data[:,:,0] 
	yc_star = array_data[:,:,1] 
	
	dc_star = array_data[:,:,3]
	uc_star = array_data[:,:,4]
	vc_star = array_data[:,:,5]
	pc_star = array_data[:,:,7]

	DC = dc_star.T
	UC = uc_star.T 
	VC = vc_star.T   
	PC = pc_star.T    
	XC = xc_star.T    
	YC = yc_star.T 
	
	#cut grid for memory efficiency
	idx_x_slice = np.array([])
	for i in range(glayer):
		idx_x_slice = np.append(idx_x_slice, np.arange(cuttail+i*zone1_i, 
								(zone1_i-cuttail)+i*zone1_i)).astype('int32')
	#print(idx_x_slice.shape[0])
	DC_star = DC[idx_x_slice,:]
	UC_star = UC[idx_x_slice,:]
	VC_star = VC[idx_x_slice,:]
	PC_star = PC[idx_x_slice,:]
	XC_star = XC[idx_x_slice,:]
	YC_star = YC[idx_x_slice,:]
 

	np.savez(save_path, 
         XC=XC_star, YC=YC_star, DC=DC_star, 
         UC=UC_star, VC=VC_star, PC=PC_star)