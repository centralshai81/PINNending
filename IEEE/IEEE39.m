% ============================================
% IEEE 39节点系统 - 全节点故障参数分析
% ============================================

clear; clc; close all;
format compact;
warning off;

%% 系统参数
baseMVA = 100;
baseKV = 345;
f0 = 50;

fprintf('================================================================\n');
fprintf('IEEE 39节点系统 - 全节点故障参数分析\n');
fprintf('================================================================\n\n');

%% 1. 加载IEEE 39节点系统
fprintf('步骤1: 加载系统数据...\n');
mpc = create_ieee39_system();
n_bus = size(mpc.bus, 1);
fprintf('✓ IEEE 39节点系统加载成功\n');
fprintf('   节点总数: %d个\n', n_bus);

%% 2. 构建导纳和阻抗矩阵
fprintf('\n步骤2: 构建系统矩阵...\n');
[Ybus, Zbus] = build_admittance_matrix(mpc);
fprintf('✓ 系统矩阵构建完成\n');

%% 3. 故障配置
fprintf('\n步骤3: 配置故障场景...\n');
faults = configure_fault_scenarios();
n_faults = length(faults);
fprintf('✓ %d个故障场景配置完成\n', n_faults);

%% 4. 预计算所有故障结果（用于快速访问）
fprintf('\n步骤4: 预计算所有故障分析结果...\n');
fprintf('----------------------------------------\n');

% 创建结果数组
all_results = cell(1, n_faults);

for fault_idx = 1:n_faults
    fault = faults(fault_idx);
    
    fprintf('计算故障场景 %d/%d: ', fault_idx, n_faults);
    fprintf('节点%d, %s\n', fault.bus, fault.type);
    
    % 执行该故障的详细分析
    results = analyze_fault_all_nodes_enhanced(mpc, fault, Zbus, baseMVA, baseKV, fault_idx, Ybus);
    
    % 保存结果到数组
    all_results{fault_idx} = results;
end

%% 5. 交互式故障分析
fprintf('\n================================================================\n');
fprintf('交互式故障分析模式\n');
fprintf('================================================================\n');

while true
    % 显示故障列表
    fprintf('\n可分析的故障列表（共%d个）：\n', n_faults);
    for i = 1:n_faults
        fault = faults(i);
        fprintf('%2d. 节点%d, %s故障 (Rf=%.3f, Xf=%.3f)\n', ...
            i, fault.bus, fault.type, fault.Rf, fault.Xf);
        fprintf('    描述: %s\n', fault.description);
        fprintf('    时间: %.2f-%.2f秒\n', fault.start_time, fault.end_time);
    end
    
    % 用户选择
    fprintf('\n请选择要分析的故障序号（1-%d），输入0生成综合报告，输入-1退出: ', n_faults);
    fault_choice = input('');
    
    if fault_choice == -1
        fprintf('退出程序\n');
        break;
        
    elseif fault_choice == 0
        % 生成综合对比分析
        fprintf('\n生成综合对比分析...\n');
        if n_faults >= 6
            generate_comparative_analysis(mpc, faults, all_results{1}, all_results{2}, ...
                all_results{3}, all_results{4}, all_results{5}, all_results{6});
        else
            fprintf('需要至少6个故障场景进行综合对比分析\n');
        end
        
        % 生成故障影响热力图
        fprintf('\n生成故障影响热力图...\n');
        if n_faults >= 6
            generate_fault_impact_heatmap(mpc, faults, all_results{1}, all_results{2}, ...
                all_results{3}, all_results{4}, all_results{5}, all_results{6});
        else
            fprintf('需要至少6个故障场景生成热力图\n');
        end
        
    elseif fault_choice >= 1 && fault_choice <= n_faults
        % 分析选定故障
        fault = faults(fault_choice);
        results = all_results{fault_choice};
        
        fprintf('\n============================================\n');
        fprintf('故障%d分析：节点%d的%s故障\n', fault_choice, fault.bus, fault.type);
        fprintf('============================================\n');
        fprintf('故障参数：\n');
        fprintf('  故障点：节点%d\n', fault.bus);
        fprintf('  故障类型：%s\n', fault.type);
        fprintf('  故障阻抗：Rf = %.3f pu, Xf = %.3f pu\n', fault.Rf, fault.Xf);
        fprintf('  故障时间：%.2f-%.2f秒 (持续时间: %.2f秒)\n', ...
            fault.start_time, fault.end_time, fault.duration);
        fprintf('  故障描述：%s\n\n', fault.description);
        
        % 显示所有节点电气参数（故障期间的稳态值）
        fprintf('所有节点电气参数（故障期间稳态值）:\n');
        fprintf('%-4s %-10s %-10s %-10s %-10s %-10s %-10s %-12s\n', ...
            '节点', '|V|(pu)', '∠V(°)', 'P(pu)', 'Q(pu)', 'ΔV(%)', '|I|(pu)', '状态');
        fprintf('%s\n', repmat('-', 85, 1));
        
        % 找到故障期间的时间索引（取故障中间时刻）
        fault_time_idx = find(results.sim_time >= (fault.start_time + fault.duration/2), 1);
        
        for node = 1:n_bus
            % 获取故障期间的参数
            V_mag = results.V_mag(node, fault_time_idx);
            V_angle = results.V_angle(node, fault_time_idx);
            P_val = results.P_inject(node, fault_time_idx);
            Q_val = results.Q_inject(node, fault_time_idx);
            
            % 计算电压变化率（相对于故障前）
            V_pre = abs(results.V_pre(node));
            V_change_percent = (V_mag - V_pre) / V_pre * 100;
            
            % 计算电流幅值（通过注入功率计算）
            if V_mag > 0
                I_mag = abs(complex(P_val, Q_val) / V_mag);
            else
                I_mag = 0;
            end
            
            % 计算故障电流贡献
            I_fault = results.I_fault(node, fault_time_idx);
            
            % 判断节点状态
            if node == fault.bus
                status = '故障点';
            elseif V_change_percent < -30
                status = '严重跌落';
            elseif V_change_percent < -15
                status = '中等跌落';
            elseif V_change_percent < -5
                status = '轻微跌落';
            elseif V_change_percent > 10
                status = '电压升高';
            else
                status = '正常';
            end
            
            fprintf('%4d %10.4f %10.2f %10.4f %10.4f %10.2f %10.4f %12s\n', ...
                node, V_mag, V_angle, P_val, Q_val, V_change_percent, I_fault, status);
        end
        
        % 显示关键统计信息
        fprintf('\n============ 统计摘要 ============\n');
        
        % 收集所有节点的参数
        V_mags = results.V_mag(:, fault_time_idx);
        V_changes = (V_mags - abs(results.V_pre)) ./ abs(results.V_pre) * 100;
        
        low_v_count = sum(V_mags < 0.85);
        high_v_count = sum(V_mags > 1.15);
        severe_drop_count = sum(V_changes < -30);
        moderate_drop_count = sum(V_changes >= -30 & V_changes < -15);
        mild_drop_count = sum(V_changes >= -15 & V_changes < -5);
        
        fprintf('  节点总数：%d\n', n_bus);
        fprintf('  低压节点（<0.85 pu）：%d个\n', low_v_count);
        fprintf('  过压节点（>1.15 pu）：%d个\n', high_v_count);
        fprintf('  严重电压跌落（>30%%）：%d个节点\n', severe_drop_count);
        fprintf('  中等电压跌落（15-30%%）：%d个节点\n', moderate_drop_count);
        fprintf('  轻微电压跌落（5-15%%）：%d个节点\n', mild_drop_count);
        fprintf('  平均电压变化：%.2f%%\n', mean(V_changes));
        fprintf('  最大电压跌落：%.2f%%（节点%d）\n', ...
            min(V_changes), find(V_changes == min(V_changes), 1));
        fprintf('  最大电压升高：%.2f%%（节点%d）\n', ...
            max(V_changes), find(V_changes == max(V_changes), 1));
        
        % 发电机节点专门分析
        fprintf('\n============ 发电机节点分析 ============\n');
        gen_buses = mpc.gen(:, 1);
        fprintf('%-6s %-10s %-10s %-10s %-10s %-10s %-10s\n', ...
            '发电机', '|V|(pu)', '∠V(°)', 'P(pu)', 'Q(pu)', 'ΔV(%)', '状态');
        fprintf('%s\n', repmat('-', 70, 1));
        
        for i = 1:length(gen_buses)
            node = gen_buses(i);
            if node <= n_bus
                V_mag = results.V_mag(node, fault_time_idx);
                V_angle = results.V_angle(node, fault_time_idx);
                P_val = results.P_inject(node, fault_time_idx);
                Q_val = results.Q_inject(node, fault_time_idx);
                V_pre = abs(results.V_pre(node));
                V_change = (V_mag - V_pre) / V_pre * 100;
                
                if V_change < -30
                    status = '严重跌落';
                elseif V_change < -15
                    status = '中等跌落';
                elseif V_change < -5
                    status = '轻微跌落';
                else
                    status = '正常';
                end
                
                fprintf('%6d %10.4f %10.2f %10.4f %10.4f %10.2f %10s\n', ...
                    node, V_mag, V_angle, P_val, Q_val, V_change, status);
            end
        end
        
        % 生成详细报告
        fprintf('\n生成详细报告和图表...\n');
        generate_fault_detailed_report_enhanced(mpc, fault, results, fault_choice);
        
        % 询问是否继续
        fprintf('\n是否继续分析其他故障？(y/n): ');
        continue_choice = input('', 's');
        if ~strcmpi(continue_choice, 'y')
            fprintf('退出交互式分析\n');
            break;
        end
        
    else
        fprintf('输入错误，请输入1-%d之间的数字\n', n_faults);
    end
end

%% 6. 生成最终报告
fprintf('\n================================================================\n');
fprintf('全节点故障分析完成！\n');
fprintf('================================================================\n');

% 保存所有结果到MAT文件
save('IEEE39_all_fault_results.mat', 'mpc', 'faults', 'all_results', 'Ybus', 'Zbus');
fprintf('✓ 所有分析结果已保存到: IEEE39_all_fault_results.mat\n');

%% ========================================================================
% 增强版辅助函数定义
% ========================================================================

function results = analyze_fault_all_nodes_enhanced(mpc, fault, Zbus, baseMVA, baseKV, fault_id, Ybus)
    % 增强版：分析单个故障下所有节点的参数变化，包含完整电气参数
    n_bus = size(mpc.bus, 1);
    
    % 定义仿真时间
    sim_time = 0:0.001:1.5;
    n_steps = length(sim_time);
    
    % 初始化结果数组
    results = struct();
    results.fault = fault;
    results.sim_time = sim_time;
    
    % 所有节点的电压、相角、电流等参数
    results.V_mag = zeros(n_bus, n_steps);   % 电压幅值
    results.V_angle = zeros(n_bus, n_steps); % 电压相角
    results.I_fault = zeros(n_bus, n_steps); % 故障电流贡献
    results.P_inject = zeros(n_bus, n_steps); % 节点注入功率
    results.Q_inject = zeros(n_bus, n_steps); % 节点注入无功
    
    % 新增：存储完整的电压相量
    results.V_complex = zeros(n_bus, n_steps); % 复数电压
    results.I_complex = zeros(n_bus, n_steps); % 复数电流
    
    % 系统序阻抗
    Z1_sys = 0.05 + 0.15j;
    Z2_sys = 0.05 + 0.15j;
    Z0_sys = 0.10 + 0.30j;
    
    % 故障节点和阻抗
    fault_bus = fault.bus;
    Zf = fault.Rf + 1j * fault.Xf;
    Zkk = Zbus(fault_bus, fault_bus);
    
    % 计算故障电流（基于对称分量法）
    % 故障前电压（使用节点基准电压）
    results.V_pre = mpc.bus(:, 8) .* exp(1j * mpc.bus(:, 9) * pi/180);
    
    % 根据故障类型计算序电流
    V_pre_fault = results.V_pre(fault_bus);
    
    switch fault.type
        case '3LG'  % 三相短路
            I_f1 = V_pre_fault / (Zkk + Zf);
            I_f2 = 0;
            I_f0 = 0;
            
        case 'LG-A' % 单相接地
            Z1 = Zkk + Z1_sys;
            Z2 = Zkk + Z2_sys;
            Z0 = Zkk + Z0_sys;
            I_f1 = V_pre_fault / (Z1 + Z2 + Z0 + 3*Zf);
            I_f2 = I_f1;
            I_f0 = I_f1;
            
        case 'LLG-BC' % 两相接地
            Z1 = Zkk + Z1_sys;
            Z2 = Zkk + Z2_sys;
            Z0 = Zkk + Z0_sys;
            I_f1 = V_pre_fault / (Z1 + (Z2*(Z0+3*Zf))/(Z2+Z0+3*Zf));
            I_f2 = -I_f1 * (Z0+3*Zf) / (Z2+Z0+3*Zf);
            I_f0 = -I_f1 * Z2 / (Z2+Z0+3*Zf);
            
        case 'LL-BC' % 两相短路
            Z1 = Zkk + Z1_sys;
            Z2 = Zkk + Z2_sys;
            I_f1 = V_pre_fault / (Z1 + Z2 + Zf);
            I_f2 = -I_f1;
            I_f0 = 0;
    end
    
    % 计算相电流
    a = exp(1j*2*pi/3);  % 120度旋转算子
    if strcmp(fault.type, 'LG-A')
        % A相故障
        I_fa = I_f1 + I_f2 + I_f0;
        I_fb = a^2*I_f1 + a*I_f2 + I_f0;
        I_fc = a*I_f1 + a^2*I_f2 + I_f0;
    else
        % 对称故障
        I_fa = I_f1;
        I_fb = a^2*I_f1;
        I_fc = a*I_f1;
    end
    
    % 故障期间各节点电压变化
    for t_idx = 1:n_steps
        t = sim_time(t_idx);
        
        if t >= fault.start_time && t <= fault.end_time
            % 故障期间
            for k = 1:n_bus
                if k == fault_bus
                    % 故障节点电压
                    if strcmp(fault.type, 'LG-A')
                        V1 = results.V_pre(k) - Zbus(k, fault_bus)*I_f1;
                        V2 = -Zbus(k, fault_bus)*I_f2;
                        V0 = -Zbus(k, fault_bus)*I_f0;
                        Va = V1 + V2 + V0;
                        Vb = a^2*V1 + a*V2 + V0;
                        Vc = a*V1 + a^2*V2 + V0;
                        
                        results.V_complex(k, t_idx) = Va;
                        results.V_mag(k, t_idx) = abs(Va);
                        results.V_angle(k, t_idx) = angle(Va)*180/pi;
                    else
                        V_fault = results.V_pre(k) - Zbus(k, fault_bus)*I_f1;
                        results.V_complex(k, t_idx) = V_fault;
                        results.V_mag(k, t_idx) = abs(V_fault);
                        results.V_angle(k, t_idx) = angle(V_fault)*180/pi;
                    end
                    
                    % 故障节点电流
                    results.I_fault(k, t_idx) = abs(I_f1) * baseMVA/(sqrt(3)*baseKV);
                    
                    % 存储故障电流（复数形式）
                    results.I_complex(k, t_idx) = I_f1;
                    
                else
                    % 非故障节点电压
                    V_k = results.V_pre(k) - Zbus(k, fault_bus)*I_f1;
                    results.V_complex(k, t_idx) = V_k;
                    results.V_mag(k, t_idx) = abs(V_k);
                    results.V_angle(k, t_idx) = angle(V_k)*180/pi;
                    
                    % 非故障节点电流贡献
                    results.I_fault(k, t_idx) = 0;
                    results.I_complex(k, t_idx) = 0;
                end
                
                % 计算节点注入功率（基于导纳矩阵和电压）
                V_k_complex = results.V_complex(k, t_idx);
                I_k_complex = 0;
                
                % 通过导纳矩阵计算注入电流
                for j = 1:n_bus
                    if j ~= k && Ybus(k, j) ~= 0
                        I_k_complex = I_k_complex + Ybus(k, j) * results.V_complex(j, t_idx);
                    end
                end
                I_k_complex = I_k_complex + Ybus(k, k) * V_k_complex;
                
                % 计算注入功率
                S_inj = V_k_complex * conj(I_k_complex);
                results.P_inject(k, t_idx) = real(S_inj);
                results.Q_inject(k, t_idx) = imag(S_inj);
                
                % 存储复数电流
                results.I_complex(k, t_idx) = results.I_complex(k, t_idx) + I_k_complex;
            end
            
        elseif t > fault.end_time && t <= fault.end_time + 0.3
            % 恢复过程
            recovery_ratio = (t - fault.end_time) / 0.3;
            
            for k = 1:n_bus
                % 电压恢复
                V_normal = abs(results.V_pre(k));
                V_fault = results.V_mag(k, find(sim_time >= fault.start_time, 1));
                
                % 线性插值恢复
                results.V_mag(k, t_idx) = V_fault + (V_normal - V_fault) * recovery_ratio;
                results.V_angle(k, t_idx) = results.V_angle(k, t_idx-1) * (1 - recovery_ratio);
                results.V_complex(k, t_idx) = results.V_mag(k, t_idx) * ...
                    exp(1j * results.V_angle(k, t_idx) * pi/180);
                
                % 功率恢复
                P_normal = mpc.bus(k, 3)/baseMVA;
                Q_normal = mpc.bus(k, 4)/baseMVA;
                
                results.P_inject(k, t_idx) = results.P_inject(k, t_idx-1) + ...
                    (P_normal - results.P_inject(k, t_idx-1)) * recovery_ratio;
                results.Q_inject(k, t_idx) = results.Q_inject(k, t_idx-1) + ...
                    (Q_normal - results.Q_inject(k, t_idx-1)) * recovery_ratio;
                
                % 电流计算
                V_k_complex = results.V_complex(k, t_idx);
                I_k_complex = 0;
                for j = 1:n_bus
                    if j ~= k && Ybus(k, j) ~= 0
                        I_k_complex = I_k_complex + Ybus(k, j) * results.V_complex(j, t_idx);
                    end
                end
                I_k_complex = I_k_complex + Ybus(k, k) * V_k_complex;
                results.I_complex(k, t_idx) = I_k_complex;
            end
            
        else
            % 正常状态
            for k = 1:n_bus
                results.V_mag(k, t_idx) = abs(results.V_pre(k));
                results.V_angle(k, t_idx) = angle(results.V_pre(k))*180/pi;
                results.V_complex(k, t_idx) = results.V_pre(k);
                results.I_fault(k, t_idx) = 0;
                results.P_inject(k, t_idx) = mpc.bus(k, 3)/baseMVA;
                results.Q_inject(k, t_idx) = mpc.bus(k, 4)/baseMVA;
                
                % 计算正常状态电流
                V_k_complex = results.V_pre(k);
                I_k_complex = 0;
                for j = 1:n_bus
                    if j ~= k && Ybus(k, j) ~= 0
                        I_k_complex = I_k_complex + Ybus(k, j) * results.V_pre(j);
                    end
                end
                I_k_complex = I_k_complex + Ybus(k, k) * V_k_complex;
                results.I_complex(k, t_idx) = I_k_complex;
            end
        end
    end
    
    % 计算关键指标
    results.max_voltage_drop = zeros(n_bus, 1);  % 最大电压跌落
    results.voltage_recovery_time = zeros(n_bus, 1);  % 电压恢复时间
    results.affected_nodes = [];  % 受影响节点
    
    % 找到故障期间的时间范围
    fault_time_idx = find(sim_time >= fault.start_time & sim_time <= fault.end_time);
    
    for k = 1:n_bus
        if ~isempty(fault_time_idx)
            % 找到故障期间最小电压
            min_V = min(results.V_mag(k, fault_time_idx));
            results.max_voltage_drop(k) = (abs(results.V_pre(k)) - min_V) / abs(results.V_pre(k)) * 100;
            
            % 判断是否为受影响节点（电压跌落>5%）
            if results.max_voltage_drop(k) > 5
                results.affected_nodes = [results.affected_nodes; k];
            end
        end
        
        % 计算电压恢复时间（恢复到95%以上）
        recovery_time_idx = find(sim_time > fault.end_time);
        if ~isempty(recovery_time_idx)
            for idx = recovery_time_idx
                if results.V_mag(k, idx) >= 0.95 * abs(results.V_pre(k))
                    results.voltage_recovery_time(k) = sim_time(idx) - fault.end_time;
                    break;
                end
            end
        end
    end
    
    % 保存增强的数据到文件
    save_fault_node_data_enhanced(results, fault_id);
end

function generate_fault_detailed_report_enhanced(mpc, fault, results, fault_id)
    % 增强版：生成单个故障的详细报告，包含完整电气参数
    n_bus = size(mpc.bus, 1);
    
    % 创建报告文件
    filename = sprintf('Fault_%d_Node_%d_%s_Report_Enhanced.txt', fault_id, fault.bus, fault.type);
    fid = fopen(filename, 'w');
    
    fprintf(fid, '================================================================\n');
    fprintf(fid, '故障场景 %d 详细分析报告（增强版）\n', fault_id);
    fprintf(fid, '================================================================\n\n');
    
    fprintf(fid, '故障信息:\n');
    fprintf(fid, '  位置: 节点%d\n', fault.bus);
    fprintf(fid, '  类型: %s\n', fault.type);
    fprintf(fid, '  阻抗: Rf=%.4f, Xf=%.4f pu\n', fault.Rf, fault.Xf);
    fprintf(fid, '  时间: %.2f-%.2f秒 (持续时间: %.2f秒)\n', ...
        fault.start_time, fault.end_time, fault.duration);
    fprintf(fid, '  描述: %s\n\n', fault.description);
    
    % 找到故障期间的稳态时刻
    fault_time_idx = find(results.sim_time >= (fault.start_time + fault.duration/2), 1);
    
    % 所有节点电气参数表
    fprintf(fid, '所有节点电气参数（故障期间稳态值）:\n');
    fprintf(fid, '%-4s %-10s %-10s %-10s %-10s %-10s %-10s %-12s\n', ...
        '节点', '|V|(pu)', '∠V(°)', 'P(pu)', 'Q(pu)', 'ΔV(%)', '|I|(pu)', '状态');
    fprintf(fid, '%s\n', repmat('-', 85, 1));
    
    for node = 1:n_bus
        V_mag = results.V_mag(node, fault_time_idx);
        V_angle = results.V_angle(node, fault_time_idx);
        P_val = results.P_inject(node, fault_time_idx);
        Q_val = results.Q_inject(node, fault_time_idx);
        V_pre = abs(results.V_pre(node));
        V_change_percent = (V_mag - V_pre) / V_pre * 100;
        I_mag = abs(results.I_complex(node, fault_time_idx));
        
        if node == fault.bus
            status = '故障点';
        elseif V_change_percent < -30
            status = '严重跌落';
        elseif V_change_percent < -15
            status = '中等跌落';
        elseif V_change_percent < -5
            status = '轻微跌落';
        elseif V_change_percent > 10
            status = '电压升高';
        else
            status = '正常';
        end
        
        fprintf(fid, '%4d %10.4f %10.2f %10.4f %10.4f %10.2f %10.4f %12s\n', ...
            node, V_mag, V_angle, P_val, Q_val, V_change_percent, I_mag, status);
    end
    
    % 统计信息
    fprintf(fid, '\n============ 统计摘要 ============\n');
    
    V_mags = results.V_mag(:, fault_time_idx);
    V_changes = (V_mags - abs(results.V_pre)) ./ abs(results.V_pre) * 100;
    
    low_v_count = sum(V_mags < 0.85);
    high_v_count = sum(V_mags > 1.15);
    severe_drop_count = sum(V_changes < -30);
    moderate_drop_count = sum(V_changes >= -30 & V_changes < -15);
    mild_drop_count = sum(V_changes >= -15 & V_changes < -5);
    
    fprintf(fid, '  节点总数: %d\n', n_bus);
    fprintf(fid, '  低压节点（<0.85 pu）: %d个\n', low_v_count);
    fprintf(fid, '  过压节点（>1.15 pu）: %d个\n', high_v_count);
    fprintf(fid, '  严重电压跌落（>30%%）: %d个节点\n', severe_drop_count);
    fprintf(fid, '  中等电压跌落（15-30%%）: %d个节点\n', moderate_drop_count);
    fprintf(fid, '  轻微电压跌落（5-15%%）: %d个节点\n\n', mild_drop_count);
    
    % 受影响最严重的10个节点
    fprintf(fid, '受影响最严重的10个节点:\n');
    fprintf(fid, '%-6s %-12s %-15s %-15s %-10s %-10s\n', ...
        '节点', '最大跌落(%)', '故障时电压(pu)', '恢复时间(秒)', 'P(pu)', 'Q(pu)');
    fprintf(fid, '%s\n', repmat('-', 75, 1));
    
    [~, sorted_idx] = sort(results.max_voltage_drop, 'descend');
    for i = 1:min(10, n_bus)
        node = sorted_idx(i);
        if results.max_voltage_drop(node) > 0
            min_V = min(results.V_mag(node, find(results.sim_time >= fault.start_time & results.sim_time <= fault.end_time)));
            P_val = results.P_inject(node, fault_time_idx);
            Q_val = results.Q_inject(node, fault_time_idx);
            fprintf(fid, '%-6d %-12.1f %-15.3f %-15.3f %-10.4f %-10.4f\n', ...
                node, results.max_voltage_drop(node), min_V, ...
                results.voltage_recovery_time(node), P_val, Q_val);
        end
    end
    
    % 发电机节点影响分析
    fprintf(fid, '\n发电机节点影响分析:\n');
    gen_buses = mpc.gen(:, 1);
    fprintf(fid, '%-6s %-10s %-10s %-10s %-10s %-10s %-10s\n', ...
        '发电机', '|V|(pu)', '∠V(°)', 'P(pu)', 'Q(pu)', 'ΔV(%)', '状态');
    fprintf(fid, '%s\n', repmat('-', 70, 1));
    
    for i = 1:length(gen_buses)
        node = gen_buses(i);
        if node <= n_bus
            V_mag = results.V_mag(node, fault_time_idx);
            V_angle = results.V_angle(node, fault_time_idx);
            P_val = results.P_inject(node, fault_time_idx);
            Q_val = results.Q_inject(node, fault_time_idx);
            V_pre = abs(results.V_pre(node));
            V_change = (V_mag - V_pre) / V_pre * 100;
            
            if V_change < -30
                status = '严重跌落';
            elseif V_change < -15
                status = '中等跌落';
            elseif V_change < -5
                status = '轻微跌落';
            else
                status = '正常';
            end
            
            fprintf(fid, '%6d %10.4f %10.2f %10.4f %10.4f %10.2f %10s\n', ...
                node, V_mag, V_angle, P_val, Q_val, V_change, status);
        end
    end
    
    fclose(fid);
    
    % 创建增强的可视化图表
    create_fault_detailed_plots_enhanced(mpc, fault, results, fault_id);
end

function create_fault_detailed_plots_enhanced(mpc, fault, results, fault_id)
    % 增强版：创建单个故障的详细图表
    figure('Position', [100, 100, 1600, 1000], ...
        'Name', sprintf('故障%d: 节点%d %s（增强版）', fault_id, fault.bus, fault.type), ...
        'NumberTitle', 'off');
    
    % 找到故障期间的稳态时刻
    fault_time_idx = find(results.sim_time >= (fault.start_time + fault.duration/2), 1);
    
    % 1. 所有节点电压动态曲线
    subplot(3, 4, 1);
    plot(results.sim_time, results.V_mag);
    xlabel('时间 (秒)');
    ylabel('电压幅值 (pu)');
    title(sprintf('所有节点电压动态响应\n故障: 节点%d %s', fault.bus, fault.type));
    grid on;
    xlim([0.9, 1.4]);
    hold on;
    plot([fault.start_time, fault.start_time], [0, 1.1], 'r--', 'LineWidth', 1);
    plot([fault.end_time, fault.end_time], [0, 1.1], 'r--', 'LineWidth', 1);
    
    % 2. 故障期间电压幅值分布
    subplot(3, 4, 2);
    V_mags = results.V_mag(:, fault_time_idx);
    bar(1:length(V_mags), V_mags);
    xlabel('节点编号');
    ylabel('电压幅值 (pu)');
    title('故障期间电压幅值分布');
    grid on;
    hold on;
    plot([0, 40], [0.85, 0.85], 'r--', 'LineWidth', 1.5);
    plot([0, 40], [1.15, 1.15], 'r--', 'LineWidth', 1.5);
    ylim([0, 1.3]);
    
    % 3. 电压相角分布
    subplot(3, 4, 3);
    V_angles = results.V_angle(:, fault_time_idx);
    bar(1:length(V_angles), V_angles);
    xlabel('节点编号');
    ylabel('电压相角 (°)');
    title('故障期间电压相角分布');
    grid on;
    
    % 4. 电压变化率分布
    subplot(3, 4, 4);
    V_pre = abs(results.V_pre);
    V_changes = (V_mags - V_pre) ./ V_pre * 100;
    bar(1:length(V_changes), V_changes);
    xlabel('节点编号');
    ylabel('电压变化率 (%)');
    title('电压变化率分布');
    grid on;
    hold on;
    plot([0, 40], [-30, -30], 'r--', 'LineWidth', 1.5);
    plot([0, 40], [-15, -15], 'y--', 'LineWidth', 1.5);
    plot([0, 40], [-5, -5], 'g--', 'LineWidth', 1.5);
    
    % 5. 节点有功功率分布
    subplot(3, 4, 5);
    P_inject = results.P_inject(:, fault_time_idx);
    bar(1:length(P_inject), P_inject);
    xlabel('节点编号');
    ylabel('有功功率 (pu)');
    title('节点注入有功功率');
    grid on;
    
    % 6. 节点无功功率分布
    subplot(3, 4, 6);
    Q_inject = results.Q_inject(:, fault_time_idx);
    bar(1:length(Q_inject), Q_inject);
    xlabel('节点编号');
    ylabel('无功功率 (pu)');
    title('节点注入无功功率');
    grid on;
    
    % 7. 节点电流幅值分布
    subplot(3, 4, 7);
    I_mags = abs(results.I_complex(:, fault_time_idx));
    bar(1:length(I_mags), I_mags);
    xlabel('节点编号');
    ylabel('电流幅值 (pu)');
    title('节点电流幅值分布');
    grid on;
    
    % 8. 关键节点电压对比
    subplot(3, 4, 8);
    critical_nodes = [fault.bus, 30, 31, 39, 16, 4, 8, 20];
    colors = lines(length(critical_nodes));
    hold on;
    
    for i = 1:length(critical_nodes)
        node = critical_nodes(i);
        if node <= size(results.V_mag, 1)
            plot(results.sim_time, results.V_mag(node, :), ...
                'LineWidth', 2, 'Color', colors(i,:), ...
                'DisplayName', sprintf('节点%d', node));
        end
    end
    
    xlabel('时间 (秒)');
    ylabel('电压幅值 (pu)');
    title('关键节点电压对比');
    grid on;
    legend('Location', 'best');
    xlim([0.9, 1.4]);
    
    % 9. 电压-功率散点图
    subplot(3, 4, 9);
    scatter(V_mags, P_inject, 50, 'filled');
    xlabel('电压幅值 (pu)');
    ylabel('有功功率 (pu)');
    title('电压-有功功率关系');
    grid on;
    
    % 10. 电压-电流散点图
    subplot(3, 4, 10);
    scatter(V_mags, I_mags, 50, 'filled');
    xlabel('电压幅值 (pu)');
    ylabel('电流幅值 (pu)');
    title('电压-电流关系');
    grid on;
    
    % 11. 电压相量图
    subplot(3, 4, 11);
    polarscatter(V_angles*pi/180, V_mags, 50, 'filled');
    title('电压相量图');
    rlim([0, 1.2]);
    
    % 12. 系统拓扑标注
    subplot(3, 4, 12);
    create_topology_with_affected_nodes_enhanced(mpc, fault, results, fault_time_idx);
    
    % 保存图表
    saveas(gcf, sprintf('Fault_%d_Analysis_Enhanced.png', fault_id));
end

function create_topology_with_affected_nodes_enhanced(mpc, fault, results, time_idx)
    % 增强版：创建标注受影响节点的拓扑图
    n_bus = size(mpc.bus, 1);
    
    % 简化位置（圆形布局）
    theta = linspace(0, 2*pi, n_bus+1);
    theta = theta(1:end-1);
    r = 10;
    x_nodes = r * cos(theta);
    y_nodes = r * sin(theta);
    
    hold on;
    
    % 绘制线路
    for k = 1:min(20, size(mpc.branch, 1))
        from = mpc.branch(k, 1);
        to = mpc.branch(k, 2);
        plot([x_nodes(from), x_nodes(to)], [y_nodes(from), y_nodes(to)], ...
            'Color', [0.8, 0.8, 0.8], 'LineWidth', 0.5);
    end
    
    % 计算电压变化率
    V_mags = results.V_mag(:, time_idx);
    V_pre = abs(results.V_pre);
    V_changes = (V_mags - V_pre) ./ V_pre * 100;
    
    % 绘制节点，根据受影响程度着色
    for i = 1:n_bus
        if i == fault.bus
            % 故障节点 - 黑色
            color = [0, 0, 0];
            size_factor = 3.0;
        elseif V_changes(i) < -30
            % 红色 - 严重影响
            color = [1, 0, 0];
            size_factor = 2.0;
        elseif V_changes(i) < -15
            % 橙色 - 中等影响
            color = [1, 0.5, 0];
            size_factor = 1.5;
        elseif V_changes(i) < -5
            % 黄色 - 轻微影响
            color = [1, 1, 0];
            size_factor = 1.2;
        else
            % 蓝色 - 无影响
            color = [0, 0, 1];
            size_factor = 1.0;
        end
        
        plot(x_nodes(i), y_nodes(i), 'o', ...
            'MarkerSize', 8*size_factor, ...
            'MarkerFaceColor', color, ...
            'MarkerEdgeColor', 'k', ...
            'LineWidth', 1);
        
        % 标注节点编号和电压
        text(x_nodes(i)+0.5, y_nodes(i)+0.5, sprintf('%d\n%.3f', i, V_mags(i)), ...
            'FontSize', 7, 'HorizontalAlignment', 'center');
    end
    
    % 标注故障节点
    plot(x_nodes(fault.bus), y_nodes(fault.bus), 'p', ...
        'MarkerSize', 25, ...
        'MarkerFaceColor', 'k', ...
        'MarkerEdgeColor', 'k', ...
        'LineWidth', 2);
    
    % 添加图例
    legend_items = {'故障节点', '严重影响(>30%)', '中等影响(15-30%)', '轻微影响(5-15%)', '无影响'};
    legend_colors = {'k', 'r', [1,0.5,0], 'y', 'b'};
    for i = 1:length(legend_items)
        plot(NaN, NaN, 'o', 'MarkerSize', 10, ...
            'MarkerFaceColor', legend_colors{i}, ...
            'MarkerEdgeColor', 'k', ...
            'DisplayName', legend_items{i});
    end
    legend('Location', 'best', 'FontSize', 8);
    
    axis equal;
    axis off;
    title(sprintf('故障影响分布图\n节点%d %s', fault.bus, fault.type), 'FontSize', 12);
    hold off;
end

function save_fault_node_data_enhanced(results, fault_id)
    % 增强版：保存故障节点数据到CSV文件
    n_bus = size(results.V_mag, 1);
    
    % 找到故障期间的稳态时刻
    fault = results.fault;
    fault_time_idx = find(results.sim_time >= (fault.start_time + fault.duration/2), 1);
    
    % 创建数据表格
    bus_index = (1:n_bus)';
    V_mag = results.V_mag(:, fault_time_idx);
    V_angle = results.V_angle(:, fault_time_idx);
    P_inject = results.P_inject(:, fault_time_idx);
    Q_inject = results.Q_inject(:, fault_time_idx);
    
    % 计算电压变化率
    V_pre = abs(results.V_pre);
    V_change_percent = (V_mag - V_pre) ./ V_pre * 100;
    
    % 计算电流幅值
    I_mag = abs(results.I_complex(:, fault_time_idx));
    
    % 状态分类
    status = cell(n_bus, 1);
    for i = 1:n_bus
        if i == fault.bus
            status{i} = 'Fault_Point';
        elseif V_change_percent(i) < -30
            status{i} = 'Severe_Drop';
        elseif V_change_percent(i) < -15
            status{i} = 'Moderate_Drop';
        elseif V_change_percent(i) < -5
            status{i} = 'Mild_Drop';
        elseif V_change_percent(i) > 10
            status{i} = 'Voltage_Rise';
        else
            status{i} = 'Normal';
        end
    end
    
    % 创建表格
    fault_table = table(bus_index, V_mag, V_angle, P_inject, Q_inject, ...
        V_change_percent, I_mag, status, ...
        'VariableNames', {'Bus', 'V_mag_pu', 'V_angle_deg', 'P_inject_pu', ...
                         'Q_inject_pu', 'V_change_percent', 'I_mag_pu', 'Status'});
    
    % 保存为CSV文件
    filename = sprintf('Fault_%d_Node_%d_%s_All_Nodes_Data.csv', ...
        fault_id, fault.bus, fault.type);
    writetable(fault_table, filename);
    fprintf('✓ 所有节点电气参数已导出为: %s\n', filename);
    
    % 保存时间序列数据
    time_data = table(results.sim_time', 'VariableNames', {'Time_s'});
    for i = 1:min(10, n_bus)  % 只保存前10个节点的时间序列数据
        time_data.(sprintf('V_%d', i)) = results.V_mag(i, :)';
        time_data.(sprintf('P_%d', i)) = results.P_inject(i, :)';
        time_data.(sprintf('Q_%d', i)) = results.Q_inject(i, :)';
    end
    
    time_filename = sprintf('Fault_%d_Time_Series_Data.csv', fault_id);
    writetable(time_data, time_filename);
    fprintf('✓ 时间序列数据已导出为: %s\n', time_filename);
end

%% ========================================================================
% 原有的基本函数（保持不变）
%% ========================================================================

function mpc = create_ieee39_system()
    % 创建IEEE 39节点系统数据
    mpc.version = '2';
    mpc.baseMVA = 100;
    
    % 节点数据
    mpc.bus = [
        1   3   0       0       0   0   1   1.04    0       345 1   1.06   0.94;
        2   2   0       0       0   0   1   1.025   0       345 1   1.06   0.94;
        3   2   322     2.4     0   0   1   1.025   0       345 1   1.06   0.94;
        4   1   500     184     0   0   1   1.025   0       345 1   1.06   0.94;
        5   1   0       0       0   0   1   1.025   0       345 1   1.06   0.94;
        6   2   0       0       0   0   1   1.025   0       345 1   1.06   0.94;
        7   1   233.8   84      0   0   1   1.025   0       345 1   1.06   0.94;
        8   2   522     176.6   0   0   1   1.025   0       345 1   1.06   0.94;
        9   1   0       0       0   0   1   1.025   0       345 1   1.06   0.94;
        10  1   0       0       0   0   1   1.025   0       345 1   1.06   0.94;
        11  1   0       0       0   0   1   1.025   0       345 1   1.06   0.94;
        12  2   8.5     88      0   0   1   1.025   0       345 1   1.06   0.94;
        13  1   0       0       0   0   1   1.025   0       345 1   1.06   0.94;
        14  1   0       0       0   0   1   1.025   0       345 1   1.06   0.94;
        15  1   320     153     0   0   1   1.025   0       345 1   1.06   0.94;
        16  1   329.4   32.3    0   0   1   1.025   0       345 1   1.06   0.94;
        17  1   0       0       0   0   1   1.025   0       345 1   1.06   0.94;
        18  1   158     30      0   0   1   1.025   0       345 1   1.06   0.94;
        19  1   0       0       0   0   1   1.025   0       345 1   1.06   0.94;
        20  1   680     103     0   0   1   1.025   0       345 1   1.06   0.94;
        21  1   274     115     0   0   1   1.025   0       345 1   1.06   0.94;
        22  1   0       0       0   0   1   1.025   0       345 1   1.06   0.94;
        23  2   247.5   84.6    0   0   1   1.025   0       345 1   1.06   0.94;
        24  1   308.6   -92.2   0   0   1   1.025   0       345 1   1.06   0.94;
        25  1   224     47.2    0   0   1   1.025   0       345 1   1.06   0.94;
        26  1   139     17      0   0   1   1.025   0       345 1   1.06   0.94;
        27  2   281     75.5    0   0   1   1.025   0       345 1   1.06   0.94;
        28  1   206     27.6    0   0   1   1.025   0       345 1   1.06   0.94;
        29  1   283.5   26.9    0   0   1   1.025   0       345 1   1.06   0.94;
        30  2   0       0       0   0   1   1.0475  0       345 1   1.06   0.94;
        31  1   9.2     4.6     0   0   1   1.025   0       345 1   1.06   0.94;
        32  2   0       0       0   0   1   1.01    0       345 1   1.06   0.94;
        33  1   0       0       0   0   1   1.01    0       345 1   1.06   0.94;
        34  2   0       0       0   0   1   1.01    0       345 1   1.06   0.94;
        35  1   0       0       0   0   1   1.01    0       345 1   1.06   0.94;
        36  2   0       0       0   0   1   1.01    0       345 1   1.06   0.94;
        37  1   0       0       0   0   1   1.01    0       345 1   1.06   0.94;
        38  1   0       0       0   0   1   1.01    0       345 1   1.06   0.94;
        39  2   1104.6  250     0   0   1   1.03    0       345 1   1.06   0.94;
    ];
    
    % 发电机数据
    mpc.gen = [
        30  250   -16.2   300   -300  1.0475  100  1  250  10;
        31  677.9 221.2   300   -300  1.025   100  1  677.9 10;
        32  650   202.9   300   -300  1.01    100  1  650  10;
        33  632   108.9   300   -300  1.01    100  1  632  10;
        34  508   166.1   300   -300  1.01    100  1  508  10;
        35  650   210.5   300   -300  1.01    100  1  650  10;
        36  560   100.5   300   -300  1.01    100  1  560  10;
        37  540   0.1     300   -300  1.01    100  1  540  10;
        38  830   21.6    300   -300  1.01    100  1  830  10;
        39  1000  78.4    300   -300  1.03    100  1  1000 10;
    ];
    
    % 支路数据
    mpc.branch = [
        1   2   0.0035  0.0411  0.6987   250  250  250  0  0  1  -360  360;
        1   39  0.0010  0.0250  0.7500   250  250  250  0  0  1  -360  360;
        2   3   0.0013  0.0151  0.2572   250  250  250  0  0  1  -360  360;
        2   25  0.0070  0.0086  0.1460   250  250  250  0  0  1  -360  360;
        2   30  0.0000  0.0181  0.0000   250  250  250  1.025 0  1  -360  360;
        3   4   0.0013  0.0213  0.2214   250  250  250  0  0  1  -360  360;
        3   18  0.0011  0.0133  0.2138   250  250  250  0  0  1  -360  360;
        4   5   0.0008  0.0128  0.1342   250  250  250  0  0  1  -360  360;
        4   14  0.0008  0.0129  0.1382   250  250  250  0  0  1  -360  360;
        5   6   0.0002  0.0026  0.0434   250  250  250  0  0  1  -360  360;
        5   8   0.0008  0.0112  0.1476   250  250  250  0  0  1  -360  360;
        6   7   0.0006  0.0092  0.1130   250  250  250  0  0  1  -360  360;
        6   11  0.0007  0.0082  0.1389   250  250  250  0  0  1  -360  360;
        6   31  0.0000  0.0250  0.0000   250  250  250  1.07  0  1  -360  360;
        7   8   0.0004  0.0046  0.0780   250  250  250  0  0  1  -360  360;
        8   9   0.0023  0.0363  0.3804   250  250  250  0  0  1  -360  360;
        9   39  0.0010  0.0250  1.2000   250  250  250  0  0  1  -360  360;
        10  11  0.0004  0.0043  0.0729   250  250  250  0  0  1  -360  360;
        10  13  0.0004  0.0043  0.0729   250  250  250  0  0  1  -360  360;
        10  32  0.0000  0.0200  0.0000   250  250  250  1.07  0  1  -360  360;
        12  11  0.0016  0.0435  0.0000   250  250  250  1.006 0  1  -360  360;
        12  13  0.0016  0.0435  0.0000   250  250  250  1.006 0  1  -360  360;
        13  14  0.0009  0.0101  0.1723   250  250  250  0  0  1  -360  360;
        14  15  0.0018  0.0217  0.3660   250  250  250  0  0  1  -360  360;
        15  16  0.0009  0.0094  0.1710   250  250  250  0  0  1  -360  360;
        16  17  0.0007  0.0089  0.1342   250  250  250  0  0  1  -360  360;
        16  19  0.0016  0.0195  0.3040   250  250  250  0  0  1  -360  360;
        16  21  0.0008  0.0135  0.2548   250  250  250  0  0  1  -360  360;
        16  24  0.0003  0.0059  0.0680   250  250  250  0  0  1  -360  360;
        17  18  0.0007  0.0082  0.1319   250  250  250  0  0  1  -360  360;
        17  27  0.0013  0.0173  0.3216   250  250  250  0  0  1  -360  360;
        19  20  0.0007  0.0138  0.2090   250  250  250  0  0  1  -360  360;
        19  33  0.0007  0.0142  0.0000   250  250  250  1.07  0  1  -360  360;
        20  34  0.0009  0.0180  0.0000   250  250  250  1.009 0  1  -360  360;
        21  22  0.0008  0.0140  0.2565   250  250  250  0  0  1  -360  360;
        22  23  0.0006  0.0096  0.1846   250  250  250  0  0  1  -360  360;
        22  35  0.0000  0.0143  0.0000   250  250  250  1.025 0  1  -360  360;
        23  24  0.0022  0.0350  0.3610   250  250  250  0  0  1  -360  360;
        23  36  0.0005  0.0272  0.0000   250  250  250  1.0   0  1  -360  360;
        25  26  0.0032  0.0323  0.5310   250  250  250  0  0  1  -360  360;
        25  37  0.0006  0.0232  0.0000   250  250  250  1.025 0  1  -360  360;
        26  27  0.0014  0.0147  0.2396   250  250  250  0  0  1  -360  360;
        26  28  0.0043  0.0474  0.7800   250  250  250  0  0  1  -360  360;
        26  29  0.0057  0.0625  1.0290   250  250  250  0  0  1  -360  360;
        28  29  0.0014  0.0151  0.2490   250  250  250  0  0  1  -360  360;
        29  38  0.0008  0.0156  0.0000   250  250  250  1.025 0  1  -360  360;
    ];
end

function [Ybus, Zbus] = build_admittance_matrix(mpc)
    % 构建导纳矩阵和阻抗矩阵
    n_bus = size(mpc.bus, 1);
    Ybus = zeros(n_bus, n_bus);
    
    for k = 1:size(mpc.branch, 1)
        i = mpc.branch(k, 1);
        j = mpc.branch(k, 2);
        R = mpc.branch(k, 3);
        X = mpc.branch(k, 4);
        B = mpc.branch(k, 5);
        
        % 计算支路导纳
        Z = R + 1j*X;
        if abs(Z) > 0
            Y = 1/Z;
        else
            Y = 1e6;
        end
        
        % 添加到导纳矩阵
        Ybus(i, i) = Ybus(i, i) + Y + 1j*B/2;
        Ybus(j, j) = Ybus(j, j) + Y + 1j*B/2;
        Ybus(i, j) = Ybus(i, j) - Y;
        Ybus(j, i) = Ybus(j, i) - Y;
        
        % 处理变压器变比
        ratio = mpc.branch(k, 9);
        if ratio ~= 0 && ratio ~= 1
            Ybus(i, i) = Ybus(i, i) + Y*(1-ratio)/ratio^2;
            Ybus(i, j) = Ybus(i, j) + Y/ratio;
            Ybus(j, i) = Ybus(j, i) + Y/ratio;
        end
    end
    
    % 计算阻抗矩阵
    Zbus = inv(Ybus);
end

function faults = configure_fault_scenarios()
    % 配置故障场景
    fault_scenarios = {
        % bus, type, Rf, Xf, start_time, end_time, description
        16, '3LG',  0.001, 0.005, 1.0, 1.1, '三相金属性接地短路@关键节点16';
        21, 'LG-A', 0.001, 0.005, 1.0, 1.2, 'A相金属性接地短路@负荷节点21';
        4,  'LLG-BC', 0.01, 0.01,  1.0, 1.15,'BC两相接地短路@枢纽节点4';
        26, 'LL-BC',  0.05, 0.05,  1.0, 1.25, 'BC两相间短路@节点26';
        31, '3LG',  0.01,  0.02,   1.0, 1.08, '三相经阻抗接地短路@发电机节点31';
        39, 'LG-A', 0.005, 0.01,   1.0, 1.18, 'A相接地短路@平衡节点附近39';
    };
    
    faults = [];
    for i = 1:size(fault_scenarios, 1)
        fault.bus = fault_scenarios{i,1};
        fault.type = fault_scenarios{i,2};
        fault.Rf = fault_scenarios{i,3};
        fault.Xf = fault_scenarios{i,4};
        fault.start_time = fault_scenarios{i,5};
        fault.end_time = fault_scenarios{i,6};
        fault.description = fault_scenarios{i,7};
        fault.duration = fault.end_time - fault.start_time;
        fault.id = i;
        
        faults = [faults; fault];
    end
end

% 原有的其他函数保持不变
function generate_comparative_analysis(mpc, faults, varargin)
    % 生成综合对比分析
    n_faults = length(faults);
    
    figure('Position', [50, 50, 1400, 900], ...
        'Name', '故障场景综合对比分析', ...
        'NumberTitle', 'off');
    
    % 收集所有故障的数据
    all_max_drops = zeros(39, n_faults);
    all_recovery_times = zeros(39, n_faults);
    affected_counts = zeros(n_faults, 3);  % 严重、中等、轻微
    
    for i = 1:n_faults
        results = varargin{i};
        all_max_drops(:, i) = results.max_voltage_drop(1:39);
        all_recovery_times(:, i) = results.voltage_recovery_time(1:39);
        
        % 统计受影响节点
        affected_counts(i, 1) = sum(results.max_voltage_drop > 30);
        affected_counts(i, 2) = sum(results.max_voltage_drop > 15 & results.max_voltage_drop <= 30);
        affected_counts(i, 3) = sum(results.max_voltage_drop > 5 & results.max_voltage_drop <= 15);
    end
    
    % 1. 各故障最大电压跌落对比
    subplot(2, 3, 1);
    boxplot(all_max_drops, 'Labels', arrayfun(@(x) sprintf('F%d', x), 1:n_faults, 'UniformOutput', false));
    ylabel('最大电压跌落 (%)');
    xlabel('故障场景');
    title('各故障场景电压跌落分布对比');
    grid on;
    
    % 2. 受影响节点数量对比
    subplot(2, 3, 2);
    bar(affected_counts, 'stacked');
    xlabel('故障场景');
    ylabel('受影响节点数量');
    title('各故障受影响节点数量对比');
    legend('严重影响', '中等影响', '轻微影响', 'Location', 'best');
    grid on;
    set(gca, 'XTickLabel', arrayfun(@(x) sprintf('F%d', x), 1:n_faults, 'UniformOutput', false));
    
    % 3. 平均恢复时间对比
    subplot(2, 3, 3);
    avg_recovery_times = mean(all_recovery_times, 1);
    bar(avg_recovery_times);
    xlabel('故障场景');
    ylabel('平均恢复时间 (秒)');
    title('各故障平均电压恢复时间');
    grid on;
    set(gca, 'XTickLabel', arrayfun(@(x) sprintf('F%d', x), 1:n_faults, 'UniformOutput', false));
    
    % 4. 最易受影响节点排名
    subplot(2, 3, 4);
    node_vulnerability = mean(all_max_drops, 2);
    [~, sorted_idx] = sort(node_vulnerability, 'descend');
    bar(node_vulnerability(sorted_idx(1:10)));
    xlabel('节点排名');
    ylabel('平均电压跌落 (%)');
    title('最易受影响节点Top 10');
    grid on;
    set(gca, 'XTickLabel', arrayfun(@(x) num2str(sorted_idx(x)), 1:10, 'UniformOutput', false));
    
    % 5. 故障严重程度雷达图
    subplot(2, 3, 5);
    radar_data = [affected_counts(:,1)'; mean(all_max_drops)'; avg_recovery_times']';
    radar_data_normalized = radar_data ./ max(radar_data);
    
    angles = linspace(0, 2*pi, size(radar_data_normalized, 2)+1);
    angles = angles(1:end-1);
    
    hold on;
    colors = lines(n_faults);
    for i = 1:n_faults
        polarplot([angles, angles(1)], [radar_data_normalized(i,:), radar_data_normalized(i,1)], ...
            'Color', colors(i,:), 'LineWidth', 2, ...
            'DisplayName', sprintf('F%d', i));
    end
    
    thetaticks(angles * 180/pi);
    thetaticklabels({'严重节点数', '平均跌落', '恢复时间'});
    title('故障严重程度雷达图');
    legend('Location', 'best');
    
    % 6. 热力图：各节点在不同故障下的表现
    subplot(2, 3, 6);
    imagesc(all_max_drops');
    colorbar;
    xlabel('节点编号');
    ylabel('故障场景');
    title('节点电压跌落热力图');
    set(gca, 'XTick', 1:5:39);
    set(gca, 'YTick', 1:n_faults);
    set(gca, 'YTickLabel', arrayfun(@(x) sprintf('F%d', x), 1:n_faults, 'UniformOutput', false));
    
    % 保存图表
    saveas(gcf, 'Fault_Comparative_Analysis.png');
end

function generate_fault_impact_heatmap(mpc, faults, varargin)
    % 生成故障影响热力图
    n_faults = length(faults);
    n_bus = 39;
    
    figure('Position', [100, 100, 1200, 600], ...
        'Name', '故障影响热力图', ...
        'NumberTitle', 'off');
    
    % 创建综合影响矩阵
    impact_matrix = zeros(n_bus, 3);  % 电压跌落、恢复时间、影响频率
    
    for i = 1:n_faults
        results = varargin{i};
        impact_matrix(:, 1) = impact_matrix(:, 1) + results.max_voltage_drop(1:n_bus);
        impact_matrix(:, 2) = impact_matrix(:, 2) + results.voltage_recovery_time(1:n_bus);
        
        % 影响频率（该节点受影响次数）
        impact_matrix(:, 3) = impact_matrix(:, 3) + (results.max_voltage_drop(1:n_bus) > 5);
    end
    
    % 标准化
    impact_matrix_norm = zeros(size(impact_matrix));
    for i = 1:3
        if max(impact_matrix(:, i)) > 0
            impact_matrix_norm(:, i) = impact_matrix(:, i) / max(impact_matrix(:, i));
        end
    end
    
    % 计算综合脆弱性指数
    vulnerability_index = mean(impact_matrix_norm, 2);
    
    % 绘制脆弱性热力图
    subplot(1, 2, 1);
    imagesc(impact_matrix_norm');
    colorbar;
    xlabel('节点编号');
    ylabel('指标');
    title('节点脆弱性热力图');
    set(gca, 'XTick', 1:5:39);
    set(gca, 'YTick', 1:3);
    set(gca, 'YTickLabel', {'电压跌落', '恢复时间', '影响频率'});
    
    % 绘制脆弱性排名
    subplot(1, 2, 2);
    [~, sorted_idx] = sort(vulnerability_index, 'descend');
    bar(vulnerability_index(sorted_idx(1:15)));
    xlabel('节点排名');
    ylabel('综合脆弱性指数');
    title('节点脆弱性排名Top 15');
    grid on;
    set(gca, 'XTickLabel', arrayfun(@(x) num2str(sorted_idx(x)), 1:15, 'UniformOutput', false));
    
    % 保存脆弱性分析结果
    save_vulnerability_analysis(sorted_idx, vulnerability_index, impact_matrix);
    
    % 保存图表
    saveas(gcf, 'Node_Vulnerability_Analysis.png');
end

function save_vulnerability_analysis(sorted_idx, vulnerability_index, impact_matrix)
    % 保存脆弱性分析结果
    n_top = min(20, length(sorted_idx));
    
    filename = 'Node_Vulnerability_Analysis.csv';
    fid = fopen(filename, 'w');
    
    fprintf(fid, '排名,节点编号,综合脆弱性指数,平均电压跌落(%%),平均恢复时间(s),影响频率\n');
    
    for i = 1:n_top
        node = sorted_idx(i);
        fprintf(fid, '%d,%d,%.4f,%.1f,%.3f,%d\n', ...
            i, node, vulnerability_index(node), ...
            impact_matrix(node,1)/6, impact_matrix(node,2)/6, impact_matrix(node,3));
    end
    
    fclose(fid);
    fprintf('脆弱性分析结果已保存到: %s\n', filename);
end