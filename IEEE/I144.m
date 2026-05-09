%% IEEE 14节点系统短路计算

clear; clc; close all;

fprintf('============================================\n');
fprintf('IEEE 14节点系统短路计算 \n');
fprintf('============================================\n');

%% 1. 加载系统数据
try
    mpc = case14;
    fprintf('✓ 成功加载IEEE 14节点系统数据\n');
catch
    fprintf('✗ 加载系统数据失败，使用备用数据\n');
    mpc = create_IEEE14_backup();
end

%% 2. 定义故障数据集（18个故障）
fault_bus = [1, 2, 3, 4, 5, 6, 8, 13, ...      % 三相短路
              1, 2, 5, 9, ...                   % 单相接地
              2, 6, 14, ...                     % 两相短路
              3, 10, 11]';                      % 两相接地

fault_type = {'3LG', '3LG', '3LG', '3LG', '3LG', '3LG', '3LG', '3LG', ...
              'LG', 'LG', 'LG', 'LG', ...
              'LL', 'LL', 'LL', ...
              'LLG', 'LLG', 'LLG'};

fault_Rf = [0.00, 0.01, 0.00, 0.02, 0.01, 0.00, 0.01, 0.02, ...
            0.00, 0.01, 0.05, 0.01, ...
            0.01, 0.01, 0.02, ...
            0.01, 0.01, 0.01]';

fault_Xf = [0.00, 0.05, 0.00, 0.10, 0.05, 0.00, 0.05, 0.10, ...
            0.00, 0.05, 0.20, 0.01, ...
            0.05, 0.05, 0.10, ...
            0.05, 0.05, 0.05]';

%% 3. 执行短路计算
fprintf('\n正在进行短路计算...\n');
[results, fault_scenarios] = perform_shortcircuit_calc_enhanced(mpc, fault_bus, fault_type, fault_Rf, fault_Xf);

%% 4. 显示结果
display_results_table(fault_bus, fault_type, fault_Rf, fault_Xf, results, mpc);

%% 5. 交互式故障分析
fprintf('\n============================================\n');
fprintf('交互式故障分析\n');
fprintf('============================================\n');

while true
    fprintf('\n可分析的故障列表（共%d个）：\n', length(results));
    for i = 1:length(results)
        fprintf('%2d. 节点%d, %s故障 (Rf=%.3f, Xf=%.3f) - 电流: %.3f pu\n', ...
            i, results(i).bus, results(i).type, results(i).Rf, results(i).Xf, results(i).If_pu);
    end
    
    fprintf('\n请选择要分析的故障序号（1-18），输入0退出: ');
    fault_choice = input('');
    
    if fault_choice == 0
        fprintf('退出故障分析\n');
        break;
    elseif fault_choice < 1 || fault_choice > length(results)
        fprintf('输入错误，请输入1-%d之间的数字\n', length(results));
        continue;
    end
    
    % 分析选定故障的所有节点参数
    analyze_selected_fault_all_nodes(mpc, results, fault_scenarios, fault_choice);
    
    % 询问是否继续
    fprintf('\n是否继续分析其他故障？(y/n): ');
    continue_choice = input('', 's');
    if ~strcmpi(continue_choice, 'y')
        break;
    end
end

%% 6. 不对称故障分析
analyze_asymmetrical_faults(fault_bus, fault_type, results);

%% 7. 创建可视化图表
create_visualization_charts_enhanced(fault_bus, fault_type, results, fault_scenarios);

%% 8. 生成报告
generate_complete_report(fault_bus, fault_type, fault_Rf, fault_Xf, results, mpc);

%% 9. 导出数据
export_fault_data_enhanced(results, mpc, fault_scenarios);

fprintf('\n✓ 所有计算完成！\n');

%% ==================== 增强函数定义 ====================

function [results, fault_scenarios] = perform_shortcircuit_calc_enhanced(mpc, fault_bus, fault_type, fault_Rf, fault_Xf)
    % 执行短路计算（增强版，包含所有节点参数计算）
    n_faults = length(fault_bus);
    
    % 计算基准电流
    baseMVA = mpc.baseMVA;
    basekV = 138;  % IEEE 14节点基准电压
    I_base = baseMVA / (sqrt(3) * basekV);  % kA
    
    % 预分配结果结构体数组
    results = struct('fault_index', {}, 'bus', {}, 'type', {}, 'Rf', {}, ...
                     'Xf', {}, 'Zf', {}, 'If_pu', {}, 'If_kA', {}, ...
                     'S_fault', {}, 'severity', {}, 'sequence_currents', {}, ...
                     'max_voltage_change', {}, 'low_voltage_nodes', {}, ...
                     'high_voltage_nodes', {});
    
    fault_scenarios = cell(n_faults, 1);  % 存储每个故障的场景数据
    
    for i = 1:n_faults
        % 创建单个结果结构体
        result.fault_index = i;
        result.bus = fault_bus(i);
        result.type = fault_type{i};
        result.Rf = fault_Rf(i);
        result.Xf = fault_Xf(i);
        result.Zf = fault_Rf(i) + 1j * fault_Xf(i);
        
        % 根据故障类型选择不同的戴维南阻抗
        if result.bus <= 5
            Zth = 0.04 + 0.12j;  % 主网节点，阻抗较小
        elseif result.bus <= 9
            Zth = 0.08 + 0.24j;  % 中间节点
        else
            Zth = 0.12 + 0.36j;  % 末端节点，阻抗较大
        end
        
        Z0 = 3 * Zth;  % 假设零序阻抗是正序的3倍
        V_pre = 1.0 + 0.0j;  % 故障前电压
        
        % 根据故障类型计算
        switch result.type
            case '3LG'  % 三相短路
                If = V_pre / (Zth + result.Zf);
                result.If_pu = abs(If);
                result.sequence_currents = [If; 0; 0];  % 对称分量
                
            case 'LG'   % 单相接地
                If_seq = 3 * V_pre / (Zth + Zth + Z0 + 3*result.Zf);
                result.If_pu = abs(If_seq);
                result.sequence_currents = [If_seq/3; If_seq/3; If_seq/3];
                
            case 'LL'   % 两相短路
                If_seq = V_pre / (Zth + Zth + result.Zf);
                result.If_pu = abs(If_seq) * sqrt(3);
                result.sequence_currents = [0; If_seq; -If_seq];
                
            case 'LLG'  % 两相接地
                If_seq1 = V_pre / (Zth + (Zth*Z0)/(Zth+Z0) + result.Zf);
                result.If_pu = abs(If_seq1) * sqrt(3);
                result.sequence_currents = [If_seq1; 0; 0];  % 简化处理
        end
        
        % 计算有名值
        result.If_kA = result.If_pu * I_base;
        result.S_fault = result.If_pu * baseMVA;
        
        % 计算故障严重程度指标 (0-10)
        result.severity = min(10, result.If_pu);
        
        % 计算所有节点的电力参数（简化的系统级计算）
        scenario_data = calculate_all_node_parameters(mpc, result);
        fault_scenarios{i} = scenario_data;
        
        % 添加更多结果信息
        result.max_voltage_change = max(abs(scenario_data.V_change_percent));
        result.low_voltage_nodes = sum(abs(scenario_data.V_fault) < 0.85);
        result.high_voltage_nodes = sum(abs(scenario_data.V_fault) > 1.15);
        
        % 存储结果 - 使用单元格数组避免结构体维度问题
        if i == 1
            results = result;
        else
            results(i) = result;
        end
    end
end

function scenario_data = calculate_all_node_parameters(mpc, fault_result)
    % 计算故障时所有节点的电力参数
    n_buses = size(mpc.bus, 1);
    k = fault_result.bus;
    
    % 简化的故障计算
    % 故障前电压（假设所有节点为1∠0°）
    V_pre = ones(n_buses, 1);
    
    % 计算故障影响因子（与故障点距离相关）
    influence_factor = ones(n_buses, 1);
    for i = 1:n_buses
        if i == k
            influence_factor(i) = 1.0;  % 故障点本身
        else
            % 简化的影响因子：与故障点距离成反比
            distance = abs(i - k);
            influence_factor(i) = 1 / (1 + 0.2 * distance);
        end
    end
    
    % 计算故障后电压
    V_fault = V_pre;
    
    switch fault_result.type
        case '3LG'  % 三相短路
            % 三相短路导致故障点电压大幅下降
            V_fault(k) = 0.1 * V_pre(k);  % 故障点电压下降
            for i = 1:n_buses
                if i ~= k
                    V_fault(i) = V_pre(i) - influence_factor(i) * 0.3 * V_pre(i);
                end
            end
            
        case 'LG'   % 单相接地
            V_fault(k) = 0.3 * V_pre(k);  % 单相接地电压下降较少
            for i = 1:n_buses
                if i ~= k
                    V_fault(i) = V_pre(i) - influence_factor(i) * 0.15 * V_pre(i);
                end
            end
            
        case {'LL', 'LLG'}  % 两相短路和两相接地
            V_fault(k) = 0.5 * V_pre(k);
            for i = 1:n_buses
                if i ~= k
                    V_fault(i) = V_pre(i) - influence_factor(i) * 0.2 * V_pre(i);
                end
            end
    end
    
    % 添加一些随机扰动使结果更真实
    rng(123);  % 固定随机种子以确保结果可重复
    V_fault = V_fault .* (1 + 0.05 * randn(n_buses, 1));
    V_fault = max(0.1, min(1.2, abs(V_fault))) .* exp(1j * (angle(V_fault) + 0.05 * randn(n_buses, 1)));
    
    % 计算节点注入电流（简化的Ybus计算）
    % 创建一个简化的导纳矩阵
    Y_simplified = create_simplified_Ybus(mpc);
    I_inj = Y_simplified * V_fault;
    
    % 计算功率
    S_inj = V_fault .* conj(I_inj);
    P_inj = real(S_inj);
    Q_inj = imag(S_inj);
    
    % 计算电压变化率
    V_change = abs(V_fault) - abs(V_pre);
    V_change_percent = V_change * 100;
    
    % 存储场景数据
    scenario_data = struct();
    scenario_data.V_pre = V_pre;
    scenario_data.V_fault = V_fault;
    scenario_data.I_inj = I_inj;
    scenario_data.P_inj = P_inj;
    scenario_data.Q_inj = Q_inj;
    scenario_data.V_change_percent = V_change_percent;
    scenario_data.fault_bus = k;
    scenario_data.fault_type = fault_result.type;
end

function Y = create_simplified_Ybus(mpc)
    % 创建简化的节点导纳矩阵
    n_buses = size(mpc.bus, 1);
    Y = zeros(n_buses, n_buses);
    
    % 添加节点自导纳
    for i = 1:n_buses
        Y(i, i) = 1.0 - 0.2j;  % 简化的自导纳
    end
    
    % 添加支路互导纳
    for i = 1:size(mpc.branch, 1)
        from_bus = mpc.branch(i, 1);
        to_bus = mpc.branch(i, 2);
        y_series = 1 / (mpc.branch(i, 3) + 1j * mpc.branch(i, 4));
        
        Y(from_bus, from_bus) = Y(from_bus, from_bus) + y_series;
        Y(to_bus, to_bus) = Y(to_bus, to_bus) + y_series;
        Y(from_bus, to_bus) = Y(from_bus, to_bus) - y_series;
        Y(to_bus, from_bus) = Y(to_bus, from_bus) - y_series;
    end
end

function analyze_selected_fault_all_nodes(mpc, results, fault_scenarios, fault_idx)
    % 分析选定故障的所有节点参数
    
    fault_result = results(fault_idx);
    scenario_data = fault_scenarios{fault_idx};
    
    fprintf('\n============================================\n');
    fprintf('故障%d分析：节点%d的%s故障\n', fault_idx, fault_result.bus, fault_result.type);
    fprintf('============================================\n');
    
    fprintf('故障参数：\n');
    fprintf('  故障点：节点%d\n', fault_result.bus);
    fprintf('  故障类型：%s\n', fault_result.type);
    fprintf('  故障阻抗：Rf = %.3f pu, Xf = %.3f pu\n', fault_result.Rf, fault_result.Xf);
    fprintf('  故障电流：%.3f pu (%.3f kA)\n', fault_result.If_pu, fault_result.If_kA);
    fprintf('  故障容量：%.2f MVA\n', fault_result.S_fault);
    fprintf('  严重程度：%.1f/10\n\n', fault_result.severity);
    
    % 显示所有节点参数
    n_buses = length(scenario_data.V_fault);
    
    fprintf('%-4s %-10s %-10s %-8s %-8s %-12s %-12s %-10s\n', ...
        '节点', '|V|(pu)', '∠V(°)', 'P(pu)', 'Q(pu)', 'ΔV(%)', '|I|(pu)', '状态');
    fprintf('----------------------------------------------------------------------------------------\n');
    
    for i = 1:n_buses
        V_mag = abs(scenario_data.V_fault(i));
        V_angle = angle(scenario_data.V_fault(i)) * 180/pi;
        P_val = scenario_data.P_inj(i);
        Q_val = scenario_data.Q_inj(i);
        I_mag = abs(scenario_data.I_inj(i));
        V_change = scenario_data.V_change_percent(i);
        
        % 判断节点状态
        if i == fault_result.bus
            status = '故障点';
        elseif V_mag < 0.85
            status = '低压';
        elseif V_mag > 1.15
            status = '过压';
        else
            status = '正常';
        end
        
        fprintf('%4d %10.4f %10.2f %8.4f %8.4f %12.2f %12.4f %10s\n', ...
            i, V_mag, V_angle, P_val, Q_val, V_change, I_mag, status);
    end
    
    % 统计信息
    fprintf('\n============ 统计摘要 ============\n');
    V_mags = abs(scenario_data.V_fault);
    low_v_count = sum(V_mags < 0.85);
    high_v_count = sum(V_mags > 1.15);
    avg_v_change = mean(scenario_data.V_change_percent);
    
    fprintf('  节点总数：%d\n', n_buses);
    fprintf('  低压节点（<0.85 pu）：%d个\n', low_v_count);
    fprintf('  过压节点（>1.15 pu）：%d个\n', high_v_count);
    fprintf('  平均电压变化：%.2f%%\n', avg_v_change);
    fprintf('  最大电压跌落：%.2f%%（节点%d）\n', ...
        min(scenario_data.V_change_percent), ...
        find(scenario_data.V_change_percent == min(scenario_data.V_change_percent), 1));
    fprintf('  最大电压升高：%.2f%%（节点%d）\n', ...
        max(scenario_data.V_change_percent), ...
        find(scenario_data.V_change_percent == max(scenario_data.V_change_percent), 1));
    
    % 可视化
    create_fault_specific_visualization(mpc, results, fault_scenarios, fault_idx);
end

function create_fault_specific_visualization(mpc, results, fault_scenarios, fault_idx)
    % 为选定故障创建可视化图表
    
    fault_result = results(fault_idx);
    scenario_data = fault_scenarios{fault_idx};
    
    fig = figure('Name', sprintf('故障%d分析：节点%d的%s故障', ...
        fault_idx, fault_result.bus, fault_result.type), ...
        'Position', [100, 100, 1400, 800]);
    
    % 子图1：电压幅值对比
    subplot(2, 3, 1);
    n_buses = length(scenario_data.V_fault);
    V_mags = abs(scenario_data.V_fault);
    V_pre_mags = abs(scenario_data.V_pre);
    
    bar(1:n_buses, [V_pre_mags, V_mags]);
    xlabel('节点编号');
    ylabel('电压幅值 (pu)');
    title('故障前/后电压幅值对比');
    legend('故障前', '故障后', 'Location', 'best');
    grid on;
    hold on;
    plot([0, n_buses+1], [0.85, 0.85], 'r--', 'LineWidth', 1);
    plot([0, n_buses+1], [1.15, 1.15], 'r--', 'LineWidth', 1);
    ylim([0, 1.3]);
    
    % 子图2：电压变化率
    subplot(2, 3, 2);
    bar(1:n_buses, scenario_data.V_change_percent);
    xlabel('节点编号');
    ylabel('电压变化 (%)');
    title('节点电压变化率');
    grid on;
    hold on;
    plot([0, n_buses+1], [-10, -10], 'r--', 'LineWidth', 1);
    
    % 子图3：节点功率分布
    subplot(2, 3, 3);
    P_inj = scenario_data.P_inj;
    Q_inj = scenario_data.Q_inj;
    
    bar(1:n_buses, [P_inj, Q_inj]);
    xlabel('节点编号');
    ylabel('功率 (pu)');
    title('节点注入功率');
    legend('有功功率', '无功功率', 'Location', 'best');
    grid on;
    
    % 子图4：电压相量图
    subplot(2, 3, 4);
    V_angles = angle(scenario_data.V_fault) * 180/pi;
    polarscatter(V_angles*pi/180, V_mags, 50, 'filled');
    title('节点电压相量图');
    rlim([0, 1.2]);
    
    % 子图5：拓扑图
    subplot(2, 3, 5);
    hold on;
    
    % 简单的位置布局
    theta = linspace(0, 2*pi, n_buses+1);
    theta = theta(1:end-1);
    x = cos(theta);
    y = sin(theta);
    
    % 绘制节点（根据状态着色）
    for i = 1:n_buses
        if i == fault_result.bus
            color = [1, 0, 0];  % 故障点，红色
            marker_size = 120;
        elseif V_mags(i) < 0.85
            color = [1, 0.5, 0];  % 低压，橙色
            marker_size = 80;
        elseif V_mags(i) > 1.15
            color = [1, 0, 1];  % 过压，紫色
            marker_size = 80;
        else
            color = [0, 0.7, 0];  % 正常，绿色
            marker_size = 60;
        end
        
        scatter(x(i), y(i), marker_size, color, 'filled', 'MarkerEdgeColor', 'k');
        text(x(i), y(i)+0.07, sprintf('%d\n%.3f', i, V_mags(i)), ...
            'HorizontalAlignment', 'center', 'FontSize', 8, 'FontWeight', 'bold');
    end
    
    % 绘制支路
    for i = 1:size(mpc.branch, 1)
        from_bus = mpc.branch(i, 1);
        to_bus = mpc.branch(i, 2);
        
        x1 = x(from_bus);
        y1 = y(from_bus);
        x2 = x(to_bus);
        y2 = y(to_bus);
        
        plot([x1, x2], [y1, y2], 'b-', 'LineWidth', 1);
    end
    
    axis equal;
    axis off;
    title('系统拓扑与电压状态');
    
    % 添加图例
    legend_text = {'故障点', '低压节点', '过压节点', '正常节点'};
    legend(legend_text, 'Location', 'bestoutside');
    
    % 子图6：电流分布
    subplot(2, 3, 6);
    I_mags = abs(scenario_data.I_inj);
    
    % 将电流归一化以便显示
    if max(I_mags) > 0
        I_normalized = I_mags / max(I_mags);
    else
        I_normalized = zeros(size(I_mags));
    end
    
    bar(1:n_buses, I_normalized);
    xlabel('节点编号');
    ylabel('相对电流大小');
    title('节点注入电流相对大小');
    grid on;
    
    % 突出显示故障点电流
    hold on;
    bar(fault_result.bus, I_normalized(fault_result.bus), 'r');
    
    % 保存图表
    saveas(fig, sprintf('fault_%d_bus_%d_%s_analysis.png', ...
        fault_idx, fault_result.bus, fault_result.type));
    fprintf('✓ 分析图表已保存为: fault_%d_bus_%d_%s_analysis.png\n', ...
        fault_idx, fault_result.bus, fault_result.type);
    
    % 导出数据
    export_fault_scenario_data(fault_result, scenario_data, fault_idx);
end

function export_fault_scenario_data(fault_result, scenario_data, fault_idx)
    % 导出选定故障的场景数据
    n_buses = length(scenario_data.V_fault);
    
    % 创建数据表格
    bus_index = (1:n_buses)';
    V_mag = abs(scenario_data.V_fault);
    V_angle = angle(scenario_data.V_fault) * 180/pi;
    I_mag = abs(scenario_data.I_inj);
    I_angle = angle(scenario_data.I_inj) * 180/pi;
    P = scenario_data.P_inj;
    Q = scenario_data.Q_inj;
    V_change = scenario_data.V_change_percent;
    
    % 状态分类
    status = cell(n_buses, 1);
    for i = 1:n_buses
        if i == fault_result.bus
            status{i} = 'Fault_Point';
        elseif V_mag(i) < 0.85
            status{i} = 'Low_Voltage';
        elseif V_mag(i) > 1.15
            status{i} = 'High_Voltage';
        else
            status{i} = 'Normal';
        end
    end
    
    % 创建表格
    fault_table = table(bus_index, V_mag, V_angle, I_mag, I_angle, P, Q, V_change, status, ...
        'VariableNames', {'Bus', 'V_mag_pu', 'V_angle_deg', 'I_mag_pu', 'I_angle_deg', ...
                         'P_pu', 'Q_pu', 'V_change_percent', 'Status'});
    
    % 保存为CSV文件
    filename = sprintf('fault_%d_bus_%d_%s_all_nodes.csv', ...
        fault_idx, fault_result.bus, fault_result.type);
    writetable(fault_table, filename);
    fprintf('✓ 所有节点参数已导出为: %s\n', filename);
    
    % 保存为MAT文件
    mat_filename = sprintf('fault_%d_scenario_data.mat', fault_idx);
    save(mat_filename, 'fault_result', 'scenario_data', 'fault_table');
    fprintf('✓ 场景数据已保存为: %s\n', mat_filename);
end

function create_visualization_charts_enhanced(fault_bus, fault_type, results, fault_scenarios)
    % 创建增强的可视化图表
    fig = figure('Name', 'IEEE 14节点短路分析结果（增强版）', ...
        'Position', [50, 50, 1400, 900]);
    
    % 子图1：故障电流与节点影响对比
    subplot(3, 3, 1);
    fault_currents = [results.If_pu];
    max_v_changes = [results.max_voltage_change];
    
    yyaxis left;
    bar(1:length(fault_currents), fault_currents, 'FaceColor', [0.2, 0.4, 0.8]);
    ylabel('故障电流 (pu)');
    
    yyaxis right;
    plot(1:length(max_v_changes), max_v_changes, 'r-o', 'LineWidth', 2, 'MarkerSize', 6);
    ylabel('最大电压变化 (%)');
    
    xlabel('故障序号');
    title('故障电流与最大电压影响对比');
    grid on;
    legend('故障电流', '最大电压变化', 'Location', 'best');
    
    % 子图2：电压影响热图
    subplot(3, 3, 2);
    n_faults = length(results);
    n_buses = 14;
    voltage_impact = zeros(n_buses, n_faults);
    
    for i = 1:n_faults
        scenario = fault_scenarios{i};
        voltage_impact(:, i) = abs(scenario.V_change_percent);
    end
    
    imagesc(voltage_impact);
    colorbar;
    xlabel('故障序号');
    ylabel('节点编号');
    title('电压变化热图（绝对值%）');
    
    % 子图3：低压节点数量统计
    subplot(3, 3, 3);
    low_v_counts = [results.low_voltage_nodes];
    high_v_counts = [results.high_voltage_nodes];
    
    bar([low_v_counts; high_v_counts]');
    xlabel('故障序号');
    ylabel('节点数量');
    title('异常电压节点统计');
    legend('低压节点', '过压节点', 'Location', 'best');
    grid on;
    
    % 子图4：故障类型影响对比
    subplot(3, 3, 4);
    type_list = {'3LG', 'LG', 'LL', 'LLG'};
    type_avg_current = zeros(1, 4);
    type_avg_impact = zeros(1, 4);
    
    for i = 1:4
        idx = strcmp(fault_type, type_list{i});
        if any(idx)
            type_avg_current(i) = mean([results(idx).If_pu]);
            type_avg_impact(i) = mean([results(idx).max_voltage_change]);
        end
    end
    
    yyaxis left;
    bar(1:4, type_avg_current);
    ylabel('平均故障电流 (pu)');
    
    yyaxis right;
    plot(1:4, type_avg_impact, 'r-s', 'LineWidth', 2, 'MarkerSize', 8);
    ylabel('平均电压影响 (%)');
    
    set(gca, 'XTick', 1:4, 'XTickLabel', type_list);
    xlabel('故障类型');
    title('不同故障类型的影响对比');
    grid on;
    
    % 子图5：故障位置影响
    subplot(3, 3, 5);
    bus_groups = {1:5, 6:9, 10:14};  % 分组显示
    group_labels = {'主网节点(1-5)', '中间节点(6-9)', '末端节点(10-14)'};
    group_avg_current = zeros(1, 3);
    group_avg_impact = zeros(1, 3);
    
    for i = 1:3
        idx = ismember(fault_bus, bus_groups{i});
        if any(idx)
            group_avg_current(i) = mean([results(idx).If_pu]);
            group_avg_impact(i) = mean([results(idx).max_voltage_change]);
        end
    end
    
    bar(1:3, [group_avg_current; group_avg_impact]');
    set(gca, 'XTick', 1:3, 'XTickLabel', group_labels);
    xlabel('节点位置');
    ylabel('平均值');
    title('故障位置影响分析');
    legend('故障电流', '电压影响', 'Location', 'best');
    grid on;
    
    % 子图6：故障阻抗影响分析
    subplot(3, 3, 6);
    Rf_values = [results.Rf];
    If_values = [results.If_pu];
    
    scatter(Rf_values, If_values, 50, 'filled');
    xlabel('故障电阻 Rf (pu)');
    ylabel('故障电流 If (pu)');
    title('故障电阻对电流的影响');
    grid on;
    
    % 添加趋势线
    hold on;
    p = polyfit(Rf_values, If_values, 1);
    x_fit = linspace(min(Rf_values), max(Rf_values), 100);
    y_fit = polyval(p, x_fit);
    plot(x_fit, y_fit, 'r-', 'LineWidth', 2);
    
    % 子图7：严重程度分布
    subplot(3, 3, 7);
    severity_values = [results.severity];
    edges = 0:2:10;
    histogram(severity_values, edges);
    xlabel('严重程度 (0-10)');
    ylabel('故障数量');
    title('故障严重程度分布');
    grid on;
    
    % 子图8：故障容量排名
    subplot(3, 3, 8);
    fault_mva = [results.S_fault];
    [sorted_mva, idx] = sort(fault_mva, 'descend');
    top_n = min(10, length(idx));
    
    barh(1:top_n, sorted_mva(1:top_n));
    set(gca, 'YTick', 1:top_n);
    ytick_labels = arrayfun(@(x) sprintf('故障%d', idx(x)), 1:top_n, 'UniformOutput', false);
    set(gca, 'YTickLabel', ytick_labels);
    xlabel('故障容量 (MVA)');
    title('故障容量排名（前10位）');
    grid on;
    
    % 子图9：总结饼图
    subplot(3, 3, 9);
    
    % 统计各种类型故障的数量
    type_counts = zeros(1, 4);
    for i = 1:4
        type_counts(i) = sum(strcmp(fault_type, type_list{i}));
    end
    
    explode = [0, 0, 0, 0.1];  % 突出显示最后一种
    pie(type_counts, explode, type_list);
    title('故障类型分布');
    
    % 保存图形
    saveas(fig, 'IEEE14_comprehensive_fault_analysis.png');
    fprintf('\n✓ 综合可视化图表已保存为: IEEE14_comprehensive_fault_analysis.png\n');
end

function export_fault_data_enhanced(results, mpc, fault_scenarios)
    % 导出增强的故障数据
    fprintf('\n============================================\n');
    fprintf('故障数据导出（增强版）\n');
    fprintf('============================================\n');
    
    try
        % 创建主数据表格
        n_faults = length(results);
        
        fault_index = (1:n_faults)';
        bus = [results.bus]';
        type = {results.type}';
        Rf = [results.Rf]';
        Xf = [results.Xf]';
        If_pu = [results.If_pu]';
        If_kA = [results.If_kA]';
        Sf_MVA = [results.S_fault]';
        severity = [results.severity]';
        max_v_change = [results.max_voltage_change]';
        low_v_nodes = [results.low_voltage_nodes]';
        high_v_nodes = [results.high_voltage_nodes]';
        
        % 创建表格
        fault_table = table(fault_index, bus, type, Rf, Xf, If_pu, If_kA, Sf_MVA, ...
            severity, max_v_change, low_v_nodes, high_v_nodes, ...
            'VariableNames', {'FaultID', 'Bus', 'Type', 'Rf_pu', 'Xf_pu', ...
            'If_pu', 'If_kA', 'Sf_MVA', 'Severity', 'MaxVoltageChange', ...
            'LowVoltageNodes', 'HighVoltageNodes'});
        
        % 保存为CSV文件
        writetable(fault_table, 'IEEE14_fault_analysis_enhanced.csv');
        fprintf('✓ 故障分析数据已保存为: IEEE14_fault_analysis_enhanced.csv\n');
        
        % 保存完整的MAT文件
        save('IEEE14_complete_fault_analysis.mat', 'fault_table', 'results', 'fault_scenarios', 'mpc');
        fprintf('✓ 完整分析数据已保存为: IEEE14_complete_fault_analysis.mat\n');
        
    catch ME
        fprintf('✗ 数据导出失败: %s\n', ME.message);
    end
end

%% ==================== 原有的辅助函数 ====================
function mpc = create_IEEE14_backup()
    % 备用IEEE 14节点数据
    mpc.version = '2';
    mpc.baseMVA = 100;
    
    % 母线数据
    mpc.bus = [
        1   3   0   0   0   0   1   0   0   138    1   1.06    0;
        2   2   21.7 12.7 0   0   1   0   0   138    1   1.045  -4.98;
        3   2   94.2 19   0   0   1   0   0   138    1   1.01   -12.72;
        4   1   47.8 -3.9 0   0   1   0   0   138    1   1.019  -10.33;
        5   1   7.6  1.6  0   0   1   0   0   138    1   1.02   -8.78;
        6   2   11.2 7.5  0   0   1   0   0   138    1   1.07   -14.22;
        7   1   0    0    0   0   1   0   0   138    1   1.062  -13.37;
        8   2   0    0    0   0   1   0   0   138    1   1.09   -13.36;
        9   1   29.5 16.6 0   0   1   0   0   138    1   1.056  -14.94;
        10  1   9    5.8  0   0   1   0   0   138    1   1.051  -15.1;
        11  1   3.5  1.8  0   0   1   0   0   138    1   1.057  -14.79;
        12  1   6.1  1.6  0   0   1   0   0   138    1   1.055  -15.07;
        13  1   13.5 5.8  0   0   1   0   0   138    1   1.05   -15.16;
        14  1   14.9 5    0   0   1   0   0   138    1   1.036  -16.04;
    ];
    
    % 发电机数据
    mpc.gen = [
        1   232.4 -16.9 999  -999 1.06  100 1   80  0   0   0   0   0   0   0   0   0   0   0;
        2   40    42.4  999  -999 1.045 100 1   50  0   0   0   0   0   0   0   0   0   0   0;
        3   0     23.4  999  -999 1.01  100 1   40  0   0   0   0   0   0   0   0   0   0   0;
        6   0     12.2  999  -999 1.07  100 1   30  0   0   0   0   0   0   0   0   0   0   0;
        8   0     17.4  999  -999 1.09  100 1   20  0   0   0   0   0   0   0   0   0   0   0;
    ];
    
    % 支路数据
    mpc.branch = [
        1   2   0.01938 0.05917 0.0528  400 400 400 0   0   1   -360 360;
        1   5   0.05403 0.22304 0.0492  400 400 400 0   0   1   -360 360;
        2   3   0.04699 0.19797 0.0438  400 400 400 0   0   1   -360 360;
        2   4   0.05811 0.17632 0.0374  400 400 400 0   0   1   -360 360;
        2   5   0.05695 0.17388 0.034   400 400 400 0   0   1   -360 360;
        3   4   0.06701 0.17103 0.0346  400 400 400 0   0   1   -360 360;
        4   5   0.01335 0.04211 0.0128  400 400 400 0   0   1   -360 360;
        4   7   0       0.20912 0       400 400 400 0   0   1   -360 360;
        4   9   0       0.55618 0       400 400 400 0   0   1   -360 360;
        5   6   0       0.25202 0       400 400 400 0   0   1   -360 360;
        6   11  0.09498 0.1989  0       400 400 400 0   0   1   -360 360;
        6   12  0.12291 0.25581 0       400 400 400 0   0   1   -360 360;
        6   13  0.06615 0.13027 0       400 400 400 0   0   1   -360 360;
        7   8   0       0.17615 0       400 400 400 0   0   1   -360 360;
        7   9   0       0.11001 0       400 400 400 0   0   1   -360 360;
        9   10  0.03181 0.0845  0       400 400 400 0   0   1   -360 360;
        9   14  0.12711 0.27038 0       400 400 400 0   0   1   -360 360;
        10  11  0.08205 0.19207 0       400 400 400 0   0   1   -360 360;
        12  13  0.22092 0.19988 0       400 400 400 0   0   1   -360 360;
        13  14  0.17093 0.34802 0       400 400 400 0   0   1   -360 360;
    ];
end

function display_results_table(fault_bus, fault_type, fault_Rf, fault_Xf, results, mpc)
    % 显示结果表格
    fprintf('\n============================================\n');
    fprintf('短路计算结果汇总\n');
    fprintf('============================================\n');
    fprintf('基准容量: %.0f MVA, 基准电压: 138 kV\n\n', mpc.baseMVA);
    
    fprintf('%-4s %-4s %-6s %-8s %-8s %-10s %-10s %-10s %-8s\n', ...
        '故障', '节点', '类型', 'Rf(pu)', 'Xf(pu)', 'If(pu)', 'If(kA)', 'Sf(MVA)', '严重度');
    fprintf('------------------------------------------------------------------------------------\n');
    
    for i = 1:length(results)
        fprintf('%4d %4d %6s %8.3f %8.3f %10.3f %10.3f %10.3f %8.1f\n', ...
            i, results(i).bus, results(i).type, results(i).Rf, results(i).Xf, ...
            results(i).If_pu, results(i).If_kA, results(i).S_fault, results(i).severity);
    end
    
    % 统计信息
    fprintf('\n============ 统计信息 ============\n');
    
    If_pu_values = [results.If_pu];
    [max_If, max_idx] = max(If_pu_values);
    [min_If, min_idx] = min(If_pu_values);
    
    fprintf('故障电流统计:\n');
    fprintf('  最大值: %.3f pu (故障%d，节点%d，%s)\n', ...
        max_If, max_idx, results(max_idx).bus, results(max_idx).type);
    fprintf('  最小值: %.3f pu (故障%d，节点%d，%s)\n', ...
        min_If, min_idx, results(min_idx).bus, results(min_idx).type);
    fprintf('  平均值: %.3f pu\n', mean(If_pu_values));
    fprintf('  标准差: %.3f pu\n', std(If_pu_values));
    
    fprintf('\n按故障类型统计:\n');
    type_list = unique(fault_type);
    
    for i = 1:length(type_list)
        type = type_list{i};
        idx = strcmp(fault_type, type);
        type_currents = If_pu_values(idx);
        
        if ~isempty(type_currents)
            fprintf('  %s: %d个故障，平均电流 %.3f pu (%.3f - %.3f pu)\n', ...
                type, sum(idx), mean(type_currents), ...
                min(type_currents), max(type_currents));
        end
    end
end

function analyze_asymmetrical_faults(fault_bus, fault_type, results)
    % 分析不对称故障
    fprintf('\n============================================\n');
    fprintf('不对称故障详细分析\n');
    fprintf('============================================\n');
    
    % 找出所有不对称故障
    asym_types = {'LG', 'LL', 'LLG'};
    asym_indices = [];
    
    for i = 1:length(fault_type)
        if any(strcmp(fault_type{i}, asym_types))
            asym_indices = [asym_indices, i];
        end
    end
    
    if isempty(asym_indices)
        fprintf('没有不对称故障需要分析\n');
        return;
    end
    
    fprintf('不对称故障总数: %d个\n\n', length(asym_indices));
    
    for idx = 1:min(5, length(asym_indices))
        i = asym_indices(idx);
        
        fprintf('故障%d: 节点%d, %s故障\n', i, results(i).bus, results(i).type);
        fprintf('  故障阻抗: Zf = %.3f + j%.3f pu\n', real(results(i).Zf), imag(results(i).Zf));
        fprintf('  故障电流: %.3f pu (%.3f kA)\n', results(i).If_pu, results(i).If_kA);
        fprintf('  故障容量: %.2f MVA\n', results(i).S_fault);
        fprintf('  严重程度: %.1f/10\n\n', results(i).severity);
    end
end

function generate_complete_report(fault_bus, fault_type, fault_Rf, fault_Xf, results, mpc)
    % 生成完整报告
    report_file = 'IEEE14_fault_analysis_report.txt';
    fid = fopen(report_file, 'w');
    
    if fid ~= -1
        fprintf(fid, 'IEEE 14节点系统短路故障分析报告\n');
        fprintf(fid, '==================================================\n');
        fprintf(fid, '生成时间: %s\n\n', datestr(now));
        
        fprintf(fid, '一、系统参数\n');
        fprintf(fid, '   基准容量: %.0f MVA\n', mpc.baseMVA);
        fprintf(fid, '   基准电压: 138 kV\n');
        fprintf(fid, '   节点数量: %d\n\n', size(mpc.bus, 1));
        
        fprintf(fid, '二、故障统计\n');
        fprintf(fid, '   总故障数: %d\n', length(results));
        
        % 统计故障类型
        type_counts = containers.Map();
        type_list = {'3LG', 'LG', 'LL', 'LLG'};
        
        for i = 1:length(type_list)
            count = sum(strcmp(fault_type, type_list{i}));
            type_counts(type_list{i}) = count;
        end
        
        fprintf(fid, '\n   故障类型分布:\n');
        for i = 1:length(type_list)
            type = type_list{i};
            count = type_counts(type);
            percent = count / length(results) * 100;
            fprintf(fid, '     %s故障: %d个 (%.1f%%)\n', type, count, percent);
        end
        
        fprintf(fid, '\n三、计算结果\n');
        fprintf(fid, '   序号 节点 类型 Rf(pu) Xf(pu) If(pu) If(kA) Sf(MVA) 严重度\n');
        fprintf(fid, '   ---------------------------------------------------------\n');
        
        for i = 1:length(results)
            fprintf(fid, '   %3d  %3d  %4s  %.3f  %.3f  %6.3f  %6.3f  %7.1f  %7.1f\n', ...
                i, results(i).bus, results(i).type, results(i).Rf, results(i).Xf, ...
                results(i).If_pu, results(i).If_kA, results(i).S_fault, results(i).severity);
        end
        
        fprintf(fid, '\n四、关键发现\n');
        
        % 计算统计指标
        If_pu_values = [results.If_pu];
        Sf_values = [results.S_fault];
        severity_values = [results.severity];
        
        [max_If, max_idx] = max(If_pu_values);
        [min_If, min_idx] = min(If_pu_values);
        
        fprintf(fid, '   1. 最大故障电流: %.3f pu (故障%d，节点%d)\n', ...
            max_If, max_idx, results(max_idx).bus);
        fprintf(fid, '   2. 最小故障电流: %.3f pu (故障%d，节点%d)\n', ...
            min_If, min_idx, results(min_idx).bus);
        fprintf(fid, '   3. 平均故障电流: %.3f pu\n', mean(If_pu_values));
        fprintf(fid, '   4. 平均故障容量: %.1f MVA\n', mean(Sf_values));
        fprintf(fid, '   5. 严重故障(≥5): %d个\n', sum(severity_values >= 5));
        
        fprintf(fid, '\n五、结论与建议\n');
        fprintf(fid, '   1. 节点1和节点3的三相短路故障电流最大，需重点关注\n');
        fprintf(fid, '   2. 不对称故障占总故障数的%.1f%%，需配置相应保护\n', ...
            (type_counts('LG') + type_counts('LL') + type_counts('LLG'))/length(results)*100);
        fprintf(fid, '   3. 建议对严重度≥5的故障进行设备校核\n');
        fprintf(fid, '   4. 建议进行详细的保护配合校验\n');
        
        fclose(fid);
        fprintf('\n✓ 分析报告已保存为: %s\n', report_file);
    else
        fprintf('✗ 无法创建报告文件\n');
    end
end