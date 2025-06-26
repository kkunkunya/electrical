"""
灾后静态调度模型测试 - CVXPY版本
"""

import pytest
import numpy as np
import cvxpy as cp
from src.datasets.loader import load_system_data
from src.models.post_disaster_static import PostDisasterStaticModel


class TestPostDisasterStaticModel:
    """测试灾后静态调度模型（CVXPY版本）"""
    
    @pytest.fixture
    def system_data(self):
        """加载系统数据"""
        return load_system_data("data")
    
    @pytest.fixture
    def model(self, system_data):
        """创建模型实例"""
        return PostDisasterStaticModel(system_data)
    
    def test_model_creation(self, model):
        """测试模型创建"""
        assert model is not None
        assert model.problem is not None
        assert model.n_buses == 33
        assert model.n_branches == 32
        
    def test_variables_creation(self, model):
        """测试变量创建"""
        # 检查变量维度
        assert model.P_shed.shape == (model.n_buses,)
        assert model.Q_shed.shape == (model.n_buses,)
        assert model.P_gen.shape == (model.n_buses,)
        assert model.Q_gen.shape == (model.n_buses,)
        assert model.V_sqr.shape == (model.n_buses,)
        assert model.P_branch.shape == (model.n_branches,)
        assert model.Q_branch.shape == (model.n_branches,)
        assert model.I_sqr.shape == (model.n_branches,)
        
    def test_problem_structure(self, model):
        """测试问题结构"""
        # 检查是否为最小化问题
        assert isinstance(model.problem.objective, cp.Minimize)
        
        # 检查约束数量
        assert len(model.problem.constraints) > 0
        
    def test_solve_optimal(self, model):
        """测试求解最优解"""
        # 求解模型
        results = model.solve(verbose=False)
        
        # 检查求解状态
        assert results is not None
        assert results['status'] in ['optimal', 'optimal_inaccurate']
        
        # 检查结果字段
        assert 'objective' in results
        assert 'P_shed' in results
        assert 'V_kV' in results
        assert 'total_P_shed' in results
        assert 'max_voltage' in results
        assert 'min_voltage' in results
        
        # 检查电压约束
        assert 11.39 <= results['max_voltage'] <= 13.92
        assert 11.39 <= results['min_voltage'] <= 13.92
        
        # 检查目标函数值（应该大于等于0）
        assert results['objective'] >= 0
        
        # 记录基准值，用于回归测试
        print(f"\n基准目标函数值: {results['objective']:.2f}")
        
    def test_load_shedding_limits(self, model):
        """测试负荷削减约束"""
        results = model.solve(verbose=False)
        
        if results and results['P_shed']:
            for node, p_shed in results['P_shed'].items():
                # 检查削减量不超过负荷
                assert p_shed <= model.P_load[node-1] * 1000 + 1e-3
                
    def test_voltage_limits(self, model):
        """测试电压约束"""
        results = model.solve(verbose=False)
        
        if results:
            for node, v_kv in results['V_kV'].items():
                if node != 1:  # 非平衡节点
                    assert 11.39 <= v_kv <= 13.92
                    
    def test_power_balance(self, model):
        """测试功率平衡（近似）"""
        results = model.solve(verbose=False)
        
        if results:
            # 总发电
            total_gen = results['total_P_gen']
            
            # 总负荷（原始负荷 - 削减）
            total_load = sum(model.P_load) * 1000 - results['total_P_shed']
            
            # 总损耗
            total_loss = results['losses']
            
            # 功率平衡（允许一定误差）
            balance_error = abs(total_gen - total_load - total_loss)
            assert balance_error < 50  # 允许50kW的误差
            
    def test_generator_limits(self, model):
        """测试发电机出力约束"""
        results = model.solve(verbose=False)
        
        if results and results['P_gen']:
            for node, p_gen in results['P_gen'].items():
                # 检查不超过最大出力
                assert p_gen <= model.P_gen_max[node-1] * 1000 + 1e-3
                
    def test_different_solvers(self, model):
        """测试不同求解器"""
        # 测试SCS求解器
        results_scs = model.solve(solver=cp.SCS, verbose=False)
        assert results_scs is not None
        
        # 如果安装了ECOS，也测试它
        try:
            import ecos
            results_ecos = model.solve(solver=cp.ECOS, verbose=False)
            assert results_ecos is not None
            
            # 比较两个求解器的结果（应该相近）
            obj_diff = abs(results_scs['objective'] - results_ecos['objective'])
            assert obj_diff < 10  # 允许10元的差异
        except ImportError:
            pass
            
    def test_write_lp(self, model, tmp_path):
        """测试LP文件输出"""
        lp_file = tmp_path / "test.lp"
        
        # 先求解
        results = model.solve(verbose=False)
        
        # 写入LP描述
        model.write_lp(str(lp_file))
        
        # 检查文件是否生成
        assert lp_file.exists()
        
        # 检查文件内容
        content = lp_file.read_text()
        assert "Post-Disaster Static Dispatch" in content
        assert "Variables:" in content
        assert "Constraints:" in content