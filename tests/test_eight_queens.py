import unittest
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.eight_queens import EightQueensSolver


def is_valid_solution(solution):
    """
    验证皇后问题的解是否有效
    
    Args:
        solution (list): 皇后位置的列表，其中索引表示行，值表示列
        
    Returns:
        bool: 如果解有效返回True，否则返回False
    """
    n = len(solution)
    
    # 检查每一行是否只有一个皇后（通过列表长度和索引自动保证）
    if len(solution) != n:
        return False
    
    # 检查同一列是否有多个皇后
    if len(set(solution)) != n:
        return False
    
    # 检查对角线
    for i in range(n):
        for j in range(i + 1, n):
            # 检查是否在同一对角线上
            if abs(solution[i] - solution[j]) == abs(i - j):
                return False
    
    return True


class TestEightQueensSolver(unittest.TestCase):
    """
    测试EightQueensSolver类的测试类
    """
    
    def setUp(self):
        """
        测试方法的设置，创建一个EightQueensSolver实例
        """
        self.solver = EightQueensSolver()
    
    def test_n_4(self):
        """
        测试N=4的情况，应该有2个解
        """
        solutions = self.solver.solve_n_queens(4)
        self.assertEqual(len(solutions), 2, f"4皇后问题应该有2个解，实际找到{len(solutions)}个")
        
        # 验证每个解是否有效
        for solution in solutions:
            self.assertTrue(is_valid_solution(solution), f"无效的解: {solution}")
    
    def test_n_8(self):
        """
        测试N=8的情况，应该有92个解
        """
        solutions = self.solver.solve_n_queens(8)
        self.assertEqual(len(solutions), 92, f"8皇后问题应该有92个解，实际找到{len(solutions)}个")
        
        # 验证每个解是否有效
        for solution in solutions:
            self.assertTrue(is_valid_solution(solution), f"无效的解: {solution}")
    
    def test_print_solution(self):
        """
        测试print_solution方法是否能正常执行
        """
        # 测试一个简单的解
        test_solution = [0, 4, 7, 5, 2, 6, 1, 3]  # 8皇后的一个解
        try:
            self.solver.print_solution(test_solution)
            # 如果执行到这里，说明print_solution方法没有抛出异常
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"print_solution方法执行失败: {e}")


if __name__ == "__main__":
    # 运行测试
    unittest.main()
