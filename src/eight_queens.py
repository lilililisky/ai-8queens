class EightQueensSolver:
    """
    八皇后问题求解器
    使用回溯法解决N皇后问题
    """
    
    def solve_n_queens(self, n):
        """
        求解N皇后问题的所有解
        
        Args:
            n (int): 棋盘大小和皇后数量
            
        Returns:
            list: 所有解的列表，每个解是皇后位置的列表，其中索引表示行，值表示列
        """
        def is_safe(board, row, col):
            """
            检查在指定位置放置皇后是否安全
            
            Args:
                board (list): 当前棋盘状态
                row (int): 要放置皇后的行
                col (int): 要放置皇后的列
                
            Returns:
                bool: 如果安全返回True，否则返回False
            """
            # 检查同一列是否有皇后
            for i in range(row):
                if board[i] == col:
                    return False
            
            # 检查左上到右下的对角线
            for i, j in zip(range(row-1, -1, -1), range(col-1, -1, -1)):
                if board[i] == j:
                    return False
            
# 检查右上到左下的对角线
            for i, j in zip(range(row-1, -1, -1), range(col+1, n)):
                if board[i] == j:  # 正确的判断条件
                    return False
            
            return True
        
        def backtrack(row, current_solution):
            """
            回溯函数，递归求解N皇后问题
            
            Args:
                row (int): 当前处理的行
                current_solution (list): 当前解决方案
            """
            # 如果已经放置了n个皇后，找到一个解
            if row == n:
                solutions.append(current_solution[:])
                return
            
            # 尝试在当前行的每一列放置皇后
            for col in range(n):
                if is_safe(current_solution, row, col):
                    # 放置皇后
                    current_solution.append(col)
                    # 递归处理下一行
                    backtrack(row + 1, current_solution)
                    # 回溯，移除当前皇后
                    current_solution.pop()
        
        solutions = []
        backtrack(0, [])
        return solutions
    
    def print_solution(self, solution):
        """
        打印棋盘
        
        Args:
            solution (list): 皇后位置的列表，其中索引表示行，值表示列
        """
        n = len(solution)
        for row in range(n):
            line = []
            for col in range(n):
                if solution[row] == col:
                    line.append('Q')
                else:
                    line.append('.')
            print(' '.join(line))
        print()


# 测试代码
if __name__ == "__main__":
    solver = EightQueensSolver()
    # 求解8皇后问题
    solutions = solver.solve_n_queens(8)
    print(f"共找到 {len(solutions)} 个解")
    
    # 打印前3个解
    print("前3个解:")
    for i, solution in enumerate(solutions[:3]):
        print(f"解 {i+1}:")
        solver.print_solution(solution)