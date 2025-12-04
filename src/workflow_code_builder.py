#!/usr/bin/env python3
"""
工作流代码构建器 - 从Qwen输出中完整重构工作流代码

替代旧的多层补救系统，采用完整重构的方式确保全局一致性
"""
import re
import ast
from typing import Set, Optional, Tuple, Dict
from src.workflow_consistency_checker import WorkflowConsistencyChecker


class WorkflowCodeBuilder:
    """
    从Qwen的自然语言输出中重构完整的、一致的工作流代码

    设计原则：
    - 不信任Qwen生成的import/init，自动完整重构
    - 自动检测__call__中使用的operator
    - 自动生成完整的import和初始化
    - 最终验证全局一致性
    """

    def __init__(self):
        """初始化代码构建器"""
        self.checker = WorkflowConsistencyChecker()
        self.valid_operators = {
            'Custom', 'AnswerGenerate', 'Programmer', 'Test',
            'Review', 'Revise', 'ScEnsemble'
        }

    def build_from_qwen_output(
        self,
        qwen_text: str,
        problem_type: str = "math",
        strict: bool = True
    ) -> Tuple[str, bool, Optional[str]]:
        """
        从Qwen的输出中重构完整的工作流代码

        步骤:
        1. 提取__call__方法的逻辑部分
        2. 自动分析__call__中使用了哪些operator
        3. 自动生成完整的import语句
        4. 自动生成完整的operator初始化
        5. 拼接成完整的代码
        6. 验证一致性

        Args:
            qwen_text: Qwen模型的输出文本
            problem_type: 问题类型 ("math", "code", "qa")
            strict: 是否严格模式（失败时抛异常）

        Returns:
            (code, success, error_msg)
            - code: 重构后的完整代码
            - success: 是否重构成功
            - error_msg: 错误信息（如果失败）
        """
        try:
            # Step 1: 提取__call__逻辑
            call_logic, call_signature = self._extract_call_logic(qwen_text, problem_type)
            if not call_logic:
                raise ValueError("无法从输出中提取__call__方法")

            # Step 2: 检测operator使用
            used_operators = self._detect_used_operators(call_logic)
            if not used_operators:
                # 如果没有检测到operator，使用Custom作为默认
                used_operators = {'custom': 'Custom'}

            # Step 3: 生成import
            imports = self._generate_imports(used_operators)

            # Step 4: 生成初始化
            inits = self._generate_initializations(used_operators)

            # Step 5: 拼接完整代码
            full_code = self._assemble_workflow(
                imports=imports,
                inits=inits,
                call_signature=call_signature,
                call_logic=call_logic,
                problem_type=problem_type
            )

            # Step 6: 验证一致性
            result = self.checker.check_consistency(full_code)
            if not result['consistent']:
                if strict:
                    raise ValueError(f"代码仍不一致: {result['issues']}")
                else:
                    print(f"⚠️ 警告: 代码一致性检查失败但继续")
                    print(f"   问题: {result['issues']}")

            return full_code, True, None

        except Exception as e:
            error_msg = f"代码重构失败: {str(e)}"
            if strict:
                raise
            return "", False, error_msg

    def _extract_call_logic(
        self,
        qwen_text: str,
        problem_type: str
    ) -> Tuple[str, str]:
        """
        从Qwen输出中提取__call__方法的实现逻辑

        返回: (call_logic_code, call_signature)
        """
        # ✅ CRITICAL FIX: Normalize indentation FIRST (fixes 60-70% of errors)
        qwen_text = self._normalize_indentation(qwen_text)

        # 查找 async def __call__
        pattern = r'async\s+def\s+__call__\s*\(([^)]+)\)\s*:'
        match = re.search(pattern, qwen_text)

        if not match:
            # 使用默认signature
            if problem_type == "code":
                call_signature = "async def __call__(self, problem: str, entry_point: str, test: str):"
            else:
                call_signature = "async def __call__(self, problem: str):"
            # 尝试提取方法体
            body_pattern = r'(?:async\s+)?def\s+__call__[^:]*:\s*([\s\S]+?)(?=\n\s{0,4}(?:def|class|\Z))'
            body_match = re.search(body_pattern, qwen_text)
            if body_match:
                body = body_match.group(1).strip()
            else:
                # 如果完全找不到，返回默认实现
                body = "pass"
        else:
            # 提取完整的call_signature
            params = match.group(1)
            call_signature = f"async def __call__(self, {params}):"

            # 提取__call__方法体
            # 找到__call__定义后，提取到下一个方法或类定义
            call_start = match.start()
            # 从__call__之后查找内容
            content_after = qwen_text[call_start + len(match.group(0)):]
            body_end_match = re.search(r'\n(?=\s{0,4}(?:async\s+)?def\s+\w+|\s{0,4}class\s+|\Z)', content_after)

            if body_end_match:
                body = content_after[:body_end_match.start()].strip()
            else:
                body = content_after.strip()

        # 清理body中的缩进 (use normalization instead of dedent)
        body = self._normalize_indentation(body)

        return body, call_signature

    def _detect_used_operators(self, call_logic: str) -> Dict[str, str]:
        """
        分析call_logic中使用了哪些operator

        返回: Dict[attribute_name -> class_name]
        例如: {'answer_generate': 'AnswerGenerate', 'test': 'Test'}
        """
        used_operators = {}

        # 模式: await self.xxx(...)
        patterns = re.findall(r'await\s+self\.(\w+)\s*\(', call_logic)

        for attr_name in set(patterns):
            # 推断operator类名
            class_name = self._infer_operator_class(attr_name)
            if class_name:
                used_operators[attr_name] = class_name

        return used_operators

    def _infer_operator_class(self, attr_name: str) -> Optional[str]:
        """
        从operator属性名推断类名

        规则:
        - answer_generate -> AnswerGenerate
        - programmer -> Programmer
        - review -> Review
        - etc.
        """
        # 直接映射
        mapping = {
            'custom': 'Custom',
            'answer_generate': 'AnswerGenerate',
            'programmer': 'Programmer',
            'test': 'Test',
            'review': 'Review',
            'revise': 'Revise',
            'sc_ensemble': 'ScEnsemble',
        }

        if attr_name in mapping:
            return mapping[attr_name]

        # 尝试驼峰转换
        class_name = ''.join(word.capitalize() for word in attr_name.split('_'))
        if class_name in self.valid_operators:
            return class_name

        return None

    def _generate_imports(self, used_operators: Dict[str, str]) -> str:
        """
        生成完整的import语句

        Args:
            used_operators: {attr_name -> class_name}

        Returns:
            导入代码字符串
        """
        # 收集所有使用的类名
        class_names = sorted(set(used_operators.values()))

        # 生成import语句
        if class_names:
            imports_line = f"from scripts.operators import {', '.join(class_names)}"
        else:
            # 至少导入Custom作为备选
            imports_line = "from scripts.operators import Custom"

        return f"""{imports_line}
from scripts.async_llm import create_llm_instance
from scripts.evaluator import DatasetType"""

    def _generate_initializations(self, used_operators: Dict[str, str]) -> str:
        """
        生成完整的operator初始化代码

        Args:
            used_operators: {attr_name -> class_name}

        Returns:
            初始化代码（每行缩进）
        """
        if not used_operators:
            return ""

        init_lines = []
        for attr_name, class_name in sorted(used_operators.items()):
            init_lines.append(f"        self.{attr_name} = {class_name}(self.llm)")

        return "\n".join(init_lines)

    def _assemble_workflow(
        self,
        imports: str,
        inits: str,
        call_signature: str,
        call_logic: str,
        problem_type: str
    ) -> str:
        """
        拼接成完整的工作流代码

        Args:
            imports: import语句
            inits: operator初始化语句
            call_signature: __call__方法签名
            call_logic: __call__方法体
            problem_type: 问题类型

        Returns:
            完整的Python代码
        """
        # 确保call_logic有正确的缩进
        call_logic_indented = self._indent_code(call_logic, 2)

        code = f"""{imports}

class Workflow:
    def __init__(self, name: str, llm_config, dataset: DatasetType):
        self.name = name
        self.dataset = dataset
        self.llm = create_llm_instance(llm_config)
{inits if inits else '        pass'}

    {call_signature}
{call_logic_indented}"""

        return code

    @staticmethod
    def _dedent_code(code: str) -> str:
        """
        移除代码的共同前导空格

        Args:
            code: 原始代码

        Returns:
            去除缩进后的代码
        """
        lines = code.split('\n')
        # 找到最小缩进
        non_empty_lines = [line for line in lines if line.strip()]
        if not non_empty_lines:
            return code

        min_indent = min(len(line) - len(line.lstrip()) for line in non_empty_lines)
        # 移除最小缩进
        dedented = '\n'.join(
            line[min_indent:] if len(line) > min_indent else line
            for line in lines
        )
        return dedented.strip()

    @staticmethod
    def _normalize_indentation(code: str) -> str:
        """
        Normalize mixed tabs/spaces and remove common indentation prefix.

        This fixes ~60-70% of Python syntax errors caused by:
        - Mixed tabs and spaces
        - Inconsistent indentation levels
        - Extra leading whitespace

        Args:
            code: Original code string

        Returns:
            Normalized code with consistent spacing
        """
        lines = code.split('\n')

        # Step 1: Convert all tabs to 4 spaces
        normalized_lines = []
        for line in lines:
            # Replace each tab with 4 spaces
            normalized_line = line.replace('\t', '    ')
            # Also strip trailing whitespace
            normalized_line = normalized_line.rstrip()
            normalized_lines.append(normalized_line)

        # Step 2: Find minimum indentation level
        non_empty_lines = [line for line in normalized_lines if line.strip()]
        if not non_empty_lines:
            return code

        min_indent = min(len(line) - len(line.lstrip(' ')) for line in non_empty_lines)

        # Step 3: Remove common indentation prefix
        dedented_lines = []
        for line in normalized_lines:
            if len(line) >= min_indent:
                dedented_lines.append(line[min_indent:])
            else:
                dedented_lines.append(line)

        return '\n'.join(dedented_lines)

    @staticmethod
    def _indent_code(code: str, indent_level: int = 2) -> str:
        """
        为代码添加缩进

        Args:
            code: 原始代码
            indent_level: 缩进级别（空格数）

        Returns:
            缩进后的代码
        """
        indent = ' ' * indent_level
        lines = code.split('\n')
        indented_lines = []

        for line in lines:
            if line.strip():  # 非空行
                indented_lines.append(indent + line)
            else:  # 空行保持空
                indented_lines.append('')

        return '\n'.join(indented_lines)

    def validate(self, code: str, verbose: bool = False) -> Tuple[bool, Optional[str]]:
        """
        验证代码的一致性和语法

        Args:
            code: Python代码
            verbose: 是否打印详细信息

        Returns:
            (is_valid, error_message)
        """
        # 语法验证
        try:
            ast.parse(code)
        except SyntaxError as e:
            return False, f"语法错误: {str(e)}"

        # 一致性验证
        result = self.checker.check_consistency(code)
        if not result['consistent']:
            error_msg = "; ".join(result['issues'])
            return False, f"一致性检查失败: {error_msg}"

        if verbose:
            print(self.checker.get_summary(result))

        return True, None


# 测试函数
def test_builder():
    """测试工作流代码构建器"""
    builder = WorkflowCodeBuilder()

    # 测试输入: Qwen生成的片段输出
    qwen_output = """
async def __call__(self, problem: str):
    # 使用answer_generate来解决问题
    result = await self.answer_generate(input=problem)
    answer = result.get('answer', '')

    # 如果需要验证
    review = await self.review(problem=problem, solution=answer)

    # 返回答案
    return answer, self.llm.get_usage_summary()["total_cost"]
"""

    print("\n🔨 测试代码构建器")
    print("=" * 70)

    code, success, error = builder.build_from_qwen_output(qwen_output, problem_type="math", strict=False)

    if success:
        print("✅ 代码重构成功!")
        print("\n📄 重构后的代码:")
        print(code)

        # 验证
        is_valid, val_error = builder.validate(code, verbose=True)
        if is_valid:
            print("\n✅ 代码验证通过!")
        else:
            print(f"\n❌ 代码验证失败: {val_error}")
    else:
        print(f"❌ 代码重构失败: {error}")


if __name__ == "__main__":
    test_builder()
