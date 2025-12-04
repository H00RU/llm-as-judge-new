#!/usr/bin/env python3
"""
答案提取器 V2 - 从模型输出和ground truth中提取标准化答案
增强功能：
1. 代码泄漏检测（在\boxed{}中检测代码片段）
2. 工作流日志清理（移除"Revised Solution"等中间输出）
3. 6级fallback链用于数学答案提取
4. 增强的代码块提取
5. 支持多种答案格式和选项题标准化
"""
import re
import json
from typing import Any, Optional, Tuple


class AnswerExtractor:
    """增强的答案提取器，用于标准化预测和真值"""

    def __init__(self, use_llm_fallback: bool = True, llm_client=None):
        """
        Args:
            use_llm_fallback: 是否使用LLM作为兜底提取器
            llm_client: LLM客户端（用于兜底提取）
        """
        self.use_llm_fallback = use_llm_fallback
        self.llm_client = llm_client

    def extract_answer(self, text: str, problem_type: str, is_ground_truth: bool = False) -> str:
        """
        主入口：从文本中提取标准化答案

        Args:
            text: 原始文本
            problem_type: 问题类型 (math/code/qa)
            is_ground_truth: 是否是ground truth（影响提取策略）

        Returns:
            标准化后的答案
        """
        if not text:
            return ""

        if problem_type == "math":
            return self._extract_math_answer(text, is_ground_truth)
        elif problem_type == "code":
            return self._extract_code_answer(text, is_ground_truth)
        elif problem_type == "qa":
            return self._extract_qa_answer(text, is_ground_truth)
        else:
            return str(text).strip()

    def _clean_workflow_logs(self, text: str) -> str:
        """清理工作流日志污染（移除中间输出）"""
        # 移除 "Revised Solution:" 及其后的内容，直到遇到\boxed或数字
        text = re.sub(r'Revised Solution:.*?(?=\\boxed|\d|$)', '', text, flags=re.DOTALL)
        # 移除 "Based on the feedback" 污染
        text = re.sub(r'Based on the feedback[^\\]*(?=\\boxed|$)', '', text, flags=re.DOTALL)
        return text

    def _detect_code_leak_in_boxed(self, boxed_content: str) -> bool:
        """检测\boxed{}中是否包含代码泄漏

        返回True如果检测到代码关键字
        """
        code_keywords = ['def ', 'return ', 'import ', 'class ', 'if __name__',
                        'print(', 'for ', 'while ', 'elif ', ':\n', 'await ', 'async ']
        return any(keyword in boxed_content for keyword in code_keywords)

    def _extract_math_answer(self, text: str, is_ground_truth: bool) -> str:
        """
        提取数学答案 - 6级fallback链

        级别1: <answer>标签（取最后一个）
        级别2: \boxed{}（LaTeX格式，含代码泄漏检测）
        级别3: #### 格式（GSM8K）
        级别4: Final Answer 标记模式
        级别5: 代数表达式（含变量）
        级别6: 提取数字（兜底）
        """
        text = str(text).strip()

        # ============ 级别1: <answer>标签 ============
        answer_text = self._try_answer_tags(text)
        if answer_text:
            return answer_text

        # 清理工作流日志
        text = self._clean_workflow_logs(text)

        # ============ 级别2: \boxed{} ============
        answer_text = self._try_boxed_notation(text)
        if answer_text:
            return answer_text

        # ============ 级别3: #### 格式（GSM8K） ============
        if is_ground_truth:
            answer_text = self._try_gsm8k_format(text)
            if answer_text:
                return answer_text

        # ============ 级别4: Final Answer 标记 ============
        answer_text = self._try_final_answer_marker(text)
        if answer_text:
            return answer_text

        # ============ 级别5: 代数表达式 ============
        answer_text = self._try_algebraic_expression(text)
        if answer_text:
            return answer_text

        # ============ 级别6: 提取数字（兜底） ============
        answer_text = self._try_extract_numbers(text, is_ground_truth)
        if answer_text:
            return answer_text

        # ============ 终极兜底: LLM ============
        if is_ground_truth and self.use_llm_fallback and self.llm_client:
            # 检测复杂性：多个数字和运算符
            has_calculations = text.count('=') >= 2 or len(re.findall(r'\d+', text)) > 3
            if has_calculations:
                llm_result = self._llm_extract_math_ground_truth(text)
                if llm_result and llm_result != text:
                    return llm_result

        # 最后兜底：检查是否有污染内容
        if 'Based on the feedback' in text or 'Revised Solution' in text or '```python' in text:
            return ""

        cleaned = re.sub(r'[^\d\-+./]', ' ', text).strip()
        if cleaned:
            nums = re.findall(r'-?\d+\.?\d*', cleaned)
            if nums:
                return nums[-1]

        return ""

    def _try_answer_tags(self, text: str) -> Optional[str]:
        """级别1: 尝试提取<answer>标签"""
        answer_matches = re.findall(r'<answer>(.*?)</answer>', text, re.DOTALL)
        if answer_matches:
            answer_text = answer_matches[-1].strip()
            return self._clean_math_answer(answer_text)
        return None

    def _try_boxed_notation(self, text: str) -> Optional[str]:
        """级别2: 尝试提取\boxed{}，含代码泄漏检测"""
        boxed = self._extract_boxed(text)
        if not boxed:
            return None

        # 检测空输出
        if not boxed.strip():
            return None

        # 🔧 P0-FIX: 检测代码泄漏
        if self._detect_code_leak_in_boxed(boxed):
            # 尝试执行代码获取答案
            executed_answer = self._execute_code_and_extract_answer(boxed, 'math')
            if executed_answer:
                return executed_answer

            # 静态分析提取答案
            code_answer = self._extract_answer_from_code_block(boxed)
            if code_answer and not any(kw in str(code_answer) for kw in ['def ', 'import ', 'class ']):
                return self._clean_math_answer(code_answer)

            # 无法提取有效答案
            return None

        # 检测代码块标记
        if '```python' in boxed or boxed.startswith('```'):
            executed_answer = self._execute_code_and_extract_answer(boxed, 'math')
            if executed_answer:
                return executed_answer

            code_answer = self._extract_answer_from_code_block(boxed)
            if code_answer:
                return code_answer

            return None

        # 检测执行错误或污染内容
        if (boxed.startswith('Error:') or 'Traceback' in boxed or 'SyntaxError' in boxed or
            'Based on the feedback' in boxed or 'Revised Solution' in boxed):
            return None

        return self._clean_math_answer(boxed)

    def _try_gsm8k_format(self, text: str) -> Optional[str]:
        """级别3: 尝试GSM8K的####格式"""
        gsm8k_match = re.search(r'####\s*(-?\d+\.?\d*)', text)
        if gsm8k_match:
            return self._clean_math_answer(gsm8k_match.group(1))
        return None

    def _try_final_answer_marker(self, text: str) -> Optional[str]:
        """级别4: 尝试Final Answer标记"""
        final_answer_patterns = [
            r"(?:the\s+final\s+answer\s+is)[：:]*\s*([-+]?\d+(?:/\d+)?(?:\.\d+)?)",
            r"(?:Final\s+Answer|最终答案)[：:]*\s*([-+]?\d+(?:/\d+)?(?:\.\d+)?)",
            r"(?:The\s+answer\s+is)[：:]*\s*([-+]?\d+(?:/\d+)?(?:\.\d+)?)",
        ]
        for pattern in final_answer_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return self._clean_math_answer(match.group(1))
        return None

    def _try_algebraic_expression(self, text: str) -> Optional[str]:
        """级别5: 尝试代数表达式（含变量）"""
        has_variables = bool(re.search(r'[a-zA-Z]', text))
        has_operators = bool(re.search(r'[+\-*/\^]', text))
        if has_variables and has_operators:
            cleaned = re.sub(r'\s+', '', text).strip()
            return cleaned
        return None

    def _try_extract_numbers(self, text: str, is_ground_truth: bool) -> Optional[str]:
        """级别6: 兜底策略 - 提取数字"""
        if is_ground_truth:
            numbers = self._extract_all_numbers(text)
            if numbers:
                return str(numbers[-1])
        else:
            # Prediction: 优先括号外的数字
            clean_text = re.sub(r'\([^)]*\)', '', text)
            clean_numbers = self._extract_all_numbers(clean_text)
            if clean_numbers:
                return str(clean_numbers[-1])
            numbers = self._extract_all_numbers(text)
            if numbers:
                return str(numbers[-1])
        return None

    def _extract_code_answer(self, text: str, is_ground_truth: bool) -> str:
        """
        提取代码答案

        对于Code任务:
        - prediction: 提取完整的函数实现代码
        - ground_truth: 同样提取函数实现代码
        - 评估: 通过test_result metadata而非字符串比较

        优先级：
        1. ```python...``` 代码块（带AST验证）
        2. def 函数定义
        3. 完整文本（如果是ground truth）
        """
        text = str(text).strip()

        # P0修复: 清理空代码块和无效占位符
        text = re.sub(r'```python\s*```', '', text)
        text = re.sub(r'```\s*```', '', text)
        text = text.replace('No code provided', '').replace('No code', '')

        # 1. 提取代码块（P1修复: 换行符可选）
        code_blocks = re.findall(r'```(?:python)?\s*\n?([^`]+)```', text)
        if code_blocks:
            for block in reversed(code_blocks):
                block = block.strip()
                if self._validate_code_syntax(block):
                    return block
            return code_blocks[-1].strip()

        # 2. 查找函数定义
        func_pattern = r'(def\s+\w+\s*\([^)]*\)[^:]*:[\s\S]+?)(?=\n(?:def\s|class\s|$))'
        funcs = re.findall(func_pattern, text)
        if funcs:
            first_func = funcs[0].strip()
            if self._validate_code_syntax(first_func):
                return first_func
            return first_func

        # 3. 如果是ground truth且看起来像代码，直接返回
        if is_ground_truth:
            return text

        # 4. LLM兜底
        if self.use_llm_fallback and self.llm_client:
            return self._llm_extract_code(text)

        return text

    def _validate_code_syntax(self, code: str) -> bool:
        """验证代码语法正确性"""
        try:
            import ast
            ast.parse(code)
            return True
        except SyntaxError:
            return False
        except Exception:
            return False

    def _extract_qa_answer(self, text: str, is_ground_truth: bool) -> str:
        """
        提取QA答案
        - 对于数值型问题: 提取最终数字答案
        - 对于文本型问题: 标准化文本
        - 对于选项题: 统一格式为单字母（A/B/C/D/E）
        """
        text = str(text).strip()

        # 0. 选项题标准化（优先处理）
        option_answer = self._normalize_option_answer(text)
        if option_answer:
            return option_answer

        # 1. 如果有明确的答案标记，先尝试提取
        answer_patterns = [
            r"(?:Answer|答案)[：:]*\s*([^\n.]+)",
            r"(?:The answer is)[：:]*\s*([^\n.]+)",
            r"(?:Final answer|Therefore)[：:]*\s*([^\n.]+)",
        ]

        for pattern in answer_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                answer_text = match.group(1).strip()
                option_normalized = self._normalize_option_answer(answer_text)
                if option_normalized:
                    return option_normalized
                numbers = self._extract_all_numbers(answer_text)
                if numbers:
                    return str(int(numbers[-1]) if numbers[-1] == int(numbers[-1]) else numbers[-1])
                return self._normalize_qa_answer(answer_text)

        # 2. 检查是否为数值型答案
        has_calculation = any(op in text for op in ['+', '-', '*', '/', '=', '<<', '>>'])
        if has_calculation or re.search(r'\d+', text):
            numbers = self._extract_all_numbers(text)
            if numbers:
                final_number = numbers[-1]
                return str(int(final_number) if final_number == int(final_number) else final_number)

        # 3. 文本型答案 - 标准化
        normalized = self._normalize_qa_answer(text)

        # 4. 如果太长，尝试提取核心信息
        if len(normalized.split()) > 50 and not is_ground_truth:
            sentences = text.split('.')
            if len(sentences) > 2:
                key_text = sentences[-2] + '.' + sentences[-1]
                return self._normalize_qa_answer(key_text)

        return normalized

    def _normalize_option_answer(self, text: str) -> Optional[str]:
        """标准化选项答案为单字母格式

        支持的格式：
        - "A" → "A"
        - "A." → "A"
        - "A. ream" → "A"
        - "Option A" → "A"
        - "(A)" → "A"
        """
        text = text.strip()

        # 格式1: 单个大写字母
        if len(text) == 1 and text.upper() in 'ABCDE':
            return text.upper()

        # 格式2: "A." 或 "A:" 或 "(A)"
        match = re.match(r'^[\(\[]?([A-E])[\)\]\.:]*\s*', text, re.IGNORECASE)
        if match:
            return match.group(1).upper()

        # 格式3: "Option A" 或 "选项A"
        match = re.search(r'(?:Option|选项)\s*([A-E])\b', text, re.IGNORECASE)
        if match:
            return match.group(1).upper()

        # 格式4: "The answer is A"
        match = re.search(r'\b([A-E])\b(?=\s*(?:is|为)\s*(?:correct|the answer)?)', text, re.IGNORECASE)
        if match:
            return match.group(1).upper()

        return None

    def _execute_code_and_extract_answer(self, code_block: str, problem_type: str) -> Optional[str]:
        """执行代码并提取答案（用于数学问题）"""
        if problem_type != "math":
            return None

        import subprocess
        import tempfile
        import os

        code = re.sub(r'^```python\n?', '', code_block)
        code = re.sub(r'```$', '', code)
        code = code.strip()

        # 安全检查
        dangerous_keywords = ['os.system', 'subprocess', 'eval', 'exec', 'open', '__import__', 'rm ', 'del ']
        if any(kw in code for kw in dangerous_keywords):
            return None

        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                if 'print(' not in code:
                    lines = code.split('\n')
                    last_var = None
                    for line in reversed(lines):
                        line = line.strip()
                        if '=' in line and not line.startswith('#'):
                            var_name = line.split('=')[0].strip()
                            if var_name.isidentifier():
                                last_var = var_name
                                break

                    if last_var:
                        code += f'\nprint({last_var})'

                f.write(code)
                temp_path = f.name

            result = subprocess.run(
                ['python3', temp_path],
                capture_output=True,
                text=True,
                timeout=5
            )

            os.unlink(temp_path)

            if result.returncode == 0 and result.stdout:
                output = result.stdout.strip()
                if output:
                    last_line = output.split('\n')[-1].strip()
                    try:
                        if '/' in last_line:
                            parts = last_line.split('/')
                            float(parts[0])
                            float(parts[1])
                            return last_line
                        else:
                            num = float(last_line)
                            return str(int(num) if num == int(num) else num)
                    except:
                        return last_line

            return None

        except subprocess.TimeoutExpired:
            try:
                os.unlink(temp_path)
            except:
                pass
            return None
        except Exception:
            try:
                os.unlink(temp_path)
            except:
                pass
            return None

    def _extract_answer_from_code_block(self, code_block: str) -> Optional[str]:
        """从代码块中提取答案（静态分析）"""
        code_block = code_block.strip()

        code_block = re.sub(r'^```python\n?', '', code_block)
        code_block = re.sub(r'```$', '', code_block)

        # 策略1: 查找print语句
        print_pattern = r'print\(([^)]+)\)'
        print_matches = re.findall(print_pattern, code_block)
        if print_matches:
            last_print = print_matches[-1].strip()
            if last_print.isidentifier():
                var_pattern = rf'{last_print}\s*=\s*(.+)'
                var_match = re.search(var_pattern, code_block)
                if var_match:
                    return var_match.group(1).strip()
            return last_print

        # 策略2: 查找return语句
        return_pattern = r'return\s+(.+?)\s*(?:\n|$)'
        return_matches = re.findall(return_pattern, code_block)
        if return_matches:
            return return_matches[-1].strip()

        # 策略3: 查找最后的赋值语句
        assignment_lines = [line for line in code_block.split('\n') if '=' in line and not line.strip().startswith('#')]
        if assignment_lines:
            last_assignment = assignment_lines[-1]
            if '=' in last_assignment:
                value = last_assignment.split('=', 1)[1].strip()
                return value

        return None

    def _extract_boxed(self, text: str) -> Optional[str]:
        """提取\boxed{}中的内容"""
        pattern = r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
        return None

    def _extract_all_numbers(self, text: str) -> list:
        """提取所有数字（支持整数、小数、分数、负数）"""
        numbers = []

        # 优先匹配分数
        fraction_pattern = r'-?\d+/\d+'
        fraction_matches = re.findall(fraction_pattern, text)
        for frac in fraction_matches:
            numbers.append(frac)

        # 匹配其他数字格式
        other_patterns = [
            r'-?\d+\.?\d*(?:[eE][+-]?\d+)?',  # 科学计数法
            r'-?\d{1,3}(?:,\d{3})*(?:\.\d+)?',  # 带千分位
        ]

        for pattern in other_patterns:
            matches = re.findall(pattern, text)
            for m in matches:
                if any(m in frac for frac in fraction_matches):
                    continue
                try:
                    clean_m = m.replace(',', '')
                    numbers.append(clean_m)
                except:
                    pass

        return numbers

    def _clean_math_answer(self, answer: str) -> str:
        """清理数学答案（去单位、标准化格式）"""
        answer = str(answer).strip()

        # 修复 "i42" 问题
        if answer.startswith('i') and len(answer) > 1 and answer[1:].replace('.', '', 1).replace('/', '').isdigit():
            answer = answer[1:]

        # 移除LaTeX命令
        answer = re.sub(r'\\boxed\{(.+?)\}', r'\1', answer)
        answer = re.sub(r'\\frac\{(.+?)\}\{(.+?)\}', r'\1/\2', answer)
        answer = re.sub(r'\\text\{(.+?)\}', r'\1', answer)

        # 移除常见单位
        units = ['grams', 'gram', 'g', 'kg', 'meters', 'meter', 'm', 'cm',
                 'seconds', 'second', 's', 'minutes', 'minute', 'min',
                 'dollars', 'dollar', '$', '元', '个', '只', 'km', 'hours', 'hour']

        for unit in units:
            answer = re.sub(rf'\s*{re.escape(unit)}\b', '', answer, flags=re.IGNORECASE)

        # 移除多余的标点和空格
        answer = re.sub(r'[,\s]+', '', answer)

        # 尝试规范化数字
        try:
            if '/' in answer:
                parts = answer.split('/')
                if len(parts) == 2:
                    try:
                        numerator = float(parts[0])
                        denominator = float(parts[1])

                        if denominator == 1:
                            return str(int(numerator) if numerator == int(numerator) else numerator)

                        from math import gcd
                        if numerator == int(numerator) and denominator == int(denominator):
                            g = gcd(int(abs(numerator)), int(abs(denominator)))
                            if g > 1:
                                numerator /= g
                                denominator /= g
                            return f"{int(numerator)}/{int(denominator)}"

                        return answer
                    except:
                        return answer

            # 处理百分号
            if '%' in answer:
                return str(float(answer.replace('%', '')) / 100)

            # 普通数字
            num = float(answer)
            if num == int(num):
                return str(int(num))
            return str(num)
        except:
            return answer

    def _normalize_qa_answer(self, text: str) -> str:
        """标准化QA答案"""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = ' '.join(text.split())
        return text.strip()

    def _llm_extract_math(self, text: str) -> str:
        """使用LLM提取数学答案"""
        if not self.llm_client:
            return text

        prompt = f"""Extract ONLY the final numerical answer from this math solution.
Return JUST the number, no explanation.

Solution: {text[:1000]}

Final answer (number only):"""

        try:
            response = self.llm_client.generate(prompt, max_tokens=20, temperature=0)
            answer = response.strip()
            float(answer.replace('/', '.').replace(',', ''))
            return answer
        except:
            return text

    def _llm_extract_math_ground_truth(self, text: str) -> str:
        """使用LLM理解ground truth中的最终答案"""
        if not self.llm_client:
            return text

        prompt = f"""You are extracting the FINAL ANSWER from a mathematical solution text.

**Instructions:**
1. **Ignore intermediate calculations** - Focus only on the concluding answer
2. **Look for concluding statements** like "So the answer is...", "Therefore...", "The result is..."
3. **Extract the final numeric value** - Return JUST the number

**Text:**
{text[:800]}

**Output Format:**
- Return ONLY the final numerical answer
- No explanation, no intermediate values
- If multiple numbers exist, return the one from the final conclusion

**Final Answer (number only):"""

        try:
            response = self.llm_client.generate(prompt, max_tokens=30, temperature=0)
            answer = response.strip()
            if '/' in answer:
                parts = answer.split('/')
                float(parts[0])
                float(parts[1])
            else:
                float(answer.replace(',', ''))
            return answer
        except:
            return text

    def _llm_extract_code(self, text: str) -> str:
        """使用LLM提取代码"""
        if not self.llm_client:
            return text

        prompt = f"""Extract ONLY the Python function code from this text.
Return JUST the code, no explanation.

Text: {text[:1000]}

Code:"""

        try:
            response = self.llm_client.generate(prompt, max_tokens=500, temperature=0)
            if 'def ' in response:
                return response.strip()
            return text
        except:
            return text


def test_extractor():
    """测试答案提取器"""
    extractor = AnswerExtractor(use_llm_fallback=False)

    test_cases = [
        {
            "text": "The probability is $\\frac{1}{27}$. So the answer is \\boxed{\\frac{8}{9}}",
            "type": "math",
            "expected": "8/9"
        },
        {
            "text": "After calculating, we get 586 grams",
            "type": "math",
            "expected": "586"
        },
        {
            "text": "Therefore, the final answer is 42.",
            "type": "math",
            "expected": "42"
        },
        {
            "text": "```python\ndef solve(n):\n    return n * 2\n```",
            "type": "code",
            "expected": "def solve(n):\n    return n * 2"
        },
        {
            "text": "The capital of France is Paris.",
            "type": "qa",
            "expected": "the capital of france is paris"
        },
    ]

    print("=" * 60)
    print("🧪 测试答案提取器 V2")
    print("=" * 60)

    for i, case in enumerate(test_cases, 1):
        result = extractor.extract_answer(case["text"], case["type"])
        print(f"\nTest {i} ({case['type']}):")
        print(f"  输入: {case['text'][:50]}...")
        print(f"  提取: {result}")
        print(f"  期望: {case['expected']}")
        print(f"  ✅ 通过" if result == case["expected"] else f"  ❌ 不匹配")


if __name__ == "__main__":
    test_extractor()
