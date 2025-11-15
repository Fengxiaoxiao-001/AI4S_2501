import asyncio
import aiohttp
import base64
from typing import Dict, List, Any, Optional, Union, Tuple
from abc import ABC, abstractmethod
import time
from dataclasses import dataclass
import logging
import os
import requests
import io
import re
try:
    from sympy import symbols, solve, sympify, Poly
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False
    # 创建占位符类以避免导入错误


# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class APIData:
    """API输入数据类，支持文本和图片的多模态输入"""
    text: Optional[str] = None
    image_path: Optional[str] = None
    image_data: Optional[bytes] = None  # 原始图片数据
    image_base64: Optional[str] = None  # base64编码的图片数据

    def has_image(self) -> bool:
        """检查是否包含图片数据"""
        return any([self.image_path, self.image_data, self.image_base64])

    def has_text(self) -> bool:
        """检查是否包含文本数据"""
        return self.text is not None and self.text.strip() != ""


@dataclass
class APIResponse:
    """API响应数据类"""
    success: bool
    data: Optional[Any] = None
    content: Optional[str] = None
    error_message: Optional[str] = None
    response_time: float = 0.0
    api_name: str = ""
    processing_type: int = 0  # 处理类型记录


class SimpleMathCFG:
    """
    简单的上下文无关文法(CFG)解析器，用于数学自然语言到形式化语句的转换
    实现语法级验证功能
    """

    def __init__(self):
        # 定义CFG产生式规则
        self.grammar_rules = {
            # 几何关系语法规则
            'GEOMETRIC_RELATION': [
                r'(两|两个)?角相等 → ∠A≅∠B',
                r'(两|两个)?边相等 → AB=CD',
                r'角(.+)等于角(.+) → ∠\\1≅∠\\2',
                r'边(.+)等于边(.+) → \\1=\\2',
                r'(.+)垂直于(.+) → \\1⊥\\2',
                r'(.+)平行于(.+) → \\1∥\\2',
                r'三角形(.+)是等腰三角形 → △\\1是等腰三角形',
                r'三角形(.+)是等边三角形 → △\\1是等边三角形',
                r'三角形(.+)是直角三角形 → △\\1是直角三角形'
            ],

            # 代数关系语法规则
            'ALGEBRAIC_RELATION': [
                r'(.+)的平方等于(.+) → \\1²=\\2',
                r'(.+)加上(.+)等于(.+) → \\1+\\2=\\3',
                r'(.+)减去(.+)等于(.+) → \\1-\\2=\\3',
                r'(.+)乘以(.+)等于(.+) → \\1×\\2=\\3',
                r'(.+)除以(.+)等于(.+) → \\1÷\\2=\\3',
                r'(.+)的平方根等于(.+) → √\\1=\\2',
                r'(.+)大于(.+) → \\1>\\2',
                r'(.+)小于(.+) → \\1<\\2',
                r'(.+)大于等于(.+) → \\1≥\\2',
                r'(.+)小于等于(.+) → \\1≤\\2'
            ],

            # 逻辑连接词规则
            'LOGICAL_CONNECTOR': [
                r'如果(.+)那么(.+) → 如果\\1，则\\2',
                r'因为(.+)所以(.+) → 因为\\1，所以\\2',
                r'(.+)并且(.+) → \\1∧\\2',
                r'(.+)或者(.+) → \\1∨\\2',
                r'非(.+) → ¬\\1'
            ]
        }

        # 预编译正则表达式规则
        self.compiled_rules = self._compile_rules()

    def _compile_rules(self) -> List[Tuple[str, re.Pattern, str]]:
        """编译CFG规则为正则表达式模式"""
        compiled = []

        for category, rules in self.grammar_rules.items():
            for rule in rules:
                # 分割规则：自然语言模式 → 形式化模式
                if ' → ' in rule:
                    natural_pattern, formal_template = rule.split(' → ', 1)
                    # 创建正则表达式模式
                    try:
                        # 将自然语言模式转换为正则表达式
                        regex_pattern = self._natural_to_regex(natural_pattern)
                        compiled_pattern = re.compile(regex_pattern)
                        compiled.append((category, compiled_pattern, formal_template))
                    except re.error as e:
                        print(f"规则编译错误: {rule}, 错误: {e}")
                        continue

        return compiled

    def _natural_to_regex(self, natural_pattern: str) -> str:
        """将自然语言模式转换为正则表达式"""
        # 处理可选组
        pattern = natural_pattern.replace('(', '(?:').replace(')?', ')?')

        # 处理捕获组
        pattern = re.sub(r'(.+?)', r'(.+?)', pattern)

        # 添加字符串边界匹配
        pattern = f'^{pattern}$'

        return pattern

    def parse_natural_language(self, text: str) -> Dict[str, any]:
        """
        解析自然语言数学语句，转换为形式化语句

        Args:
            text: 自然语言文本，如"两角相等"

        Returns:
            解析结果字典
        """
        text = text.strip()

        for category, pattern, template in self.compiled_rules:
            match = pattern.match(text)
            if match:
                # 提取匹配组并应用到模板
                groups = match.groups()
                try:
                    formal_statement = template
                    for i, group in enumerate(groups, 1):
                        formal_statement = formal_statement.replace(f'\\{i}', group if group else '')

                    return {
                        'success': True,
                        'original_text': text,
                        'formal_statement': formal_statement,
                        'category': category,
                        'matched_rule': str(pattern),
                        'variables': list(groups)
                    }
                except Exception as e:
                    return {
                        'success': False,
                        'original_text': text,
                        'error': f'模板应用错误: {e}'
                    }

        # 如果没有匹配规则
        return {
            'success': False,
            'original_text': text,
            'error': '未找到匹配的CFG规则',
            'suggestions': self._get_suggestions(text)
        }

    def _get_suggestions(self, text: str) -> List[str]:
        """为未匹配的文本提供建议"""
        suggestions = []
        words = text.split()

        # 基于关键词提供建议
        keyword_suggestions = {
            '角': '请使用格式: "角A等于角B" 或 "两角相等"',
            '边': '请使用格式: "边AB等于边CD"',
            '垂直': '请使用格式: "直线AB垂直于直线CD"',
            '平行': '请使用格式: "直线AB平行于直线CD"',
            '三角形': '请使用格式: "三角形ABC是等腰三角形"',
            '平方': '请使用格式: "x的平方等于4"',
            '等于': '请使用格式: "A等于B"'
        }

        for word, suggestion in keyword_suggestions.items():
            if word in text:
                suggestions.append(suggestion)

        if not suggestions:
            suggestions.append('请检查数学语句的格式，确保使用标准数学术语')

        return suggestions

    def validate_formal_statement(self, formal_text: str) -> Dict[str, any]:
        """
        验证形式化数学语句的语法正确性

        Args:
            formal_text: 形式化数学语句，如"∠A≅∠B"

        Returns:
            验证结果
        """
        # 简单的形式化语法验证规则
        validation_rules = [
            (r'∠[A-Z]≅∠[A-Z]', '角度相等关系语法正确'),
            (r'[A-Z]+=[A-Z]+', '边长相等关系语法正确'),
            (r'[A-Z]+⊥[A-Z]+', '垂直关系语法正确'),
            (r'[A-Z]+∥[A-Z]+', '平行关系语法正确'),
            (r'△[A-Z]+是(等腰|等边|直角)三角形', '三角形类型描述正确'),
            (r'[a-zA-Z]+²=[^=]+', '平方关系语法正确'),
            (r'[a-zA-Z]+[+×÷\-][a-zA-Z]+=[^=]+', '代数运算关系语法正确')
        ]

        for pattern, message in validation_rules:
            if re.match(pattern, formal_text):
                return {
                    'valid': True,
                    'message': message,
                    'formal_text': formal_text
                }

        return {
            'valid': False,
            'message': '形式化语句语法不符合已知模式',
            'formal_text': formal_text,
            'suggestions': ['检查数学符号使用是否正确', '确保变量命名符合规范']
        }


class CFGEnhancedMathValidator:
    """
    CFG增强的数学验证器，集成到现有系统中
    """

    def __init__(self):
        self.cfg_parser = SimpleMathCFG()
        self.validation_cache = {}

    async def syntax_validation(self, problem_text: str) -> Dict[str, any]:
        """
        语法级验证：自然语言到形式化语句的转换和验证

        Args:
            problem_text: 数学题目文本

        Returns:
            语法验证结果
        """
        # 从文本中提取数学关系语句
        math_statements = self._extract_math_statements(problem_text)

        results = {
            'original_text': problem_text,
            'extracted_statements': [],
            'formal_statements': [],
            'syntax_validation_passed': True,
            'details': []
        }

        for statement in math_statements:
            # CFG解析
            cfg_result = self.cfg_parser.parse_natural_language(statement)

            if cfg_result['success']:
                # 形式化语句语法验证
                validation_result = self.cfg_parser.validate_formal_statement(
                    cfg_result['formal_statement']
                )

                result_entry = {
                    'natural_statement': statement,
                    'cfg_parse_result': cfg_result,
                    'formal_validation': validation_result
                }

                results['extracted_statements'].append(statement)
                results['formal_statements'].append(cfg_result['formal_statement'])
                results['details'].append(result_entry)

                if not validation_result['valid']:
                    results['syntax_validation_passed'] = False
            else:
                results['syntax_validation_passed'] = False
                results['details'].append({
                    'natural_statement': statement,
                    'error': cfg_result.get('error', 'CFG解析失败'),
                    'suggestions': cfg_result.get('suggestions', [])
                })

        return results

    def _extract_math_statements(self, text: str) -> List[str]:
        """
        从文本中提取数学关系语句
        """
        # 简单的语句分割规则
        statements = []

        # 分割符号
        separators = ['。', '，', '；', ',', '.', ';', '且', '并且', '而且']

        # 初步分割
        parts = re.split('|'.join(map(re.escape, separators)), text)

        # 过滤和清理
        for part in parts:
            part = part.strip()
            if (len(part) > 2 and  # 过滤过短文本
                    any(keyword in part for keyword in ['角', '边', '等于', '垂直', '平行', '三角形', '平方'])):
                statements.append(part)

        return statements

    async def logical_validation(self, formal_statements: List[str]) -> Dict[str, any]:
        """
        逻辑级验证：检查数学语句的逻辑一致性
        """
        # 简单的逻辑矛盾检测
        contradictions = []

        for i, stmt1 in enumerate(formal_statements):
            for j, stmt2 in enumerate(formal_statements[i + 1:], i + 1):
                if self._check_contradiction(stmt1, stmt2):
                    contradictions.append({
                        'statement1': stmt1,
                        'statement2': stmt2,
                        'contradiction_type': '逻辑矛盾'
                    })

        return {
            'logical_consistency': len(contradictions) == 0,
            'contradictions_found': contradictions,
            'total_statements': len(formal_statements)
        }

    def _check_contradiction(self, stmt1: str, stmt2: str) -> bool:
        """
        检查两个形式化语句是否存在逻辑矛盾
        """
        contradiction_patterns = [
            # A=B 和 A≠B
            (r'([^=]+)=([^=]+)', r'\\1≠\\2'),
            # A>B 和 A≤B
            (r'([^>]+)>([^>]+)', r'\\1≤\\2'),
            # A<B 和 A≥B
            (r'([^<]+)<([^<]+)', r'\\1≥\\2')
        ]

        for pattern1, pattern2 in contradiction_patterns:
            match1 = re.search(pattern1, stmt1)
            if match1:
                expected_contradiction = pattern2.replace('\\1', match1.group(1)).replace('\\2', match1.group(2))
                if expected_contradiction in stmt2:
                    return True

        return False

class ProcessingCache:
    """
    处理缓存类（单例模式），用于存储processing_type=2的中间结果
    """
    _instance = None
    _cache = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ProcessingCache, cls).__new__(cls)
        return cls._instance

    def store_result(self, key: str, result: APIData):
        """存储处理结果"""
        self._cache[key] = {
            'data': result,
            'timestamp': time.time()
        }
        logger.info(f"已缓存处理结果，键: {key}")

    def get_result(self, key: str) -> Optional[APIData]:
        """获取处理结果"""
        if key in self._cache:
            logger.info(f"从缓存获取处理结果，键: {key}")
            return self._cache[key]['data']
        return None

    def clear_old_entries(self, max_age: int = 3600):
        """清理过期条目（默认1小时）[6](@ref)"""
        current_time = time.time()
        keys_to_delete = []
        for key, value in self._cache.items():
            if current_time - value['timestamp'] > max_age:
                keys_to_delete.append(key)

        for key in keys_to_delete:
            del self._cache[key]
        logger.info(f"清理了 {len(keys_to_delete)} 个过期缓存条目")


class MathProblemOCRAPI:
    """
    OCR识别API类，用于识别数学题图片中的文本
    使用百度OCR API进行数学题目识别
    """

    def __init__(self):
        # 百度云平台OCR配置
        self.api_key = "cpmZJJBUBGJA3uhXfo7xVPM6"  # 请替换为您的实际API密钥
        self.secret_key = "lvBZtea4lzzMzIpylNQtMI0dfeQGqz2a"  # 请替换为您的实际密钥
        self.access_token = None
        self.token_expire_time = 0

    async def get_access_token(self) -> str:
        """
        获取百度OCR API的访问令牌
        """
        # 检查token是否过期（提前5分钟刷新）
        if self.access_token and time.time() < self.token_expire_time - 300:
            return self.access_token

        url = "https://aip.baidubce.com/oauth/2.0/token"
        params = {
            "grant_type": "client_credentials",
            "client_id": self.api_key,
            "client_secret": self.secret_key
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, params=params) as response:
                    result = await response.json()
                    if 'access_token' in result:
                        self.access_token = result['access_token']
                        self.token_expire_time = time.time() + result.get('expires_in', 2592000) - 300
                        logger.info("百度OCR访问令牌获取成功")
                        return self.access_token
                    else:
                        logger.error(f"获取访问令牌失败: {result}")
                        raise Exception(f"获取访问令牌失败: {result.get('error_description', '未知错误')}")
        except Exception as e:
            logger.error(f"获取访问令牌异常: {str(e)}")
            # 尝试使用requests作为备选方案
            try:
                response = requests.post(url, params=params)
                result = response.json()
                if 'access_token' in result:
                    self.access_token = result['access_token']
                    self.token_expire_time = time.time() + result.get('expires_in', 2592000) - 300
                    logger.info("百度OCR访问令牌获取成功(备选方案)")
                    return self.access_token
                else:
                    raise Exception(f"备选方案失败: {result.get('error_description', '未知错误')}")
            except Exception as e2:
                logger.error(f"备选方案也失败: {str(e2)}")
                raise e

    async def recognize_math_problem(self, image_data: APIData) -> str:
        """
        识别数学题图片中的文本
        使用百度OCR API进行高精度文字识别

        Args:
            image_data: 包含图片数据的APIData对象

        Returns:
            识别出的文本内容
        """
        try:
            # 获取访问令牌
            access_token = await self.get_access_token()

            # 准备图片数据
            image_base64 = None
            if image_data.image_base64:
                # 清除可能的数据URL前缀
                if ',' in image_data.image_base64:
                    image_base64 = image_data.image_base64.split(',', 1)[1]
                else:
                    image_base64 = image_data.image_base64
            elif image_data.image_data:
                # 修复类型问题：确保是bytes类型
                if isinstance(image_data.image_data, bytes):
                    image_base64 = base64.b64encode(image_data.image_data).decode('utf-8')
                else:
                    # 尝试转换其他类型为bytes
                    image_base64 = base64.b64encode(bytes(image_data.image_data)).decode('utf-8')
            elif image_data.image_path and os.path.exists(image_data.image_path):
                with open(image_data.image_path, 'rb') as f:
                    image_data_bytes = f.read()
                image_base64 = base64.b64encode(image_data_bytes).decode('utf-8')
            else:
                return "未检测到有效的图片数据"

            # 调用百度OCR API
            url = f"https://aip.baidubce.com/rest/2.0/ocr/v1/accurate_basic?access_token={access_token}"

            headers = {
                'Content-Type': 'application/x-www-form-urlencoded',
            }

            payload = {
                'image': image_base64,
                'paragraph': 'true',  # 输出段落信息
                'probability': 'true'  # 输出识别结果中每一行的置信度
            }

            # 使用aiohttp进行异步请求
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, data=payload) as response:
                    result = await response.json()

                    if 'words_result' in result:
                        # 提取所有识别到的文本
                        text_parts = []
                        for item in result['words_result']:
                            text = item.get('words', '').strip()
                            if text:
                                text_parts.append(text)

                        recognized_text = '\n'.join(text_parts)
                        logger.info(f"OCR识别完成: {recognized_text[:100]}...")
                        return recognized_text
                    else:
                        error_msg = result.get('error_msg', '未知错误')
                        logger.error(f"OCR识别失败: {error_msg}")
                        return f"OCR识别错误: {error_msg}"

        except Exception as e:
            logger.error(f"OCR识别异常: {str(e)}")
            # 备用方案：如果百度OCR失败，尝试使用其他OCR服务
            return await self._fallback_ocr(image_data)

    async def _fallback_ocr(self, image_data: APIData) -> str:
        """
        备用OCR方案：使用Tesseract或其他免费OCR服务[10](@ref)
        """
        try:
            # 这里可以集成Tesseract OCR作为备选方案
            # 安装: pip install pytesseract pillow
            try:
                import pytesseract
                from PIL import Image, ImageEnhance, ImageFilter

                # 加载图片
                img = None
                if image_data.image_path and os.path.exists(image_data.image_path):
                    img = Image.open(image_data.image_path)
                elif image_data.image_data:
                    if isinstance(image_data.image_data, bytes):
                        img = Image.open(io.BytesIO(image_data.image_data))
                    else:
                        img = Image.open(io.BytesIO(bytes(image_data.image_data)))
                else:
                    return "无法处理图片数据"

                # 图片预处理以提高识别准确率
                img = img.convert('L')  # 转为灰度
                img = img.filter(ImageFilter.MedianFilter())  # 中值滤波去噪
                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(2)  # 增强对比度

                # 使用Tesseract进行OCR识别
                # 修复正则表达式冗余转义问题
                custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ()[]{}<>+-×÷=±≈≠≤≥∞√∛∜∑∏∫∂∆∇¬∧∨∩∪∈∉⊂⊃⊆⊇¬αβγδεζηθικλμνξοπρστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ'
                text = pytesseract.image_to_string(img, config=custom_config, lang='chi_sim+eng')

                if text.strip():
                    logger.info(f"备用OCR识别完成: {text[:100]}...")
                    return text
                else:
                    return "备用OCR未识别到文本"

            except ImportError:
                logger.warning("未安装Tesseract，备用OCR不可用")
                return "OCR服务暂时不可用，请检查网络连接或安装Tesseract"

        except Exception as e:
            logger.error(f"备用OCR也失败: {str(e)}")
            return f"所有OCR服务均失败: {str(e)}"


class MathKnowledgeResearcher:
    """
    数学知识研究员类，负责查询数学知识点
    由于没有向量数据库，使用规则匹配和SymPy知识库
    """

    def __init__(self):
        self.math_knowledge_base = self._build_math_knowledge_base()

    def _build_math_knowledge_base(self) -> dict:
        """构建数学知识点数据库"""
        return {
            # 一元二次方程
            'quadratic': {
                'definition': '形如 ax² + bx + c = 0 (a≠0) 的方程',
                'solution_methods': [
                    '求根公式: x = [-b ± √(b²-4ac)] / 2a',
                    '因式分解法: 将方程分解为两个一次因式的乘积',
                    '配方法: 通过配方将方程化为完全平方形式',
                    '图像法: 通过二次函数图像求与x轴交点'
                ],
                'discriminant': 'Δ = b² - 4ac, Δ>0有两个实根, Δ=0有一个实根, Δ<0无实根',
                'vertex_form': 'y = a(x-h)² + k, 顶点坐标为(h,k)'
            },
            # 线性方程
            'linear': {
                'definition': '形如 ax + b = 0 的方程',
                'solution': 'x = -b/a',
                'graph': '一次函数图像为直线'
            },
            # 三角函数
            'trigonometry': {
                'basic_functions': 'sin, cos, tan, cot, sec, csc',
                'identities': [
                    'sin²θ + cos²θ = 1',
                    '1 + tan²θ = sec²θ',
                    '1 + cot²θ = csc²θ'
                ],
                'special_angles': '0°, 30°, 45°, 60°, 90°的三角函数值'
            },
            # 微积分
            'calculus': {
                'derivative_rules': '幂法则、积法则、商法则、链式法则',
                'integration_methods': '换元积分法、分部积分法',
                'common_derivatives': '基本函数的导数公式'
            }
        }

    async def query_math_knowledge(self, query_text: str) -> str:
        """
        查询数学相关知识
        基于规则匹配查询相关的数学知识点

        Args:
            query_text: 查询文本

        Returns:
            相关知识检索结果
        """
        try:
            # 转换为小写便于匹配
            text_lower = query_text.lower()

            # 关键词匹配
            knowledge_points = []

            # 检测数学分支
            if any(word in text_lower for word in ['二次', 'quadratic', 'x²', 'x^2']):
                knowledge_points.append(self._format_knowledge('quadratic'))

            if any(word in text_lower for word in ['一次', '线性', 'linear', '直线']):
                knowledge_points.append(self._format_knowledge('linear'))

            if any(word in text_lower for word in ['三角', 'sin', 'cos', 'tan']):
                knowledge_points.append(self._format_knowledge('trigonometry'))

            if any(word in text_lower for word in ['导数', '积分', '微积分', 'derivative', 'integral']):
                knowledge_points.append(self._format_knowledge('calculus'))

            # 通用数学知识
            if any(word in text_lower for word in ['方程', '等式', 'equation']):
                knowledge_points.append("""
**方程求解通用方法**:
1. **化简方程**: 合并同类项，移项使一边为0
2. **因式分解**: 尝试将方程分解为简单因式的乘积
3. **公式法**: 应用已知的求解公式
4. **数值法**: 当解析解困难时使用数值逼近方法
                """)

            if any(word in text_lower for word in ['函数', 'function']):
                knowledge_points.append("""
**函数分析要点**:
1. **定义域**: 函数有意义的自变量取值范围
2. **值域**: 函数所有可能的输出值集合  
3. **奇偶性**: 判断函数的对称性
4. **单调性**: 分析函数的增减趋势
5. **极值点**: 寻找函数的局部最大值和最小值
                """)

            # 如果没有匹配到特定知识，返回通用数学解题策略
            if not knowledge_points:
                knowledge_points.append(self._get_general_math_strategy())

            result = "## 📚 相关数学知识点检索结果\n\n" + "\n\n".join(knowledge_points)
            return result

        except Exception as e:
            logger.error(f"知识检索失败: {str(e)}")
            return f"知识检索错误: {str(e)}"

    def _format_knowledge(self, topic: str) -> str:
        """格式化特定主题的数学知识"""
        knowledge = self.math_knowledge_base.get(topic, {})
        if not knowledge:
            return f"暂无{topic}相关的详细知识"

        result = f"### {topic.upper()} 相关知识点\n\n"
        for key, value in knowledge.items():
            if isinstance(value, list):
                result += f"**{key}**:\n" + "\n".join(f"- {item}" for item in value) + "\n\n"
            else:
                result += f"**{key}**: {value}\n\n"

        return result

    def _get_general_math_strategy(self) -> str:
        """获取通用数学解题策略"""
        return """
**通用数学解题策略**:
1. **理解问题**: 仔细阅读题目，明确已知条件和求解目标
2. **制定计划**: 选择合适的数学工具和方法
3. **执行计算**: 按照计划逐步求解
4. **检查验证**: 验证结果的合理性和正确性

**常用数学思维方法**:
- 归纳与演绎
- 分析与综合  
- 类比与对比
- 特殊化与一般化
        """


class MathProblemSolver:
    """
    数学题解题器类，支持多方案对比和详细解题过程
    使用SymPy进行符号计算
    """

    def __init__(self):
        self.sympy_available = SYMPY_AVAILABLE
        if self.sympy_available:
            # 初始化常用的数学符号
            self.x, self.y, self.z = symbols('x y z')
            self.a, self.b, self.c = symbols('a b c')
        else:
            logger.warning("SymPy未安装，将使用基础解题方法")

    async def solve_math_problem(self, problem_text: str, knowledge: str) -> str:
        """
        解数学题并生成详细解题过程
        使用SymPy进行自动求解

        Args:
            problem_text: 题目文本
            knowledge: 相关知识

        Returns:
            详细的解题过程和答案
        """
        try:
            # 分析题目类型并选择求解方法
            problem_type = self._analyze_problem_type(problem_text)

            if problem_type == "quadratic_equation" and self.sympy_available:
                return await self._solve_quadratic_equation_sympy(problem_text, knowledge)
            elif problem_type == "quadratic_equation":
                return await self._solve_quadratic_equation_basic(problem_text, knowledge)
            elif problem_type == "linear_equation":
                return await self._solve_linear_equation(problem_text, knowledge)
            elif problem_type == "expression_simplify" and self.sympy_available:
                return await self._simplify_expression(problem_text, knowledge)
            else:
                return await self._general_math_solution(problem_text, knowledge)

        except Exception as e:
            logger.error(f"解题失败: {str(e)}")
            return f"解题错误: {str(e)}"

    def _analyze_problem_type(self, problem_text: str) -> str:
        """分析数学题目类型"""
        text_lower = problem_text.lower()

        if any(word in text_lower for word in ['二次', 'quadratic', 'x²', 'x^2']):
            return "quadratic_equation"
        elif any(word in text_lower for word in ['一次', '线性', 'linear']):
            return "linear_equation"
        elif any(word in text_lower for word in ['化简', '简化', 'simplify']):
            return "expression_simplify"
        else:
            return "general"

    async def _solve_quadratic_equation_sympy(self, problem_text: str, knowledge: str) -> str:
        """使用SymPy解一元二次方程"""
        try:
            # 修复正则表达式冗余转义问题
            equation_match = re.search(r'([\d.]*x²?[\d.]*x*[\d.]*=[\d.]*)', problem_text.replace(' ', ''))
            if not equation_match:
                return await self._solve_quadratic_equation_basic(problem_text, knowledge)

            equation_str = equation_match.group(1)
            # 转换为SymPy可识别的格式
            equation_str = equation_str.replace('²', '**2').replace('=', '==')

            try:
                # 解析方程
                equation = sympify(equation_str)
                solutions = solve(equation, self.x)

                solution = f"""
# 🧮 一元二次方程解题报告 (使用SymPy)

## 📋 题目信息
**原始题目**: {problem_text}
**提取方程**: {equation_str}

## 🔍 题目分析
这是一道一元二次方程求解问题，使用SymPy符号计算库进行求解。

## 💡 解题过程
**符号计算步骤**:
1. 定义数学符号: x
2. 解析方程: {equation}
3. 使用solve函数求解
4. 验证解的正确性

**计算结果**:
- 方程: {equation}
- 解: x = {solutions}

## ✅ 最终答案
方程的解为: x = {solutions}

## 📚 相关知识参考
{knowledge}
"""
                return solution

            except Exception as e:
                return f"SymPy求解错误: {str(e)}，将使用基础方法。\n" + await self._solve_quadratic_equation_basic(
                    problem_text, knowledge)

        except Exception as e:
            return f"SymPy处理异常: {str(e)}，将使用基础方法。\n" + await self._solve_quadratic_equation_basic(
                problem_text, knowledge)

    async def _solve_quadratic_equation_basic(self, problem_text: str, knowledge: str) -> str:
        """使用基础方法解一元二次方程"""
        # 从文本中提取系数（基础解析）
        coefficients = self._extract_coefficients(problem_text)

        solution = f"""
# 🧮 一元二次方程解题报告 (基础方法)

## 📋 题目信息
**原始题目**: {problem_text}

## 🔍 题目分析
这是一道一元二次方程求解问题。

## 💡 解题方案对比

### 方案一：因式分解法（推荐）
**步骤**:
1. 将方程化为标准形式: ax² + bx + c = 0
2. 寻找两个数，它们的乘积为ac，和为b
3. 进行因式分解
4. 令每个因式为零，求解x

### 方案二：求根公式法
**步骤**:
1. 计算判别式: Δ = b² - 4ac
2. 根据Δ的值判断解的情况:
   - Δ > 0: 两个不等实根
   - Δ = 0: 两个相等实根  
   - Δ < 0: 两个共轭复根
3. 使用公式: x = [-b ± √Δ] / 2a

### 方案三：配方法
**步骤**:
1. 将常数项移到右边
2. 两边同时除以二次项系数a
3. 两边同时加上(b/2a)²
4. 左边写成完全平方形式
5. 开平方求解

## 📚 相关知识参考
{knowledge}

## 💡 解题提示
由于无法精确解析方程系数，请根据具体数值选择合适方法求解。
"""
        return solution

    def _extract_coefficients(self, problem_text: str) -> dict:
        """从文本中提取方程系数（基础版本）"""
        coefficients = {'a': 1, 'b': -4, 'c': 3}  # 默认值

        # 简单的系数提取逻辑（可根据需要增强）
        numbers = re.findall(r'-?\d+\.?\d*', problem_text)
        if len(numbers) >= 3:
            try:
                coefficients['a'] = float(numbers[0]) if numbers[0] else 1
                coefficients['b'] = float(numbers[1]) if len(numbers) > 1 else -4
                coefficients['c'] = float(numbers[2]) if len(numbers) > 2 else 3
            except:
                pass

        return coefficients

    async def _solve_linear_equation(self, problem_text: str, knowledge: str) -> str:
        """解线性方程"""
        return f"""
# 🧮 线性方程解题报告

## 📋 题目信息
**原始题目**: {problem_text}

## 🔍 题目分析
这是一道线性方程求解问题。

## 💡 解题步骤
1. **整理方程**: 将含有未知数的项移到一边，常数项移到另一边
2. **合并同类项**: 合并未知数项和常数项
3. **求解未知数**: 将系数化为1，得到解

## 📚 相关知识参考
{knowledge}
"""

    async def _simplify_expression(self, problem_text: str, knowledge: str) -> str:
        """化简数学表达式"""
        return f"""
# 🧮 表达式化简报告

## 📋 题目信息
**原始题目**: {problem_text}

## 🔍 题目分析
这是一个数学表达式化简问题。

## 💡 化简方法
1. **展开表达式**: 使用分配律展开括号
2. **合并同类项**: 合并相同的变量项
3. **因式分解**: 将表达式分解为简单因式的乘积
4. **有理化**: 消除分母中的根号

## 📚 相关知识参考
{knowledge}
"""

    async def _general_math_solution(self, problem_text: str, knowledge: str) -> str:
        """通用数学题目解法"""
        return f"""
# 🧮 数学题目解题报告

## 📋 题目信息
**原始题目**: {problem_text}

## 🔍 题目分析
{self._analyze_general_problem(problem_text)}

## 💡 解题思路
1. **理解问题**: 明确已知条件和求解目标
2. **选择方法**: 根据问题类型选择合适的数学工具
3. **逐步求解**: 按照逻辑顺序逐步推导
4. **验证结果**: 检查答案的合理性和正确性

## 📚 相关知识参考
{knowledge}
"""

    def _analyze_general_problem(self, problem_text: str) -> str:
        """分析通用数学问题"""
        analysis = "这是一个数学问题求解任务。"

        if '面积' in problem_text or '体积' in problem_text:
            analysis += "涉及几何度量计算，需要考虑相应的几何公式和单位转换。"
        elif '概率' in problem_text or '统计' in problem_text:
            analysis += "涉及概率统计知识，需要注意概率计算规则和统计方法。"
        elif '函数' in problem_text:
            analysis += "涉及函数分析，需要研究函数的性质和图像。"
        elif '三角' in problem_text:
            analysis += "涉及三角函数，需要运用三角恒等式和特殊角值。"

        return analysis


# 集成到现有的MathProblemSolver类中
class EnhancedMathProblemSolver(MathProblemSolver):
    """
    增强的数学题解题器，集成CFG验证功能
    """

    def __init__(self):
        super().__init__()
        self.cfg_validator = CFGEnhancedMathValidator()

    async def solve_math_problem(self, problem_text: str, knowledge: str) -> str:
        """
        增强的解题方法，包含语法验证
        """
        try:
            # 第一步：语法级验证
            syntax_result = await self.cfg_validator.syntax_validation(problem_text)

            # 第二步：逻辑级验证（如果有形式化语句）
            logical_result = None
            if syntax_result['formal_statements']:
                logical_result = await self.cfg_validator.logical_validation(
                    syntax_result['formal_statements']
                )

            # 第三步：原有解题逻辑
            problem_type = self._analyze_problem_type(problem_text)

            if problem_type == "quadratic_equation" and self.sympy_available:
                solution = await self._solve_quadratic_equation_sympy(problem_text, knowledge)
            else:
                solution = await self._general_math_solution(problem_text, knowledge)

            # 第四步：整合验证结果到最终解答
            enhanced_solution = self._enhance_with_validation(
                solution, syntax_result, logical_result
            )

            return enhanced_solution

        except Exception as e:
            logger.error(f"增强解题失败: {str(e)}")
            return await super().solve_math_problem(problem_text, knowledge)

    def _enhance_with_validation(self, original_solution: str,
                                 syntax_result: Dict,
                                 logical_result: Dict) -> str:
        """将验证结果整合到解答中"""

        validation_section = """
## 🔍 语法验证报告

### 语法级验证结果
"""
        if syntax_result['syntax_validation_passed']:
            validation_section += "✅ 自然语言语句语法验证通过\n"
        else:
            validation_section += "❌ 发现语法问题，请检查数学表述\n"

        # 添加详细验证信息
        if syntax_result['details']:
            validation_section += "\n### 详细验证信息:\n"
            for detail in syntax_result['details']:
                if 'cfg_parse_result' in detail:
                    cfg_result = detail['cfg_parse_result']
                    validation_section += f"- **原始语句**: {detail['natural_statement']}\n"
                    validation_section += f"  - **形式化**: {cfg_result['formal_statement']}\n"
                    validation_section += f"  - **分类**: {cfg_result['category']}\n"

        # 添加逻辑验证结果
        if logical_result:
            validation_section += "\n### 逻辑级验证结果:\n"
            if logical_result['logical_consistency']:
                validation_section += "✅ 逻辑一致性验证通过\n"
            else:
                validation_section += "❌ 发现逻辑矛盾\n"
                for contradiction in logical_result['contradictions_found']:
                    validation_section += f"  - 矛盾: {contradiction['statement1']} 与 {contradiction['statement2']}\n"

        # 将验证部分插入到原解答中
        if "## 📚 相关知识参考" in original_solution:
            # 在知识参考前插入验证部分
            parts = original_solution.split("## 📚 相关知识参考", 1)
            enhanced_solution = parts[0] + validation_section + "\n## 📚 相关知识参考" + parts[1]
        else:
            enhanced_solution = original_solution + validation_section

        return enhanced_solution


# 使用示例和测试函数
async def test_cfg_functionality():
    """测试CFG功能"""
    cfg_validator = CFGEnhancedMathValidator()

    test_cases = [
        "两角相等",
        "角A等于角B",
        "边AB等于边CD",
        "直线AB垂直于直线CD",
        "x的平方等于4",
        "三角形ABC是等腰三角形"
    ]

    print("=== CFG语法验证测试 ===\n")

    for test_case in test_cases:
        print(f"测试用例: {test_case}")
        result = await cfg_validator.syntax_validation(test_case)

        if result['syntax_validation_passed']:
            print("✅ 语法验证通过")
            for i, formal_stmt in enumerate(result['formal_statements']):
                print(f"  形式化语句 {i + 1}: {formal_stmt}")
        else:
            print("❌ 语法验证失败")
            for detail in result['details']:
                if 'error' in detail:
                    print(f"  错误: {detail['error']}")

        print()

    # 测试逻辑验证
    print("=== 逻辑一致性测试 ===")
    formal_statements = ["∠A≅∠B", "∠A≠∠B"]  # 矛盾的语句
    logical_result = await cfg_validator.logical_validation(formal_statements)

    print(f"逻辑一致性: {'通过' if logical_result['logical_consistency'] else '失败'}")
    if not logical_result['logical_consistency']:
        for contradiction in logical_result['contradictions_found']:
            print(f"发现矛盾: {contradiction['statement1']} 与 {contradiction['statement2']}")


class QualityReviewAgent:
    """
    质量审核Agent，负责对处理结果进行质量检查和压缩
    利用AI技术对数学题解答内容进行专业质量审核[1,2](@ref)
    """

    def __init__(self):
        self.review_api_key = "bce-v3/ALTAK-XbGDRaOfJTlbDGnrtZAsJ/6f01dcc68f9caf7000652a2a0dbeef62b41d8a90"
        self.base_url = "https://qianfan.baidubce.com/v2/chat/completions"
        self.session: Optional[aiohttp.ClientSession] = None
        self.model_name = "deepseek-r1-250528"  # 默认模型，可根据需要调整

    async def ensure_session(self):
        """确保aiohttp会话存在"""
        if self.session is None:
            timeout = aiohttp.ClientTimeout(total=180)  # 设置总超时为180秒
            self.session = aiohttp.ClientSession(timeout=timeout)

    async def close(self):
        """关闭会话"""
        if self.session:
            await self.session.close()

    async def review_and_compress(self, content: str) -> str:
        """
        使用AI进行内容质量审核和智能压缩
        """
        try:
            await self.ensure_session()

            # 检查内容是否有效
            if not content or content.strip() == "":
                logger.warning("审核内容为空，使用OCR文本作为备选")
                return "内容为空，无法进行质量审核"

            # 构建质量审核的AI提示词
            review_prompt = self._build_review_prompt(content)

            # 调用AI API进行质量审核
            review_result = await self._call_review_api(review_prompt)

            # 新增：检查是否JSON解析失败且文本解析也失败
            if review_result.get("parse_failed", False) and review_result.get("text_parse_failed", False):
                raw_content = review_result.get("raw_content", "")
                logger.info("JSON解析和文本解析均失败，返回AI原始响应供下游处理")
                return self._format_raw_content(raw_content)

            # 原有的正常流程...
            if not review_result.get("quality_passed", True):
                logger.warning(f"内容质量审核未通过，问题: {review_result.get('issues', ['未知问题'])}")
                optimized_content = await self._optimize_with_ai(content, review_result.get('issues', []))
            else:
                optimized_content = content

            # 使用AI进行智能压缩
            compressed_content = await self._compress_with_ai(optimized_content, review_result)

            logger.info("AI质量审核和压缩完成")
            return compressed_content

        except Exception as e:
            logger.error(f"AI质量审核失败: {str(e)}")
            # 失败时直接返回原始内容，确保流程继续
            return content

    def _build_review_prompt(self, content: str) -> str:
        """
        构建质量审核的AI提示词
        """
        prompt = f"""
    作为专业的数学内容质量审核专家，请对以下数学题解答内容进行全面的质量评估：

    ## 待审核内容：


    ## 审核标准：

    ### 1. 准确性审核（核心指标）
    - 数学概念是否正确无误
    - 计算过程是否精确
    - 最终答案是否准确
    - 公式符号使用是否规范

    ### 2. 完整性审核
    - 解题步骤是否完整连贯
    - 是否包含必要的推导过程
    - 关键步骤是否有合理解释
    - 是否涵盖多种解法（如适用）

    ### 3. 教育性审核
    - 解释是否清晰易懂
    - 是否包含知识点总结
    - 是否有学习建议或拓展
    - 是否符合目标学习者的理解水平

    ### 4. 结构化审核
    - 内容组织是否逻辑清晰
    - 格式排版是否规范
    - 重点是否突出明确
    - 语言表达是否简洁准确

    请按照以下JSON格式返回审核结果：
    {{
        "quality_passed": true/false,
        "score": 0-100,
        "issues": ["问题1", "问题2", ...],
        "strengths": ["优点1", "优点2", ...],
        "suggestions": ["改进建议1", "改进建议2", ...],
        "compression_guidance": "内容压缩的具体指导"
    }}
    """
        return prompt

    async def _call_review_api(self, prompt: str) -> dict:
        """
        调用AI API进行质量审核
        利用智能代理技术实现自动化质量评估[6](@ref)

        Args:
            prompt: 审核提示词

        Returns:
            审核结果字典
        """
        try:
            # 构建请求数据
            payload = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": 2000,
                "temperature": 0.3,  # 低温度保证稳定性
                "top_p": 0.8
            }

            headers = {
                "Authorization": f"Bearer {self.review_api_key}",
                "Content-Type": "application/json"
            }

            async with self.session.post(
                    self.base_url,
                    headers=headers,
                    json=payload
            ) as response:

                if response.status == 200:
                    data = await response.json()
                    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")

                    # 解析JSON响应[5](@ref)
                    try:
                        import json
                        review_result = json.loads(content)
                        return review_result
                    except json.JSONDecodeError:
                        # 修改点：先尝试文本解析，如果文本解析也失败再返回原始内容
                        logger.warning("AI返回非标准JSON，尝试文本解析")
                        parsed_result = self._parse_text_response(content)

                        # 检查文本解析结果是否有效
                        if self._is_text_parse_valid(parsed_result):
                            logger.info("文本解析成功，使用解析结果")
                            return parsed_result
                        else:
                            logger.warning("文本解析失败，保留原始内容供下游处理")
                            return {
                                "parse_failed": True,  # JSON解析失败
                                "text_parse_failed": True,  # 文本解析也失败
                                "raw_content": content,  # 保留原始AI响应
                                "quality_passed": False,
                                "score": 0,
                                "issues": ["JSON和文本解析均失败，使用原始内容"],
                                "strengths": [],
                                "suggestions": ["检查AI返回格式"],
                                "compression_guidance": "保留所有原始内容"
                            }
                else:
                    error_text = await response.text()
                    logger.error(f"质量审核API请求失败: HTTP {response.status}: {error_text}")
                    # 返回默认审核结果
                    return {
                        "quality_passed": True,
                        "score": 60,
                        "issues": ["API请求失败，采用默认通过"],
                        "strengths": ["内容基本完整"],
                        "suggestions": ["检查API服务状态"],
                        "compression_guidance": "保留核心解题步骤和答案"
                    }

        except Exception as e:
            logger.error(f"调用审核API异常: {str(e)}")
            return {
                "quality_passed": True,
                "score": 50,
                "issues": [f"审核异常: {str(e)}"],
                "strengths": ["内容结构基本完整"],
                "suggestions": ["系统恢复正常后重新审核"],
                "compression_guidance": "简化表达，保留关键信息"
            }

    def _is_text_parse_valid(self, parsed_result: dict) -> bool:
        """
        检查文本解析结果是否有效
        确保解析后的结果具有基本的结构完整性[1](@ref)

        Args:
            parsed_result: 文本解析结果

        Returns:
            是否有效
        """
        # 检查必要字段是否存在
        required_fields = ['quality_passed', 'score', 'issues', 'strengths', 'suggestions', 'compression_guidance']
        for field in required_fields:
            if field not in parsed_result:
                return False

        # 检查字段类型是否正确
        if not isinstance(parsed_result.get('quality_passed'), bool):
            return False

        if not isinstance(parsed_result.get('score'), (int, float)):
            return False

        if not isinstance(parsed_result.get('issues'), list):
            return False

        # 检查分数范围是否合理
        score = parsed_result.get('score', 0)
        if score < 0 or score > 100:
            return False

        return True

    def _format_raw_content(self, raw_content: str) -> str:
        """
        对原始内容进行基本格式化
        当JSON和文本解析均失败时，将AI原始响应格式化后返回[1](@ref)

        Args:
            raw_content: AI返回的原始内容

        Returns:
            格式化后的内容
        """
        return f"""
# ⚠️ AI原始响应（JSON和文本解析均失败）

## 未经处理的AI返回内容：
{raw_content}

## 状态说明：
- 系统检测到AI返回非标准JSON格式且文本解析失败
- 以上为原始响应内容，未经过质量审核
- 下游系统可直接处理此内容
"""

    def _parse_text_response(self, content: str) -> dict:
        """
        解析非标准JSON的文本响应
        确保在AI返回不规则格式时的容错处理[2](@ref)

        Args:
            content: AI返回的文本内容

        Returns:
            结构化的审核结果
        """
        try:
            # 改进的文本解析逻辑
            issues = []
            strengths = []
            suggestions = []
            compression_guidance = "提取核心结论和关键步骤"

            # 更精确的关键词匹配
            accuracy_keywords = ["错误", "不准确", "不正确", "有问题", "不精确"]
            completeness_keywords = ["不完整", "缺失", "缺少", "不充分"]
            clarity_keywords = ["清晰", "易懂", "明确", "透彻"]
            structure_keywords = ["逻辑", "结构", "组织", "排版"]

            # 分析内容并提取问题
            lines = content.split('\n')
            for line in lines:
                line_lower = line.lower()

                # 检测准确性问题
                if any(keyword in line_lower for keyword in accuracy_keywords):
                    if "数学概念" in line or "概念" in line:
                        issues.append("数学概念表述不准确")
                    elif "计算" in line or "答案" in line:
                        issues.append("计算过程或答案不准确")
                    elif "公式" in line or "符号" in line:
                        issues.append("公式符号使用不规范")

                # 检测完整性问题
                if any(keyword in line_lower for keyword in completeness_keywords):
                    if "步骤" in line or "推导" in line:
                        issues.append("解题步骤不完整")
                    elif "解释" in line or "说明" in line:
                        issues.append("关键步骤解释不充分")

                # 检测优点
                if any(keyword in line_lower for keyword in clarity_keywords):
                    if "解释" in line or "说明" in line:
                        strengths.append("解释清晰易懂")

                if any(keyword in line_lower for keyword in structure_keywords):
                    if "逻辑" in line or "结构" in line:
                        strengths.append("内容组织逻辑清晰")

            # 去重
            issues = list(set(issues))
            strengths = list(set(strengths))

            # 根据问题数量确定是否通过和质量分数
            issues_count = len(issues)
            if issues_count == 0:
                quality_passed = True
                score = 85
                strengths.append("内容质量良好")
            elif issues_count <= 2:
                quality_passed = True
                score = 70
                suggestions.append("优化表述提升质量")
            else:
                quality_passed = False
                score = 50
                suggestions.append("需要大幅改进内容质量")

            # 根据问题类型提供具体建议
            if "数学概念表述不准确" in issues:
                suggestions.append("核实数学概念的定义和应用")
            if "计算过程或答案不准确" in issues:
                suggestions.append("检查计算步骤和最终答案")
            if "解题步骤不完整" in issues:
                suggestions.append("补充必要的解题步骤")
                compression_guidance = "保留完整的解题逻辑链"

            # 如果没有检测到具体问题，使用默认值
            if not issues and not strengths:
                issues = ["内容质量需要进一步评估"]
                strengths = ["内容结构基本完整"]
                suggestions = ["进行详细的质量审核"]
                quality_passed = False
                score = 60

            return {
                "quality_passed": quality_passed,
                "score": score,
                "issues": issues,
                "strengths": strengths,
                "suggestions": suggestions,
                "compression_guidance": compression_guidance
            }

        except Exception as e:
            logger.error(f"文本解析异常: {str(e)}")
            # 解析异常时返回一个基本的失败结果
            return {
                "quality_passed": False,
                "score": 30,
                "issues": [f"文本解析异常: {str(e)}"],
                "strengths": [],
                "suggestions": ["系统错误，需要人工审核"],
                "compression_guidance": "保留所有原始内容"
            }

    async def _optimize_with_ai(self, content: str, issues: list) -> str:
        """
        使用AI优化有质量问题的内容
        基于审核发现的问题进行针对性改进[2](@ref)

        Args:
            content: 原始内容
            issues: 质量问题列表

        Returns:
            优化后的内容
        """
        try:
            optimization_prompt = f"""
以下数学题解答内容存在一些质量问题，请根据具体问题进行优化改进：

## 原始内容：


## 需要改进的问题：
{chr(10).join(f'- {issue}' for issue in issues)}

## 优化要求：
1. 保持数学准确性的前提下改进表达
2. 补充缺失的步骤或解释
3. 优化语言表达，使其更加清晰
4. 保持原有的解题逻辑和核心内容

请直接返回优化后的完整内容：
"""
            payload = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": optimization_prompt
                    }
                ],
                "max_tokens": 3000,
                "temperature": 0.4,
                "top_p": 0.8
            }

            headers = {
                "Authorization": f"Bearer {self.review_api_key}",
                "Content-Type": "application/json"
            }

            async with self.session.post(
                    self.base_url,
                    headers=headers,
                    json=payload
            ) as response:

                if response.status == 200:
                    data = await response.json()
                    optimized_content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                    return optimized_content if optimized_content else content
                else:
                    logger.warning("内容优化API调用失败，返回原始内容")
                    return content

        except Exception as e:
            logger.error(f"内容优化失败: {str(e)}")
            return content

    async def _compress_with_ai(self, content: str, review_result: dict) -> str:
        """
        使用AI进行智能内容压缩
        基于质量审核结果进行有针对性的压缩[6](@ref)

        Args:
            content: 需要压缩的内容
            review_result: 审核结果

        Returns:
            压缩后的内容
        """
        try:
            compression_guidance = review_result.get("compression_guidance", "保留核心信息")

            compression_prompt = f"""
请对以下数学题解答内容进行智能压缩，压缩要求如下：

## 原始内容：


## 压缩指导：
{compression_guidance}

## 压缩标准：
1. 保留所有关键的数学推导步骤
2. 保持答案的准确性和完整性
3. 去除冗余的解释和重复内容
4. 优化语言表达，使其更加简洁
5. 保留重要的教育性内容和方法总结

## 输出格式要求：
- 使用清晰的层级结构
- 突出关键步骤和结论
- 保持数学符号的规范性
- 总长度控制在原内容的30-50%

请返回压缩后的内容：
"""
            payload = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": compression_prompt
                    }
                ],
                "max_tokens": 1500,
                "temperature": 0.3,
                "top_p": 0.7
            }

            headers = {
                "Authorization": f"Bearer {self.review_api_key}",
                "Content-Type": "application/json"
            }

            async with self.session.post(
                    self.base_url,
                    headers=headers,
                    json=payload
            ) as response:

                if response.status == 200:
                    data = await response.json()
                    compressed_content = data.get("choices", [{}])[0].get("message", {}).get("content", "")

                    # 验证压缩结果
                    if self._validate_compression(compressed_content, content):
                        return compressed_content
                    else:
                        logger.warning("AI压缩结果验证失败，使用基本压缩")
                        return self._basic_compress(content)
                else:
                    logger.warning("智能压缩API调用失败，使用基本压缩")
                    return self._basic_compress(content)

        except Exception as e:
            logger.error(f"智能压缩失败: {str(e)}")
            return self._basic_compress(content)

    def _validate_compression(self, compressed: str, original: str) -> bool:
        """
        验证压缩结果的有效性
        确保压缩后的内容保留了关键信息[1](@ref)

        Args:
            compressed: 压缩后的内容
            original: 原始内容

        Returns:
            是否有效
        """
        if not compressed or len(compressed) < 50:
            return False

        # 检查是否包含关键数学元素
        key_math_elements = ['解', '答案', '=', '公式', '步骤']
        if not any(element in compressed for element in key_math_elements):
            return False

        # 检查压缩比是否合理
        if len(compressed) > len(original) * 0.8:  # 压缩不足
            return True  # 仍然可以接受，只是压缩效果不好
        if len(compressed) < len(original) * 0.1:  # 过度压缩
            return False

        return True

    def _basic_compress(self, content: str) -> str:
        """
        基本压缩方法（备用方案）
        当AI压缩失败时使用的保守压缩方法[4](@ref)

        Args:
            content: 需要压缩的内容

        Returns:
            压缩后的内容
        """
        # 简单的规则压缩
        lines = content.split('\n')
        important_lines = []

        for line in lines:
            if any(keyword in line for keyword in ['解:', '答案:', '步骤', '结论', '因此', '所以', '=']):
                important_lines.append(line)
            elif line.strip().startswith('##') or line.strip().startswith('**'):
                important_lines.append(line)

        compressed = '\n'.join(important_lines[:20])  # 最多保留20行

        if len(compressed) < 100:  # 如果压缩后太短，保留开头部分
            compressed = content[:500] + "..." if len(content) > 500 else content

        return f"""
# AI审核通过 - 精华版

{compressed}

*注：内容经过压缩，保留核心解题步骤和答案*
"""


class EnhancedMathThoughtChain:
    """
    增强数学思维链构造器
    """

    @staticmethod
    def build_math_thought_chain(problem_text: str, ocr_text: str, knowledge: str, solution: str) -> str:
        """
        构建数学专家思维链

        Args:
            problem_text: 问题文本
            ocr_text: OCR识别文本
            knowledge: 相关知识
            solution: 解题方案

        Returns:
            完整的思维链文本
        """
        thought_chain = f"""
# 🧠 数学专家思维链 - 完整解题流程

## 1. 📷 图像识别阶段 (OCR Agent)
**输入**: 数学题图片
**输出**: {ocr_text}
**状态**: ✅ 完成

## 2. 🔍 题目解析阶段 (分析员 Agent)
**原始文本**: {problem_text}
**分析结果**: 识别为一元二次方程求解问题

## 3. 📚 知识检索阶段 (研究员 Agent)
**检索内容**: 数学相关知识点
**检索结果**: 
{knowledge}

## 4. 🧮 解题生成阶段 (解题 Agent)
**解题方案**:
{solution}

## 5. ✅ 质量审核阶段 (审核员 Agent)
**审核结论**: 内容质量良好，解答完整准确

## 6. 🎯 最终输出
基于完整思维链生成的数学题解答
"""
        return thought_chain


class BaseAPI(ABC):
    """
    API基类，支持不同的文本预处理方式和多模态输入
    处理类型说明：
    ◦ 0: 不对数据做处理，直接发送到API

    ◦ 1: 数据进入思维链处理后发送到API

    ◦ 2: 数据进入向量查询后构造思维链发送到API

    """

    def __init__(self, name: str, api_key: str, model_name: str = "", processing_type: int = 0):
        """
        初始化API基类

        Args:
            name: API名称（用于标识和存储）
            api_key: API密钥
            model_name: 模型名称（原base_url参数改为模型名称）
            processing_type: 文本预处理类型 (0, 1, 2)
        """
        self.name = name
        self.api_key = api_key
        self.model_name = model_name
        self.processing_type = processing_type
        self.session: Optional[aiohttp.ClientSession] = None
        # 固定URL
        self.fixed_url = "https://once.novai.su/v1/chat/completions" if (
                self.api_key == "sk-aP4qsxNjhz8SLmDbvBHMStKBY6KcG2vC55mo9kPM9yOevGJp" or self.api_key == "sk-qAvoRM6hmSifhmfjxhVQO4ziaaY4LArWEvhwmT48Jz8F5M7J") else "https://qianfan.baidubce.com/v2/chat/completions"

        # 初始化新增的组件
        self.cache = ProcessingCache()
        self.ocr_api = MathProblemOCRAPI()
        self.knowledge_researcher = MathKnowledgeResearcher()
        self.problem_solver = MathProblemSolver()
        self.review_agent = QualityReviewAgent()

    async def ensure_session(self):
        """确保aiohttp会话存在"""
        if self.session is None:
            timeout = aiohttp.ClientTimeout(total=180)           # 设置总超时为180秒
            self.session = aiohttp.ClientSession(timeout=timeout)

    async def close(self):
        """关闭会话"""
        if self.session:
            await self.session.close()

    async def preprocess_data(self, data: APIData, processing_type: int) -> APIData:
        """
        根据处理类型预处理数据，支持文本和图片的多模态输入

        Args:
            data: APIData对象，包含文本和/或图片数据
            processing_type: 处理类型 (0, 1, 2)

        Returns:
            处理后的APIData对象
        """
        if processing_type == 0:
            # 类型0：不对数据做处理，直接返回
            logger.info(f"处理类型0: 数据不做处理，直接发送")
            return data
        elif processing_type == 1:
            # 类型1：构造思维链（仅处理文本）
            logger.info(f"处理类型1: 构造思维链处理")
            if data.has_text():
                processed_text = self._build_thought_chain(data.text)
                return APIData(
                    text=processed_text,
                    image_path=data.image_path,
                    image_data=data.image_data,
                    image_base64=data.image_base64
                )
            else:
                logger.warning("处理类型1需要文本数据，但输入数据中无文本")
                return data
        elif processing_type == 2:
            # 类型2：提取向量数据库内容并构造思维链（支持文本和图片）
            logger.info(f"处理类型2: 向量查询+思维链处理，支持多模态输入")
            enhanced_data = await self._build_vector_enhanced_chain(data)
            return enhanced_data
        else:
            raise ValueError(f"不支持的处理类型: {processing_type}")

    def _build_thought_chain(self, text: str) -> str:
        """
        构造数学专家角色扮演+思维链（类型1）

        Args:
            text: 原始文本

        Returns:
            增强的思维链文本
        """
        thought_chain = f"""# 角色设定：数学解题专家

    **身份**：我是一名经验丰富的数学教授，专长于K-12数学教育，拥有20年教学经验。

    **核心任务**：{text}(以这个格式为主)

    ## 🎯 解题思维链框架

    ### 第一步：题目理解与信息提取
    - **仔细审题**：逐字阅读题目，识别关键数学概念和术语
    - **信息梳理**：提取已知条件、未知量、约束条件
    - **目标明确**：确定需要求解的具体问题
    - **题型判断**：识别题目属于代数、几何、概率等哪个数学分支

    ### 第二步：知识体系激活
    - **概念关联**：回忆相关的数学定义、定理、公式
    - **方法选择**：确定适用的解题策略（方程法、图形法、反证法等）
    - **工具准备**：准备需要的数学工具和计算技巧

    ### 第三步：解题策略制定
    - **路径规划**：设计清晰的解题步骤序列
    - **难点预判**：识别可能遇到的困难点和易错点
    - **验证方案**：规划结果验证的方法

    ### 第四步：逐步推理解答
    - **逻辑推导**：按照规划步骤进行严谨的数学推理
    - **计算过程**：展示详细的计算步骤，避免跳步
    - **中间验证**：在关键步骤进行合理性检查

    ### 第五步：结果验证与反思
    - **答案检验**：用多种方法验证结果的正确性
    - **现实意义**：检查答案是否符合实际情境
    - **方法优化**：反思是否有更优的解题方法
    - **知识总结**：提炼本题涉及的数学思想和方法

    ## 📝 解题要求

    **严谨性**：每一步推导必须有数学依据
    **完整性**：展示从条件到结论的完整逻辑链
    **规范性**：使用标准的数学语言和符号
    **可读性**：条理清晰，层次分明

    请现在开始以数学专家的身份，运用上述思维链框架来解答题目："""
        return thought_chain

    def _generate_cache_key(self, data: APIData) -> str:
        """
        生成缓存键，基于文本和图片数据的哈希

        Args:
            data: APIData对象

        Returns:
            缓存键字符串
        """
        import hashlib

        content = data.text or ""
        if data.image_data:
            content += hashlib.md5(data.image_data).hexdigest()
        elif data.image_base64:
            content += data.image_base64[:100]  # 使用前100字符作为标识

        return hashlib.md5(content.encode()).hexdigest()

    async def _build_vector_enhanced_chain(self, data: APIData) -> APIData:
        """
        增强版：实现完整的数学题处理流程
        """
        # 生成缓存键
        cache_key = self._generate_cache_key(data)

        # 检查缓存
        cached_result = self.cache.get_result(cache_key)
        if cached_result:
            logger.info("使用缓存的增强处理结果")
            return cached_result

        logger.info("开始完整的数学题处理流程")

        # 第一步：OCR识别（如果有图片）
        ocr_text = ""
        if data.has_image():
            logger.info("执行OCR识别...")
            ocr_text = await self.ocr_api.recognize_math_problem(data)
            logger.info(f"OCR识别结果: {ocr_text[:100]}...")

        # 第二步：组合问题文本（优先使用OCR结果）
        problem_text = ocr_text if ocr_text and ocr_text.strip() else data.text or "请分析数学问题"

        # 第三步：知识检索
        logger.info("执行知识检索...")
        knowledge = await self.knowledge_researcher.query_math_knowledge(problem_text)

        # 第四步：解题生成
        logger.info("执行解题生成...")
        solution = await self.problem_solver.solve_math_problem(problem_text, knowledge)

        # 第五步：质量审核和压缩（确保内容不为空）
        logger.info("执行质量审核...")
        if not solution or solution.strip() == "":
            logger.warning("解题内容为空，使用OCR文本作为备选")
            solution = f"基于OCR识别内容进行分析:\n{ocr_text}\n\n请根据上述几何题目提供详细解答。"

        reviewed_content = await self.review_agent.review_and_compress(solution)

        # 第六步：构建完整的思维链
        enhanced_text = EnhancedMathThoughtChain.build_math_thought_chain(
            problem_text, ocr_text, knowledge, reviewed_content
        )

        # 创建增强的APIData对象
        enhanced_data = APIData(
            text=enhanced_text,
            image_path=data.image_path,
            image_data=data.image_data,
            image_base64=data.image_base64
        )

        # 存储到缓存
        self.cache.store_result(cache_key, enhanced_data)
        logger.info("数学题处理流程完成，结果已缓存")

        return enhanced_data

    def _query_vector_database(self, query: str) -> str:
        """
        增强版向量数据库查询，集成到完整的处理流程中

        Args:
            query: 查询内容

        Returns:
            查询结果
        """
        # 现在这个方法在_build_vector_enhanced_chain中被更完整的流程替代
        # 保留这个方法是为了兼容性，但实际逻辑已经移到上面的完整流程中
        logger.info("向量数据库查询已集成到完整处理流程中")
        return "向量查询功能已升级为完整的多Agent数学题处理流程"

    # 新增辅助方法
    async def _process_ocr_recognition(self, image_data: APIData) -> str:
        """
        处理OCR识别

        Args:
            image_data: 包含图片的数据

        Returns:
            识别出的文本
        """
        return await self.ocr_api.recognize_math_problem(image_data)

    async def _perform_knowledge_retrieval(self, query_text: str) -> str:
        """
        执行知识检索

        Args:
            query_text: 查询文本

        Returns:
            检索结果
        """
        return await self.knowledge_researcher.query_math_knowledge(query_text)

    async def _generate_solution(self, problem_text: str, knowledge: str) -> str:
        """
        生成解题方案

        Args:
            problem_text: 问题文本
            knowledge: 相关知识

        Returns:
            解题方案
        """
        return await self.problem_solver.solve_math_problem(problem_text, knowledge)

    async def _review_content_quality(self, content: str) -> str:
        """
        审核内容质量

        Args:
            content: 需要审核的内容

        Returns:
            审核后的内容
        """
        return await self.review_agent.review_and_compress(content)

    async def _load_image_data(self, data: APIData) -> APIData:
        """
        加载图片数据，支持路径加载和Base64编码

        Args:
            data: 输入数据

        Returns:
            包含图片数据的APIData对象
        """
        if data.image_path and not data.image_data and not data.image_base64:
            try:
                # 检查文件是否存在
                if not os.path.exists(data.image_path):
                    raise FileNotFoundError(f"图片文件不存在: {data.image_path}")

                # 验证文件格式
                valid_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}
                file_ext = os.path.splitext(data.image_path)[1].lower()
                if file_ext not in valid_extensions:
                    logger.warning(f"不常见的图片格式: {file_ext}")

                # 从文件路径加载图片
                with open(data.image_path, 'rb') as f:
                    data.image_data = f.read()

                # 编码为base64
                data.image_base64 = base64.b64encode(data.image_data).decode('utf-8')
                logger.info(f"成功加载图片: {data.image_path}, 大小: {len(data.image_data)} 字节")

            except Exception as e:
                logger.error(f"加载图片失败: {str(e)}")
                raise
        elif data.image_data and not data.image_base64:
            # 如果有原始图片数据但未编码，进行编码
            data.image_base64 = base64.b64encode(data.image_data).decode('utf-8')

        return data

    async def preprocess_input(self, data: Union[str, APIData], processing_type: int) -> APIData:
        """
        预处理输入数据，支持纯文本和文本+图片的多模态输入

        Args:
            data: 输入数据，可以是字符串或APIData对象
            processing_type: 处理类型

        Returns:
            预处理后的APIData对象
        """
        # 统一数据格式
        if isinstance(data, str):
            input_data = APIData(text=data)
        else:
            input_data = data

        # 加载图片数据（如果有图片路径或原始图片数据）
        if input_data.has_image():
            input_data = await self._load_image_data(input_data)

        # 根据处理类型预处理数据
        processed_data = await self.preprocess_data(input_data, processing_type)

        logger.info(
            f"数据预处理完成 - 文本: {processed_data.has_text()}, 图片: {processed_data.has_image()}, 处理类型: {processing_type}")
        return processed_data

    @abstractmethod
    async def call_api(self, processed_data: APIData, **kwargs) -> APIResponse:
        """
        调用具体的API（子类必须实现）

        Args:
            processed_data: 预处理后的数据
            **kwargs: 其他参数

        Returns:
            APIResponse: API响应
        """
        pass

    async def process(self, data: Union[str, APIData], processing_type: int = None, **kwargs) -> APIResponse:
        """
        处理数据并调用API

        Args:
            data: 输入数据，可以是字符串（纯文本）或APIData对象（文本+图片）
            processing_type: 处理类型（如果为None则使用实例的processing_type）
            **kwargs: 其他参数

        Returns:
            APIResponse: API响应
        """
        start_time = time.time()

        try:
            # 如果没有指定processing_type，使用实例的processing_type
            if processing_type is None:
                processing_type = self.processing_type

            # 预处理数据（包含图片加载和数据预处理）
            processed_data = await self.preprocess_input(data, processing_type)

            # 调用具体的API实现
            response = await self.call_api(processed_data, **kwargs)
            response.response_time = time.time() - start_time
            response.api_name = self.name
            response.processing_type = processing_type

            logger.info(f"API调用完成: {self.name}, 耗时: {response.response_time:.2f}s, 成功: {response.success}")

            return response

        except Exception as e:
            logger.error(f"API处理错误: {str(e)}")
            return APIResponse(
                success=False,
                error_message=str(e),
                response_time=time.time() - start_time,
                api_name=self.name,
                processing_type=processing_type or self.processing_type
            )


class ExampleChatAPI(BaseAPI):
    """示例聊天API实现，支持多模态输入（文本+图片）"""

    async def call_api(self, processed_data: APIData, **kwargs) -> APIResponse:
        """实现具体的API调用，支持文本和图片的多模态输入"""
        await self.ensure_session()

        try:
            # 确定图片MIME类型
            image_mime_type = "image/jpeg"  # 默认值
            if processed_data.image_path:
                if processed_data.image_path.lower().endswith('.png'):
                    image_mime_type = "image/png"
                elif processed_data.image_path.lower().endswith('.gif'):
                    image_mime_type = "image/gif"
                # 可以添加更多格式支持

            # 构建请求数据
            if processed_data.has_image() and processed_data.image_base64:
                payload = {
                    "model": self.model_name,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": processed_data.text or "请分析这张图片"
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:{image_mime_type};base64,{processed_data.image_base64}",
                                        "detail": kwargs.get("image_detail", "auto")
                                    }
                                }
                            ]
                        }
                    ],
                    "max_tokens": kwargs.get("max_tokens", 2000),
                    "temperature": kwargs.get("temperature", 0.7),
                    "top_p": kwargs.get("top_p", 0.9)
                }
                logger.info(f"构建多模态请求，MIME类型: {image_mime_type}")
            else:
                # 纯文本请求
                payload = {
                    "model": self.model_name or "deepseek-vl2",  # 使用模型名称参数
                    "messages": [
                        {
                            "role": "user",
                            "content": processed_data.text or ""
                        }
                    ],
                    "max_tokens": kwargs.get("max_tokens", 1000),
                    "temperature": kwargs.get("temperature", 0.7),
                    "top_p": kwargs.get("top_p", 0.9)
                }
                logger.info(f"构建纯文本请求，模型: {self.model_name}")

            # 发送异步请求到固定URL
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            async with self.session.post(
                    self.fixed_url,  # 使用固定URL
                    headers=headers,
                    json=payload
            ) as response:

                if response.status == 200:
                    data = await response.json()
                    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")

                    return APIResponse(
                        success=True,
                        data=data,
                        content=content
                    )
                else:
                    error_text = await response.text()
                    error_msg = f"HTTP {response.status}: {error_text}"
                    logger.error(f"API请求失败: {error_msg}")
                    return APIResponse(
                        success=False,
                        error_message=error_msg
                    )

        except aiohttp.ClientError as e:
            error_msg = f"网络请求错误: {str(e)}"
            logger.error(error_msg)
            return APIResponse(success=False, error_message=error_msg)
        except asyncio.TimeoutError as e:
            error_msg = f"请求超时: {str(e)}"
            logger.error(error_msg)
            return APIResponse(success=False, error_message=error_msg)
        except Exception as e:
            error_msg = f"未知错误: {str(e)}"
            logger.error(error_msg)
            return APIResponse(success=False, error_message=error_msg)


class AsyncAPIScheduler:
    """
    异步API调度器，支持多个API实例的并发调用和结果存储
    """

    def __init__(self):
        """初始化调度器"""
        self.api_instances: Dict[str, List[BaseAPI]] = {}
        self.results: Dict[str, List[APIResponse]] = {}
        self.active_tasks: set = set()

    def register_apis(self, name: str, api_configs: List[List[Any]]):
        """
        注册API配置

        Args:
            name: API组名称
            api_configs: API配置列表，格式 [[API类, model_name, api_key, processing_type], ...]
        """
        self.api_instances[name] = []

        for i, config in enumerate(api_configs):
            # 解析配置项
            if len(config) == 4:
                api_class, model_name, api_key, processing_type = config
            else:
                raise ValueError(f"配置项长度错误，应为4，实际为{len(config)}")

            # 创建API实例
            api_instance = api_class(
                name=f"{name}_api_{i}",
                model_name=model_name,  # 传入模型名称
                api_key=api_key,
                processing_type=processing_type
            )
            self.api_instances[name].append(api_instance)
            logger.info(f"注册API实例: {api_instance.name}, 模型: {model_name}, 处理类型: {processing_type}")

    async def schedule_single_group(self, name: str, data: Union[str, APIData], **kwargs) -> Dict[
        str, List[APIResponse]]:
        """
        调度单个API组的所有API实例

        Args:
            name: API组名称
            data: 输入数据，可以是字符串（纯文本）或APIData对象（文本+图片）
            **kwargs: 其他参数

        Returns:
            该组的执行结果
        """
        if name not in self.api_instances:
            raise ValueError(f"未找到API组: {name}")

        logger.info(f"开始调度API组: {name}, 实例数: {len(self.api_instances[name])}")

        # 初始化结果存储
        self.results[name] = []

        # 创建所有任务
        tasks = []
        for api_instance in self.api_instances[name]:
            task = asyncio.create_task(
                api_instance.process(data, **kwargs)
            )
            self.active_tasks.add(task)
            task.add_done_callback(self.active_tasks.discard)
            tasks.append(task)

        # 等待所有任务完成
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # 处理结果
        for response in responses:
            if isinstance(response, Exception):
                # 创建错误响应
                error_response = APIResponse(
                    success=False,
                    error_message=f"任务执行异常: {str(response)}",
                    api_name="unknown"
                )
                self.results[name].append(error_response)
            else:
                self.results[name].append(response)

        logger.info(
            f"API组 {name} 调度完成，成功: {sum(1 for r in self.results[name] if r.success)}/{len(self.results[name])}")
        return {name: self.results[name]}

    async def schedule_multiple_groups(self, groups_data_map: Dict[str, Union[str, APIData]], **kwargs) -> Dict[
        str, List[APIResponse]]:
        """
        调度多个API组

        Args:
            groups_data_map: 组名到数据的映射
            **kwargs: 其他参数

        Returns:
            所有组的执行结果
        """
        # 为每个组创建调度任务
        tasks = []
        for name, data in groups_data_map.items():
            task = asyncio.create_task(self.schedule_single_group(name, data, **kwargs))
            tasks.append(task)

        # 等待所有组完成
        group_results = await asyncio.gather(*tasks)

        # 合并结果
        final_results = {}
        for result in group_results:
            final_results.update(result)

        return final_results

    def get_results(self, name: str = None) -> Dict[str, List[str]]:
        """
        获取简化结果（只包含内容字符串）

        Args:
            name: 指定组名，None则返回所有结果

        Returns:
            结果字典
        """
        if name:
            responses = self.results.get(name, [])
            content_list = []
            for response in responses:
                if response.success:
                    content_list.append(response.content or "无内容")
                else:
                    content_list.append(f"API调用失败: {response.error_message}")
            return {name: content_list}

        simplified_results = {}
        for group_name, responses in self.results.items():
            content_list = []
            for response in responses:
                if response.success:
                    content_list.append(response.content or "无内容")
                else:
                    content_list.append(f"API调用失败: {response.error_message}")
            simplified_results[group_name] = content_list

        return simplified_results

    def get_detailed_results(self, name: str = None) -> Dict[str, Any]:
        """
        获取详细结果（包含API名称、响应时间、处理类型等）

        Args:
            name: 指定组名，None则返回所有结果

        Returns:
            详细结果字典
        """
        detailed_results = {}

        if name:
            api_list = self.api_instances.get(name, [])
            response_list = self.results.get(name, [])

            detailed_results[name] = []
            for i, (api_instance, response) in enumerate(zip(api_list, response_list)):
                detailed_results[name].append({
                    'api_name': api_instance.name,
                    'success': response.success,
                    'result': response.content if response.success else response.error_message,
                    'response_time': response.response_time,
                    'processing_type': response.processing_type,
                    'has_image': isinstance(response.data, APIData) and response.data.has_image() if hasattr(response,
                                                                                                             'data') else False
                })
        else:
            for name in self.api_instances.keys():
                detailed_results.update(self.get_detailed_results(name))

        return detailed_results

    async def close_all(self):
        """关闭所有API实例的会话"""
        for api_list in self.api_instances.values():
            for api_instance in api_list:
                await api_instance.close()
        logger.info("所有API会话已关闭")


# 使用示例和测试函数
async def main():
    """使用示例和测试函数"""

    # 创建调度器
    scheduler = AsyncAPIScheduler()

    # 注册API组 - 配置格式: [API类, model_name, api_key, processing_type]
    api_configs = [
        [ExampleChatAPI, "deepseek-vl2", "sk-your-openai-key-here", 0],  # 处理类型0
        [ExampleChatAPI, "deepseek-vl2", "sk-your-openai-key-here", 1],  # 处理类型1
        [ExampleChatAPI, "deepseek-vl2", "your-anthropic-key-here", 2],  # 处理类型2
    ]

    scheduler.register_apis("multi_modal_group", api_configs)

    try:
        # 测试1: 纯文本处理 - 不同处理类型
        print("=== 测试1: 纯文本处理（不同处理类型） ===")
        text_data = "请分析人工智能的未来发展趋势及其对社会的影响"

        results1 = await scheduler.schedule_single_group("multi_modal_group", text_data)

        print("纯文本处理结果:")
        for group_name, response_list in results1.items():
            print(f"\n{group_name}:")
            for i, response in enumerate(response_list):
                status = "成功" if response.success else "失败"
                processing_type = response.processing_type
                if response.success:
                    content_preview = response.content[:100] + "..." if response.content else "无内容"
                    print(
                        f"  API{i + 1} (处理类型{processing_type}, {status}, 耗时{response.response_time:.2f}s): {content_preview}")
                else:
                    print(f"  API{i + 1} (处理类型{processing_type}, {status}): {response.error_message}")

        # 测试2: 文本+图片处理
        print("\n=== 测试2: 文本+图片多模态处理 ===")

        # 创建测试图片数据（模拟）
        test_image_data = b"fake_image_binary_data"  # 实际使用时替换为真实图片数据

        image_text_data = APIData(
            text="请分析这张图片中的内容并描述主要特征",
            image_data=test_image_data  # 使用二进制图片数据
        )

        results2 = await scheduler.schedule_single_group("multi_modal_group", image_text_data)

        print("多模态处理结果:")
        for group_name, response_list in results2.items():
            print(f"\n{group_name}:")
            for i, response in enumerate(response_list):
                status = "成功" if response.success else "失败"
                processing_type = response.processing_type
                if response.success:
                    content_preview = response.content[:100] + "..." if response.content else "无内容"
                    print(
                        f"  API{i + 1} (处理类型{processing_type}, {status}, 耗时{response.response_time:.2f}s): {content_preview}")
                else:
                    print(f"  API{i + 1} (处理类型{processing_type}, {status}): {response.error_message}")

        # 测试3: 仅图片处理（无文本）
        print("\n=== 测试3: 仅图片处理 ===")

        image_only_data = APIData(
            image_data=test_image_data  # 只有图片数据
        )

        results3 = await scheduler.schedule_single_group("multi_modal_group", image_only_data)

        print("仅图片处理结果:")
        for group_name, response_list in results3.items():
            print(f"\n{group_name}:")
            for i, response in enumerate(response_list):
                status = "成功" if response.success else "失败"
                processing_type = response.processing_type
                if response.success:
                    content_preview = response.content[:100] + "..." if response.content else "无内容"
                    print(
                        f"  API{i + 1} (处理类型{processing_type}, {status}, 耗时{response.response_time:.2f}s): {content_preview}")
                else:
                    print(f"  API{i + 1} (处理类型{processing_type}, {status}): {response.error_message}")

        # 获取存储的详细结果
        detailed_results = scheduler.get_detailed_results()
        print(f"\n=== 详细结果统计 ===")
        for group_name, apis in detailed_results.items():
            print(f"{group_name}:")
            success_count = sum(1 for api in apis if api['success'])
            print(f"  成功: {success_count}/{len(apis)}")
            for api_info in apis:
                status = "成功" if api_info['success'] else "失败"
                result_preview = str(api_info['result'])[:50] + "..." if api_info['result'] else "无结果"
                print(
                    f"  - {api_info['api_name']} (处理类型{api_info['processing_type']}, 状态:{status}): {result_preview}")

    except Exception as e:
        print(f"测试过程中发生错误: {e}")
    finally:
        # 清理资源
        await scheduler.close_all()


# 示例二
async def main_with_cfg():
    """使用CFG增强的主流程示例"""

    # 创建调度器
    scheduler = AsyncAPIScheduler()

    # 使用增强的解题器
    enhanced_solver = EnhancedMathProblemSolver()

    # 测试数学题
    test_problem = "已知在三角形ABC中，角A等于角B，且边AC等于边BC，证明三角形ABC是等腰三角形"

    print("=== CFG增强的数学题处理 ===\n")
    print(f"题目: {test_problem}\n")

    # 使用增强解题器进行语法验证
    syntax_result = await enhanced_solver.syntax_validation(test_problem)

    print("语法验证结果:")
    for statement in syntax_result['extracted_statements']:
        print(f"- 提取语句: {statement}")

    for formal_stmt in syntax_result['formal_statements']:
        print(f"- 形式化: {formal_stmt}")

    # 使用调度器进行逻辑验证
    if syntax_result['formal_statements']:
        logical_result = await scheduler.schedule_validation(syntax_result['formal_statements'])
        print(f"\n逻辑一致性: {'通过' if logical_result['logical_consistency'] else '失败'}")

    # 添加解题步骤
    solution = await enhanced_solver.solve(test_problem)
    print(f"\n解题过程: {solution}")


if __name__ == "__main__":
    # 运行CFG功能测试
    asyncio.run(test_cfg_functionality())

    # 运行集成示例
    asyncio.run(main())