import asyncio
import json
import statistics
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
import logging
import aiohttp
from datetime import datetime
import os
import re

from InputAndOutput_Layer import APIData, APIResponse, BaseAPI

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SmartReviewSystem")


class DimensionScore:
    """七个维度评分类（基于教育目标分类）"""

    def __init__(self, memorization: float = 0.0, comprehension: float = 0.0,
                 application: float = 0.0, analysis: float = 0.0,
                 synthesis: float = 0.0, evaluation: float = 0.0,
                 innovation: float = 0.0):
        """
        初始化七个维度评分

        Args:
            memorization: 识记：复述基本事实与概念
            comprehension: 理解：解释概念、原理与过程
            application: 应用：在标准场景下运用知识解决问题
            analysis: 分析：分解复杂信息，辨析逻辑关系
            synthesis: 综合：整合多种要素，构建新的解决方案或完整论述
            evaluation: 评价：依据既定标准进行批判性判断与评估
            innovation: 创新：提出新颖的假设、见解或探索性方案
        """
        self.memorization: float = memorization  # 识记：复述基本事实与概念
        self.comprehension: float = comprehension  # 理解：解释概念、原理与过程
        self.application: float = application  # 应用：在标准场景下运用知识解决问题
        self.analysis: float = analysis  # 分析：分解复杂信息，辨析逻辑关系
        self.synthesis: float = synthesis  # 综合：整合多种要素，构建新的解决方案或完整论述
        self.evaluation: float = evaluation  # 评价：依据既定标准进行批判性判断与评估
        self.innovation: float = innovation  # 创新：提出新颖的假设、见解或探索性方案

    def to_list(self) -> List[float]:
        """转换为列表"""
        return [
            self.memorization, self.comprehension, self.application,
            self.analysis, self.synthesis, self.evaluation, self.innovation
        ]

    def to_dict(self) -> Dict[str, float]:
        """转换为字典"""
        return {
            "memorization": self.memorization,
            "comprehension": self.comprehension,
            "application": self.application,
            "analysis": self.analysis,
            "synthesis": self.synthesis,
            "evaluation": self.evaluation,
            "innovation": self.innovation
        }


@dataclass
class ModelResponseRecord:
    """测试模型回答记录"""
    model_name: str  # 模型显示名称
    internal_id: str  # 模型内部ID
    input_content: str  # 模型输入的内容（题目）
    response_content: str  # 模型的回答内容
    is_correct: Optional[bool] = None  # 回答正确与否（在统合时确定）
    response_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class AIEvaluationResult:
    """单个AI的评测结果"""
    ai_id: str  # AI标识
    dimension_scores: DimensionScore  # 七个维度评分
    final_score: float = 0.0  # 最终加权得分
    evaluation_text: str = ""  # 评价文本
    model_response_record: Optional[ModelResponseRecord] = None  # 新增：模型回答记录
    success: bool = True  # 新增：回答是否正确

    def calculate_final_score(self, weights: Dict[str, float], apply_penalty: bool = True) -> float:
        """
        计算加权最终得分

        Args:
            weights: 权重配置
            apply_penalty: 是否应用惩罚（错误回答乘以0.8）

        Returns:
            最终得分
        """
        scores_dict = self.dimension_scores.to_dict()

        # 如果回答错误且需要应用惩罚，则对每个维度评分乘以0.8
        penalty_factor = 0.8 if (apply_penalty and not self.success) else 1.0

        weighted_sum = sum(scores_dict[dim] * weight * penalty_factor for dim, weight in weights.items())
        total_weight = sum(weights.values())
        self.final_score = weighted_sum / total_weight if total_weight > 0 else 0
        return self.final_score


@dataclass
class ReviewerVote:
    """评审智能体投票结果"""
    reviewer_id: str
    ai_evaluations: Dict[str, AIEvaluationResult]  # AI ID -> 评测结果
    confidence: float = 1.0  # 置信度
    model_response_judgments: Dict[str, bool] = field(default_factory=dict)  # 新增：对每个模型回答正确性的判断


class DebateMode(Enum):
    """辩论模式枚举"""
    DISABLED = 0  # 关闭辩论模式
    ENABLED = 1  # 开启辩论模式


class ResponseRecorder:
    """模型回答记录器"""

    def __init__(self, storage_dir: str = "model_responses"):
        """
        初始化记录器

        Args:
            storage_dir: 存储目录
        """
        self.storage_dir = storage_dir
        self.records: Dict[str, ModelResponseRecord] = {}  # internal_id -> record
        os.makedirs(storage_dir, exist_ok=True)

    def record_response(self, internal_id: str, model_name: str, input_content: str,
                        response_content: str) -> ModelResponseRecord:
        """
        记录模型回答

        Args:
            internal_id: 模型内部ID
            model_name: 模型显示名称
            input_content: 输入内容
            response_content: 回答内容

        Returns:
            记录对象
        """
        record = ModelResponseRecord(
            model_name=model_name,
            internal_id=internal_id,
            input_content=input_content,
            response_content=response_content
        )
        self.records[internal_id] = record
        logger.info(f"记录模型回答: {model_name} ({internal_id})")
        return record

    def determine_correctness(self, internal_id: str, is_correct: bool) -> None:
        """
        确定回答正确性（在统合评审结果时调用）

        Args:
            internal_id: 模型内部ID
            is_correct: 是否正确
        """
        if internal_id in self.records:
            self.records[internal_id].is_correct = is_correct
            logger.info(f"设置模型回答正确性: {internal_id} -> {is_correct}")

    def save_to_json(self, filename: str = None) -> str:
        """
        保存记录到JSON文件

        Args:
            filename: 文件名，如果为None则自动生成

        Returns:
            文件路径
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"model_responses_{timestamp}.json"

        filepath = os.path.join(self.storage_dir, filename)

        # 转换为可序列化的字典
        records_dict = {}
        for internal_id, record in self.records.items():
            records_dict[internal_id] = {
                "model_name": record.model_name,
                "internal_id": record.internal_id,
                "input_content": record.input_content,
                "response_content": record.response_content,
                "is_correct": record.is_correct,
                "response_timestamp": record.response_timestamp,
                "success": record.is_correct if record.is_correct is not None else False  # 新增：success字段
            }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(records_dict, f, ensure_ascii=False, indent=2)

        logger.info(f"模型回答记录已保存: {filepath}")
        return filepath

    def load_from_json(self, filepath: str) -> None:
        """
        从JSON文件加载记录

        Args:
            filepath: 文件路径
        """
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                records_dict = json.load(f)

            for internal_id, record_data in records_dict.items():
                self.records[internal_id] = ModelResponseRecord(
                    model_name=record_data["model_name"],
                    internal_id=record_data["internal_id"],
                    input_content=record_data["input_content"],
                    response_content=record_data["response_content"],
                    is_correct=record_data.get("is_correct"),
                    response_timestamp=record_data["response_timestamp"]
                )

            logger.info(f"模型回答记录已加载: {filepath}")


class SmartReviewSystem:
    """自动化智能评审系统"""

    def __init__(self, debate_mode: DebateMode = DebateMode.DISABLED,
                 response_storage_dir: str = "model_responses",
                 scoring_criteria: List[Dict[str, Any]] = None,
                 historical_data: List[Dict[str, Any]] = None):
        """
        初始化评审系统

        Args:
            debate_mode: 辩论模式开关
            response_storage_dir: 模型回答存储目录
            scoring_criteria: 评分标准数据
            historical_data: 历史评测数据
        """
        self.debate_mode = debate_mode
        self.reviewers = []  # 评审智能体列表
        self.dimension_weights = {  # 七个维度权重配置
            "memorization": 0.05,  # 识记权重5%
            "comprehension": 0.15,  # 理解权重15%
            "application": 0.20,  # 应用权重20%
            "analysis": 0.25,  # 分析权重25%
            "synthesis": 0.15,  # 综合权重15%
            "evaluation": 0.15,  # 评价权重15%
            "innovation": 0.05  # 创新权重5%
        }
        self.debate_threshold = 20  # 辩论触发阈值（分差）
        self.response_recorder = ResponseRecorder(response_storage_dir)  # 新增：回答记录器
        self.model_responses: Dict[str, str] = {}  # 新增：存储模型回答内容（internal_id -> response）
        self.scoring_criteria = scoring_criteria or []  # 评分标准数据
        self.historical_data = historical_data or []  # 历史评测数据

    def add_reviewer(self, reviewer: 'ReviewerAgent'):
        """添加评审智能体"""
        self.reviewers.append(reviewer)
        logger.info(f"添加评审智能体: {reviewer.reviewer_id}")

    def record_model_response(self, internal_id: str, model_name: str, input_content: str,
                              response_content: str) -> None:
        """
        记录测试模型的回答内容

        Args:
            internal_id: 模型内部ID
            model_name: 模型显示名称
            input_content: 输入内容
            response_content: 回答内容
        """
        # 存储到回答记录器
        self.response_recorder.record_response(internal_id, model_name, input_content, response_content)

        # 同时存储到内存中供评审使用
        self.model_responses[internal_id] = response_content
        logger.info(f"已记录模型回答: {model_name} ({internal_id}), 内容长度: {len(response_content)}")

    async def process_evaluation(self, input_data: Any, ai_count: int = 8, model_names: List[str] = None) -> Dict[
        str, Any]:
        """
        处理完整的评测流程

        Args:
            input_data: 输入数据（APIData对象，包含文本和图片）
            ai_count: AI数量（默认8个）
            model_names: 被评测模型的有序名称列表，如["Qwen2.5-72B-Instruct", "gpt-4o-mini", ...]
                        如果为None，则使用默认的AI1, AI2...标识符

        Returns:
            完整的评测结果
        """
        logger.info("开始智能评审流程")

        # 验证模型名称列表
        if model_names is not None:
            if len(model_names) != ai_count:
                logger.warning(f"模型名称列表长度({len(model_names)})与AI数量({ai_count})不匹配，使用默认命名")
                model_names = None
            else:
                logger.info(f"使用指定的模型名称列表: {model_names}")

        # 生成模型标识符映射
        self.model_id_mapping = self._create_model_id_mapping(ai_count, model_names)

        # 阶段0: 记录模型回答（假设模型回答已通过其他方式获得并调用record_model_response记录）
        # 这里需要您在实际调用时预先记录模型回答

        # 阶段1: 数据整合
        final_data = await self._integrate_data(input_data, ai_count, self.model_id_mapping)

        # 阶段2: 分布式评审
        votes = await self._distributed_review(final_data, ai_count, self.model_id_mapping)

        # 阶段3: 多维度评分统计
        dimension_stats = self._calculate_dimension_statistics(votes, self.model_id_mapping)

        # 阶段4: 仲裁与决策
        final_results = await self._arbitration_center(votes, dimension_stats, self.model_id_mapping)

        logger.info("智能评审流程完成")
        return final_results

    def _create_model_id_mapping(self, ai_count: int, model_names: List[str] = None) -> Dict[str, str]:
        """
        创建模型标识符映射

        Args:
            ai_count: AI数量
            model_names: 模型名称列表

        Returns:
            映射字典：内部ID -> 显示名称
        """
        mapping = {}

        for i in range(ai_count):
            internal_id = f"AI{i + 1}"
            if model_names and i < len(model_names):
                display_name = model_names[i]
            else:
                display_name = internal_id  # 使用默认标识符

            mapping[internal_id] = display_name

        logger.info(f"创建模型标识符映射: {mapping}")
        return mapping

    async def _integrate_data(self, input_data: Any, ai_count: int, model_id_mapping: Dict[str, str]) -> Any:
        """
        整合题目和记忆体数据

        Args:
            input_data: 原始输入数据（APIData对象）
            ai_count: AI数量
            model_id_mapping: 模型标识符映射

        Returns:
            整合后的最终数据（APIData对象）
        """
        logger.info("开始数据整合")

        # 构建增强的输入数据，包含模型回答内容
        enhanced_text = self._build_enhanced_input(input_data.text, ai_count, model_id_mapping)

        # 创建新的APIData对象，保留原始图片数据
        final_data = type(input_data)()  # 创建相同类型的对象
        final_data.text = enhanced_text

        # 复制图片相关属性
        if hasattr(input_data, 'image_path'):
            final_data.image_path = input_data.image_path
        if hasattr(input_data, 'image_data'):
            final_data.image_data = input_data.image_data
        if hasattr(input_data, 'image_base64'):
            final_data.image_base64 = input_data.image_base64

        logger.info("数据整合完成")
        return final_data

    def _build_enhanced_input(self, original_text: str, ai_count: int, model_id_mapping: Dict[str, str]) -> str:
        """构建增强的输入文本，包含模型回答内容"""
        criteria_text = "评分标准:\n" + "\n".join(
            [str(item.get('content', '')) for item in self.scoring_criteria]) if self.scoring_criteria else "使用默认评分标准"
        history_text = "历史参考:\n" + "\n".join(
            [str(item.get('content', '')) for item in self.historical_data[:3]]) if self.historical_data else "无历史参考数据"

        # 构建模型回答内容部分
        response_section = "模型回答内容:\n"
        for internal_id, display_name in model_id_mapping.items():
            response_content = self.model_responses.get(internal_id, "暂无回答内容")
            # 限制显示长度，避免提示词过长
            preview_content = response_content[:500] + "..." if len(response_content) > 500 else response_content
            response_section += f"{display_name} ({internal_id}): {preview_content}\n\n"

        # 构建模型列表描述
        model_list = ", ".join([f"{internal_id}({display_name})"
                                for internal_id, display_name in model_id_mapping.items()])

        enhanced_text = f"""
{original_text}

{criteria_text}

{history_text}

{response_section}

请根据以上内容对以下{ai_count}个AI模型的解题能力进行七个维度评分，并判断每个模型的回答是否正确：
{model_list}
        """
        return enhanced_text

    async def _distributed_review(self, final_data: Any, ai_count: int, model_id_mapping: Dict[str, str]) -> List[
        ReviewerVote]:
        """
        分布式评审 - 三个评审智能体并行评审

        Args:
            final_data: 整合后的数据（APIData对象）
            ai_count: AI数量
            model_id_mapping: 模型标识符映射

        Returns:
            三个评审智能体的投票结果
        """
        logger.info("开始分布式评审")

        review_tasks = []
        for reviewer in self.reviewers:
            task = reviewer.evaluate_ais(final_data, ai_count, model_id_mapping)
            review_tasks.append(task)

        # 使用asyncio.gather并设置return_exceptions=True避免一个任务失败影响其他任务
        results = await asyncio.gather(*review_tasks, return_exceptions=True)

        votes = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"评审智能体 {self.reviewers[i].reviewer_id} 执行失败: {result}")
                # 生成默认投票结果
                default_vote = self.reviewers[i]._generate_default_vote(ai_count, model_id_mapping)
                votes.append(default_vote)
            else:
                votes.append(result)

        logger.info(f"分布式评审完成，成功{len(votes)}个评审结果")
        return votes

    def _calculate_dimension_statistics(self, votes: List[ReviewerVote], model_id_mapping: Dict[str, str]) -> Dict[
        str, Dict[str, Any]]:
        """
        计算七个维度的统计信息

        Args:
            votes: 评审投票结果
            model_id_mapping: 模型标识符映射

        Returns:
            维度统计信息
        """
        logger.info("开始计算维度统计")

        dimension_stats = {}
        dimension_names = ["memorization", "comprehension", "application", "analysis", "synthesis", "evaluation",
                           "innovation"]

        for dimension in dimension_names:
            dimension_data = {}

            for internal_id, display_name in model_id_mapping.items():
                scores = []

                # 收集三个评审对该AI该维度的评分
                for vote in votes:
                    if internal_id in vote.ai_evaluations:
                        score = getattr(vote.ai_evaluations[internal_id].dimension_scores, dimension)
                        scores.append(score)

                if scores:
                    dimension_data[display_name] = {
                        "scores": scores,
                        "average": statistics.mean(scores),
                        "std_dev": statistics.stdev(scores) if len(scores) > 1 else 0,
                        "max_diff": max(scores) - min(scores) if scores else 0
                    }

            dimension_stats[dimension] = dimension_data

        logger.info("维度统计计算完成")
        return dimension_stats

    async def _arbitration_center(self, votes: List[ReviewerVote],
                                  dimension_stats: Dict[str, Dict[str, Any]],
                                  model_id_mapping: Dict[str, str]) -> Dict[str, Any]:
        """
        仲裁中心 - 处理评分差异和辩论

        Args:
            votes: 投票结果
            dimension_stats: 维度统计信息
            model_id_mapping: 模型标识符映射

        Returns:
            最终评审结果
        """
        logger.info("进入仲裁中心")

        # 阶段4.1: 先统合评审结果，确定回答正确性（在计算得分之前）
        await self._finalize_evaluation_before_scoring(votes)

        # 阶段4.2: 检查是否需要辩论（仅当开启模式时计算）
        if self.debate_mode == DebateMode.ENABLED:
            debate_cases = self._identify_debate_cases(dimension_stats)
        else:
            debate_cases = []

        final_votes = votes.copy()

        # 阶段4.3: 如果开启辩论模式且存在需要辩论的情况
        if self.debate_mode == DebateMode.ENABLED and debate_cases:
            logger.info(f"发现{len(debate_cases)}个需要辩论的情况，开始辩论流程")
            final_votes = await self._conduct_debate(votes, debate_cases)

        # 阶段4.4: 生成最终评价（应用惩罚机制）
        final_evaluation = await self._generate_final_evaluation(final_votes, model_id_mapping)

        # 初始化最终结果字典
        final_results = {
            'debate_cases': debate_cases,
            'final_votes': self._serialize_votes(final_votes),
            'dimension_stats': dimension_stats,
            'model_id_mapping': model_id_mapping
        }

        # 阶段4.5: 保存JSON文件并添加到最终结果
        await self._finalize_evaluation_after_scoring(final_results, final_evaluation)

        logger.info("仲裁完成")
        return final_results

    def _serialize_votes(self, votes: List[ReviewerVote]) -> List[Dict[str, Any]]:
        """
        将ReviewerVote对象序列化为可JSON化的字典
        """
        serialized_votes = []
        for vote in votes:
            # 序列化每个AI的评测结果
            serialized_evaluations = {}
            for ai_id, evaluation in vote.ai_evaluations.items():
                serialized_evaluations[ai_id] = {
                    'ai_id': evaluation.ai_id,
                    'dimension_scores': evaluation.dimension_scores.to_dict(),
                    'final_score': evaluation.final_score,
                    'evaluation_text': evaluation.evaluation_text,
                    'success': evaluation.success,
                    'model_response_record': {
                        'model_name': evaluation.model_response_record.model_name if evaluation.model_response_record else None,
                        'internal_id': evaluation.model_response_record.internal_id if evaluation.model_response_record else None,
                        'input_content': evaluation.model_response_record.input_content if evaluation.model_response_record else None,
                        'response_content': evaluation.model_response_record.response_content if evaluation.model_response_record else None,
                        'is_correct': evaluation.model_response_record.is_correct if evaluation.model_response_record else None,
                        'response_timestamp': evaluation.model_response_record.response_timestamp if evaluation.model_response_record else None
                    } if evaluation.model_response_record else None
                }

            serialized_vote = {
                'reviewer_id': vote.reviewer_id,
                'ai_evaluations': serialized_evaluations,
                'confidence': vote.confidence,
                'model_response_judgments': vote.model_response_judgments
            }
            serialized_votes.append(serialized_vote)

        return serialized_votes

    async def _finalize_evaluation_before_scoring(self, votes: List[ReviewerVote]) -> None:
        """
        在计算最终得分之前统合评审结果，确定回答正确性

        Args:
            votes: 评审投票结果
        """
        logger.info("开始统合评审结果，确定回答正确性（在计算得分之前）")

        # 统合三个评审对每个模型回答正确性的判断
        for internal_id, display_name in self.model_id_mapping.items():
            judgments = []

            for vote in votes:
                if internal_id in vote.model_response_judgments:
                    judgments.append(vote.model_response_judgments[internal_id])

            if judgments:
                # 只要有一个True，结果就是True
                is_correct = all(judgments)  # 修改：从"一票否决"

                # 记录到回答记录器
                self.response_recorder.determine_correctness(internal_id, is_correct)

                logger.info(f"模型 {display_name} 回答正确性: {is_correct} (评审判断: {judgments})")

    async def _finalize_evaluation_after_scoring(self, final_results: Dict[str, Any],
                                                 final_evaluation: Dict[str, Any]) -> None:
        """
        在计算最终得分之后保存JSON文件并添加到最终结果

        Args:
            final_results: 最终评审结果
            final_evaluation: 最终评价结果
        """
        logger.info("保存模型回答记录到JSON文件")

        # 保存模型回答记录到JSON文件
        response_file = self.response_recorder.save_to_json()

        # 将回答记录文件路径添加到最终结果中
        final_results["model_responses_file"] = response_file
        final_results["model_responses"] = {
            internal_id: {
                "model_name": record.model_name,
                "input_content": record.input_content,
                "response_content": record.response_content,
                "is_correct": record.is_correct,
                "success": record.is_correct if record.is_correct is not None else False,
                "response_timestamp": record.response_timestamp
            }
            for internal_id, record in self.response_recorder.records.items()
        }

        # 将最终评价结果添加到最终结果中
        final_results["final_evaluation"] = final_evaluation

        logger.info(f"模型回答记录已保存: {response_file}")

    def _identify_debate_cases(self, dimension_stats: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        识别需要辩论的情况

        Args:
            dimension_stats: 维度统计信息

        Returns:
            需要辩论的情况列表
        """
        debate_cases = []

        for dimension, ai_data in dimension_stats.items():
            for model_name, stats in ai_data.items():
                # 如果最大分差超过阈值，需要辩论
                if stats["max_diff"] > self.debate_threshold:
                    debate_case = {
                        "dimension": dimension,
                        "model_id": model_name,  # 使用显示名称而非内部ID
                        "scores": stats["scores"],
                        "max_difference": stats["max_diff"],
                        "average_score": stats["average"]
                    }
                    debate_cases.append(debate_case)
                    logger.info(f"辩论案例: {model_name}的{dimension}维度分差{stats['max_diff']:.2f}超过阈值")

        return debate_cases

    async def _conduct_debate(self, original_votes: List[ReviewerVote],
                              debate_cases: List[Dict[str, Any]]) -> List[ReviewerVote]:
        """
        执行辩论流程

        Args:
            original_votes: 原始投票结果
            debate_cases: 辩论案例

        Returns:
            辩论后的投票结果
        """
        logger.info("开始辩论流程")

        # 这里可以实现更复杂的辩论逻辑
        # 目前简化为重新评审分差过大的项目
        debated_votes = []

        for i, vote in enumerate(original_votes):
            debated_vote = await self._reviewer_redebate(vote, debate_cases)
            debated_votes.append(debated_vote)

        logger.info("辩论流程完成")
        return debated_votes

    async def _reviewer_redebate(self, original_vote: ReviewerVote,
                                 debate_cases: List[Dict[str, Any]]) -> ReviewerVote:
        """
        单个评审智能体重新评审辩论案例

        Args:
            original_vote: 原始投票
            debate_cases: 辩论案例

        Returns:
            重新评审后的投票
        """
        # 这里可以添加具体的重新评审逻辑
        # 目前返回原始投票（实际应用中应该调用评审智能体的重新评审方法）
        return original_vote

    async def _generate_final_evaluation(self, votes: List[ReviewerVote], model_id_mapping: Dict[str, str]) -> Dict[
        str, Any]:
        """
        生成最终评价

        Args:
            votes: 投票结果
            model_id_mapping: 模型标识符映射

        Returns:
            最终评价结果
        """
        logger.info("生成最终评价")

        final_results = {}

        for internal_id, display_name in model_id_mapping.items():
            ai_results = []

            # 收集所有评审对该AI的评价
            for vote in votes:
                if internal_id in vote.ai_evaluations:
                    evaluation = vote.ai_evaluations[internal_id]
                    ai_results.append({
                        "reviewer_id": vote.reviewer_id,
                        "dimension_scores": evaluation.dimension_scores.to_dict(),
                        "final_score": evaluation.final_score,
                        "evaluation_text": evaluation.evaluation_text,
                        "success": evaluation.success  # 新增：正确性判断
                    })

            # 获取该模型的最终正确性判断（从回答记录器）
            is_correct = False
            if internal_id in self.response_recorder.records:
                is_correct = self.response_recorder.records[internal_id].is_correct or False

            # 计算平均分（应用惩罚：错误回答乘以0.3）
            if ai_results:
                # 对每个评审的评分应用惩罚
                penalized_final_scores = []
                penalized_dimension_avgs = {}
                dimension_names = ["memorization", "comprehension", "application", "analysis", "synthesis",
                                   "evaluation", "innovation"]

                for dimension in dimension_names:
                    dimension_scores = []
                    for result in ai_results:
                        # 如果回答错误，评分乘以0.3；正确则不变
                        penalty_factor = 0.3 if not is_correct else 1.0
                        penalized_score = result["dimension_scores"][dimension] * penalty_factor
                        dimension_scores.append(penalized_score)

                    penalized_dimension_avgs[dimension] = statistics.mean(dimension_scores) if dimension_scores else 0

                # 计算惩罚后的最终得分
                for result in ai_results:
                    # 创建临时的AIEvaluationResult来计算惩罚后的得分
                    temp_eval = AIEvaluationResult(
                        ai_id=internal_id,
                        dimension_scores=DimensionScore(**result["dimension_scores"]),
                        success=is_correct
                    )
                    # 计算惩罚后的最终得分
                    penalized_score = temp_eval.calculate_final_score(self.dimension_weights, apply_penalty=True)
                    penalized_final_scores.append(penalized_score)

                avg_final_score = statistics.mean(penalized_final_scores) if penalized_final_scores else 0

                final_results[display_name] = {  # 使用显示名称作为键
                    "average_final_score": avg_final_score,
                    "dimension_averages": penalized_dimension_avgs,  # 使用惩罚后的维度平均分
                    "reviewer_evaluations": ai_results,
                    "internal_id": internal_id,  # 保留内部ID用于追溯
                    "success": is_correct,  # 新增：正确性字段
                    "ranking": 0  # 将在后续计算排名
                }

        # 计算排名（基于惩罚后的最终得分）
        ranked_models = sorted(final_results.items(),
                               key=lambda x: x[1]["average_final_score"],
                               reverse=True)

        for rank, (model_name, _) in enumerate(ranked_models, 1):
            final_results[model_name]["ranking"] = rank

        logger.info("最终评价生成完成")
        return final_results


class ReviewerAgent:
    """评审智能体类"""

    def __init__(self, reviewer_id: str, base_api: BaseAPI):
        """
        初始化评审智能体

        Args:
            reviewer_id: 智能体ID
            base_api: 基础API实例（由外部提供）
        """
        self.reviewer_id = reviewer_id
        self.base_api = base_api
        self.dimension_prompts = self._load_dimension_prompts()

    def _load_dimension_prompts(self) -> Dict[str, str]:
        """加载七个维度的评分提示词"""
        return {
            "memorization": "识记：复述基本事实与概念的能力（基础能力）",
            "comprehension": "理解：解释概念、原理与过程的能力（内化能力）",
            "application": "应用：在标准场景下运用知识解决问题的能力（实践能力）",
            "analysis": "分析：分解复杂信息，辨析逻辑关系的能力（解构能力）",
            "synthesis": "综合：整合多种要素，构建新的解决方案或完整论述的能力（整合创造能力）",
            "evaluation": "评价：依据既定标准，对观点、方案或作品进行批判性判断与评估的能力（评判能力）",
            "innovation": "创新：提出新颖的假设、见解或探索性方案的能力（前沿探索能力）"
        }

    async def evaluate_ais(self, input_data: APIData, ai_count: int,
                           model_id_mapping: Dict[str, str] = None) -> ReviewerVote:
        """
        评测多个AI的解题能力

        Args:
            input_data: 输入数据（APIData对象）
            ai_count: AI数量
            model_id_mapping: 模型标识符映射

        Returns:
            评审投票结果
        """
        logger.info(f"评审智能体 {self.reviewer_id} 开始评测 {ai_count} 个AI")

        # 构建评审提示词
        evaluation_prompt = self._build_evaluation_prompt(ai_count, model_id_mapping)

        # 创建增强的数据对象，保留原始图片数据
        enhanced_data = type(input_data)()  # 创建相同类型的对象
        enhanced_data.text = evaluation_prompt + "\n\n" + input_data.text

        # 复制图片相关属性
        if hasattr(input_data, 'image_path'):
            enhanced_data.image_path = input_data.image_path
        if hasattr(input_data, 'image_data'):
            enhanced_data.image_data = input_data.image_data
        if hasattr(input_data, 'image_base64'):
            enhanced_data.image_base64 = input_data.image_base64

        try:
            # 调用API进行评测（使用处理类型0：不做处理）
            response = await self.base_api.process(enhanced_data, processing_type=0)

            if response.success and response.content:
                # 解析API返回的评分结果和正确性判断
                ai_evaluations, response_judgments = self._parse_evaluation_results(
                    response.content, ai_count, model_id_mapping)
                return ReviewerVote(
                    reviewer_id=self.reviewer_id,
                    ai_evaluations=ai_evaluations,
                    model_response_judgments=response_judgments  # 新增：回答正确性判断
                )
            else:
                logger.error(f"评审智能体 {self.reviewer_id} API调用失败: {response.error_message}")
                return self._generate_default_vote(ai_count, model_id_mapping)

        except Exception as e:
            logger.error(f"评审智能体 {self.reviewer_id} 评测失败: {str(e)}")
            return self._generate_default_vote(ai_count, model_id_mapping)

    def _build_evaluation_prompt(self, ai_count: int, model_id_mapping: Dict[str, str] = None) -> str:
        """构建评测提示词"""
        dimension_descriptions = "\n".join([f"{i + 1}. {dim}: {desc}"
                                            for i, (dim, desc) in enumerate(self.dimension_prompts.items())])

        # 构建模型列表描述
        if model_id_mapping:
            model_list = "\n".join([f"{internal_id}: {display_name}"
                                    for internal_id, display_name in model_id_mapping.items()])
            model_header = f"以下是要评测的{ai_count}个AI模型：\n{model_list}"
        else:
            model_header = f"请对{ai_count}个AI的解题能力进行七个维度评分"

        prompt = f"""
{model_header}

七个维度的评分取自所有模型相互比较的结果,而不是选自一个绝对标准。
七个维度评分标准（每个维度满分100分）：
{dimension_descriptions}

评分要求：
1. 请严格按照七个维度分别评分
2. 每个AI的输出格式为：AI编号|识记分数|理解分数|应用分数|分析分数|综合分数|评价分数|创新分数|是否正确(True/False)|简要评价
3. 分数范围为0-100的整数
4. 是否正确字段请填写True或False
5. 评价要客观公正，且不超过50个字

请严格按以下格式返回结果：（不能有其他额外的内容）（你不需要说其他任何内容，你返回的内容有且只能有下面的格式！！！，请你严格，严格遵守！）（一共8个AI）
AI1|85|90|88|92|75|80|82|True|解题思路清晰，逻辑严密
AI2|85|90|88|92|75|80|82|True|解题思路清晰，逻辑严密
AI3|85|90|88|92|75|80|82|True|解题思路清晰，逻辑严密
AI4|85|90|88|92|75|80|82|True|解题思路清晰，逻辑严密
AI5|85|90|88|92|75|80|82|True|解题思路清晰，逻辑严密
AI6|85|90|88|92|75|80|82|True|解题思路清晰，逻辑严密
AI7|85|90|88|92|75|80|82|True|解题思路清晰，逻辑严密
AI8|85|90|88|92|75|80|82|True|解题思路清晰，逻辑严密
        """
        return prompt

    def _parse_evaluation_results(self, content: str, ai_count: int, model_id_mapping: Dict[str, str] = None) -> tuple:
        """
        增强版解析函数 - 支持多种格式匹配
        使用多级回退机制提高解析鲁棒性
        """
        evaluations = {}
        response_judgments = {}

        # 定义多种匹配模式，按优先级排序
        patterns = [
            # 模式1: 标准竖线分隔格式 AI1|85|90|88|92|75|80|82|True|评价内容
            r'AI(\d+)\s*[|：:]\s*(\d+)\s*[|：:]\s*(\d+)\s*[|：:]\s*(\d+)\s*[|：:]\s*(\d+)\s*[|：:]\s*(\d+)\s*[|：:]\s*(\d+)\s*[|：:]\s*(\d+)\s*[|：:]\s*(True|False|正确|错误|√|×|对|错)\s*[|：:]\s*(.+)',

            # 模式2: 逗号分隔格式 AI1,85,90,88,92,75,80,82,True,评价内容
            r'AI(\d+)\s*[,，]\s*(\d+)\s*[,，]\s*(\d+)\s*[,，]\s*(\d+)\s*[,，]\s*(\d+)\s*[,，]\s*(\d+)\s*[,，]\s*(\d+)\s*[,，]\s*(\d+)\s*[,，]\s*(True|False|正确|错误|√|×|对|错)\s*[,，]\s*(.+)',

            # 模式3: 空格分隔格式 AI1 85 90 88 92 75 80 82 True 评价内容
            r'AI(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(True|False|正确|错误|√|×|对|错)\s+(.+)',

            # 模式4: 带维度名称的格式 AI1-识记:85 理解:90 应用:88 分析:92 综合:75 评价:80 创新:82 正确:True 评价:内容
            r'AI(\d+)[^\d]*识记[^\d]*(\d+)[^\d]*理解[^\d]*(\d+)[^\d]*应用[^\d]*(\d+)[^\d]*分析[^\d]*(\d+)[^\d]*综合[^\d]*(\d+)[^\d]*评价[^\d]*(\d+)[^\d]*创新[^\d]*(\d+)[^\d]*(True|False|正确|错误|√|×|对|错)[^\d]*(.+)',

            # 模式5: JSON格式 { "ai_id": "AI1", "scores": [85,90,88,92,75,80,82], "correct": true, "comment": "评价" }
            r'["\']?ai_?id["\']?\s*[:=]\s*["\']?AI(\d+)["\']?[^}]*["\']?scores?["\']?\s*[:=]\s*\[([\d\s,]+)\][^}]*["\']?correct["\']?\s*[:=]\s*(true|false|True|False)[^}]*["\']?comment["\']?\s*[:=]\s*["\']?([^"\']+)["\']?',
        ]

        lines = content.strip().split('\n')
        successful_matches = 0

        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line or len(line) < 10:  # 跳过空行和过短的行
                continue

            matched = False
            match_details = None

            # 尝试各种匹配模式
            for pattern_idx, pattern in enumerate(patterns):
                try:
                    match = re.search(pattern, line, re.IGNORECASE)
                    if match:
                        match_details = self._process_match_result(match, pattern_idx, line)
                        if match_details:
                            ai_id, scores, is_correct, evaluation_text = match_details

                            # 验证AI ID有效性
                            if model_id_mapping and ai_id not in model_id_mapping:
                                logger.warning(f"解析到无效的AI标识符: {ai_id}，跳过该行")
                                continue

                            # 验证分数范围
                            if all(0 <= score <= 100 for score in scores):
                                dimension_scores = DimensionScore(*scores)
                                evaluation = AIEvaluationResult(
                                    ai_id=ai_id,
                                    dimension_scores=dimension_scores,
                                    evaluation_text=evaluation_text,
                                    success=is_correct
                                )
                                evaluations[ai_id] = evaluation
                                response_judgments[ai_id] = is_correct
                                successful_matches += 1
                                matched = True
                                logger.info(f"使用模式{pattern_idx + 1}成功解析 {ai_id} 的评测结果")
                                break  # 匹配成功，跳出模式循环
                            else:
                                logger.warning(f"分数超出范围: {scores}")

                except Exception as e:
                    logger.debug(f"模式{pattern_idx + 1}匹配失败: {e}")
                    continue

            if not matched:
                logger.debug(f"第{line_num}行无法匹配任何模式: {line[:100]}...")

        # 后处理：检查是否所有AI都有结果，尝试从内容中提取缺失的AI
        if successful_matches < ai_count:
            self._try_extract_missing_ais(content, evaluations, response_judgments,
                                          ai_count, model_id_mapping)

        # 确保所有AI都有评测结果
        expected_ai_ids = list(model_id_mapping.keys()) if model_id_mapping else [f"AI{i + 1}" for i in range(ai_count)]

        missing_ais = set(expected_ai_ids) - set(evaluations.keys())
        for ai_id in missing_ais:
            logger.warning(f"AI {ai_id} 的评测结果缺失，使用默认评价")
            evaluations[ai_id] = self._create_default_evaluation(ai_id)
            response_judgments[ai_id] = False  # 解析失败默认判断为错误

        logger.info(f"成功解析 {len(evaluations)}/{len(expected_ai_ids)} 个AI的评测结果")
        return evaluations, response_judgments

    def _process_match_result(self, match, pattern_idx: int, original_line: str) -> tuple:
        """
        处理匹配结果，提取AI ID、分数、正确性和评价文本
        """
        try:
            if pattern_idx in [0, 1, 2]:  # 前三种标准格式
                ai_number = match.group(1)
                ai_id = f"AI{ai_number}"

                # 提取7个维度分数
                scores = []
                for i in range(2, 9):  # 7个分数
                    score_str = match.group(i).strip()
                    scores.append(float(score_str))

                # 处理正确性字段
                correctness_str = match.group(9).strip().lower()
                is_correct = self._parse_correctness(correctness_str)

                # 评价文本
                evaluation_text = match.group(10).strip() if len(match.groups()) >= 10 else "无评价"

            elif pattern_idx == 3:  # 带维度名称的格式
                ai_number = match.group(1)
                ai_id = f"AI{ai_number}"

                scores = []
                for i in range(2, 9):  # 7个分数
                    score_str = match.group(i).strip()
                    scores.append(float(score_str))

                correctness_str = match.group(9).strip().lower()
                is_correct = self._parse_correctness(correctness_str)
                evaluation_text = match.group(10).strip() if len(match.groups()) >= 10 else "无评价"

            elif pattern_idx == 4:  # JSON格式
                ai_number = match.group(1)
                ai_id = f"AI{ai_number}"

                # 解析分数数组
                scores_str = match.group(2)
                scores = [float(s.strip()) for s in re.findall(r'\d+', scores_str)]

                # 确保有7个分数，不足用0填充，多余截断
                if len(scores) < 7:
                    scores.extend([0] * (7 - len(scores)))
                elif len(scores) > 7:
                    scores = scores[:7]

                correctness_str = match.group(3).strip().lower()
                is_correct = self._parse_correctness(correctness_str)
                evaluation_text = match.group(4).strip() if len(match.groups()) >= 4 else "无评价"

            else:
                return None

            return ai_id, scores, is_correct, evaluation_text[:200]  # 限制评价文本长度

        except Exception as e:
            logger.error(f"处理匹配结果时出错: {e}, 行: {original_line[:100]}")
            return None

    def _parse_correctness(self, correctness_str: str) -> bool:
        """
        解析正确性字符串，支持多种表示方式
        """
        true_indicators = ['true', '正确', '√', '对', 'yes', '是', '1', 'ok', 'pass']
        false_indicators = ['false', '错误', '×', '错', 'no', '否', '0', 'fail']

        correctness_str_lower = correctness_str.lower().strip()

        if correctness_str_lower in true_indicators:
            return True
        elif correctness_str_lower in false_indicators:
            return False
        else:
            # 尝试从字符串中提取布尔值
            if re.search(r'true|正确|√|对|yes|是', correctness_str_lower):
                return True
            elif re.search(r'false|错误|×|错|no|否', correctness_str_lower):
                return False
            else:
                # 默认值或尝试数字解析
                try:
                    num = float(correctness_str_lower)
                    return num > 0
                except:
                    logger.warning(f"无法解析正确性字段: {correctness_str}, 默认设为False")
                    return False

    def _try_extract_missing_ais(self, content: str, evaluations: dict, response_judgments: dict,
                                 ai_count: int, model_id_mapping: Dict[str, str] = None):
        """
        尝试从内容中提取缺失的AI评测结果
        使用更宽松的搜索模式
        """
        expected_ai_ids = list(model_id_mapping.keys()) if model_id_mapping else [f"AI{i + 1}" for i in range(ai_count)]
        missing_ais = set(expected_ai_ids) - set(evaluations.keys())

        if not missing_ais:
            return

        logger.info(f"尝试补充提取缺失的AI: {missing_ais}")

        for ai_id in missing_ais:
            # 在内容中搜索该AI的相关信息
            ai_pattern = re.escape(
                ai_id) + r'[^\d]*?(\d+)[^\d]*?(\d+)[^\d]*?(\d+)[^\d]*?(\d+)[^\d]*?(\d+)[^\d]*?(\d+)[^\d]*?(\d+)'
            match = re.search(ai_pattern, content, re.IGNORECASE)

            if match and len(match.groups()) >= 7:
                try:
                    scores = [float(match.group(i)) for i in range(1, 8)]
                    if all(0 <= score <= 100 for score in scores):
                        # 尝试查找正确性
                        correctness_match = re.search(re.escape(ai_id) + r'.*?(True|False|正确|错误)', content,
                                                      re.IGNORECASE)
                        is_correct = False  # 默认值设为False，更保守
                        if correctness_match:
                            is_correct = self._parse_correctness(correctness_match.group(1))

                        dimension_scores = DimensionScore(*scores)
                        evaluation = AIEvaluationResult(
                            ai_id=ai_id,
                            dimension_scores=dimension_scores,
                            evaluation_text="从内容中提取的评分",
                            success=is_correct
                        )
                        evaluations[ai_id] = evaluation
                        response_judgments[ai_id] = is_correct
                        logger.info(f"成功从内容中提取 {ai_id} 的评分")

                except Exception as e:
                    logger.debug(f"从内容中提取 {ai_id} 评分失败: {e}")

    def _create_default_evaluation(self, ai_id: str) -> AIEvaluationResult:
        """创建默认评测结果"""
        default_scores = DimensionScore(60, 60, 60, 60, 60, 60, 60)
        return AIEvaluationResult(
            ai_id=ai_id,
            dimension_scores=default_scores,
            evaluation_text="默认评价",
            success=False  # 默认判断为错误，更保守
        )

    def _generate_default_vote(self, ai_count: int, model_id_mapping: Dict[str, str] = None) -> ReviewerVote:
        """
        生成默认投票结果（API失败时使用）

        Args:
            ai_count: AI数量
            model_id_mapping: 模型标识符映射（用于确定要生成的AI评测结果）

        Returns:
            默认投票结果
        """
        evaluations = {}
        response_judgments = {}

        # 确定要生成评测结果的AI ID列表
        if model_id_mapping:
            ai_ids = list(model_id_mapping.keys())
        else:
            ai_ids = [f"AI{i + 1}" for i in range(ai_count)]

        for ai_id in ai_ids:
            evaluations[ai_id] = self._create_default_evaluation(ai_id)
            response_judgments[ai_id] = False  # 默认判断为错误

        return ReviewerVote(
            reviewer_id=self.reviewer_id,
            ai_evaluations=evaluations,
            model_response_judgments=response_judgments,
            confidence=0.5  # 低置信度
        )


class ConcreteAPI(BaseAPI):
    """
    BaseAPI的具体实现类，支持多模态输入和异步处理
    适配新的BaseAPI接口，使用模型名称而非URL
    修复了认证问题，避免401错误
    """

    def __init__(self, name: str, api_key: str, model_name: str = "", processing_type: int = 0):
        """
        初始化ConcreteAPI，适配新的BaseAPI接口

        Args:
            name: API名称
            api_key: API密钥
            model_name: 模型名称（替代原来的base_url）
            processing_type: 处理类型
        """
        # 调用父类初始化，传递模型名称而非URL
        super().__init__(name, api_key, model_name, processing_type)

    async def call_api(self, processed_data: APIData, **kwargs) -> APIResponse:
        """
        实现BaseAPI的抽象方法，执行具体的API调用
        适配新的BaseAPI接口，使用正确的认证方式避免401错误

        Args:
            processed_data: 预处理后的数据
            **kwargs: 其他参数

        Returns:
            APIResponse: API响应结果
        """
        await self.ensure_session()

        try:
            # 构建请求数据 - 根据是否有图片选择不同的格式
            if processed_data.has_image() and processed_data.image_base64:
                # 多模态请求：文本+图片
                payload = {
                    "model": self.model_name or "deepseek-vl2",  # 使用模型名称参数
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
                                        "url": f"data:image/jpeg;base64,{processed_data.image_base64}",
                                        "detail": kwargs.get("image_detail", "auto")
                                    }
                                }
                            ]
                        }
                    ],
                    "max_tokens": kwargs.get("max_tokens", 2000),
                    "temperature": kwargs.get("temperature", 0.3),  # 正确的写法
                    "top_p": kwargs.get("top_p", 0.9)
                }
                logger.info(f"构建多模态请求（文本+图片），模型: {self.model_name}")
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
                    "max_tokens": kwargs.get("max_tokens", 900000),
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


# 使用示例
async def main():
    """主函数示例"""
    # 初始化三个评审智能体（使用修复后的ConcreteAPI）
    reviewer1 = ReviewerAgent("reviewer_1", ConcreteAPI("api1", "key1", "model-1"))
    reviewer2 = ReviewerAgent("reviewer_2", ConcreteAPI("api2", "key2", "model-2"))
    reviewer3 = ReviewerAgent("reviewer_3", ConcreteAPI("api3", "key3", "model-3"))

    # 初始化评分标准和历史数据
    scoring_criteria = [
        {"content": "识记维度：准确复述基本概念和事实"},
        {"content": "理解维度：能够解释原理和过程"}
    ]

    historical_data = [
        {"content": "历史评测1：模型A在分析维度表现优秀"},
        {"content": "历史评测2：模型B在创新维度有待提升"}
    ]

    # 初始化评审系统（开启辩论模式）
    review_system = SmartReviewSystem(
        debate_mode=DebateMode.ENABLED,
        scoring_criteria=scoring_criteria,
        historical_data=historical_data
    )
    review_system.add_reviewer(reviewer1)
    review_system.add_reviewer(reviewer2)
    review_system.add_reviewer(reviewer3)

    # 准备输入数据（使用APIData类）
    input_data = type('APIData', (), {})()  # 模拟APIData对象
    input_data.text = "8个AI的解题思路和过程文本..."
    input_data.image_path = "题目和答案图片路径.png"

    # 定义要被评测的8个模型的具体名称
    models_to_evaluate = [
        "Qwen2.5-72B-Instruct",
        "gpt-4o-mini",
        "claude-3-5-sonnet-20241022",
        "Llama-3.1-70B-Instruct",
        "ERNIE-4.0-8B-0324",
        "Spark-3.0-16B",
        "GLM-4-9B-Chat",
        "Baichuan2-53B-Chat"
    ]

    # 在实际使用中，您需要先获取8个测试模型的回答，然后记录它们
    # 这里模拟记录模型回答的过程
    sample_responses = [
        "模型1的详细解题过程...",
        "模型2的详细解题过程...",
        # ... 其他模型的回答
    ]

    for i, model_name in enumerate(models_to_evaluate):
        internal_id = f"AI{i + 1}"
        # 在实际应用中，这里应该是模型的实际回答内容
        response_content = sample_responses[i] if i < len(sample_responses) else f"{model_name}的模拟回答内容"

        review_system.record_model_response(
            internal_id=internal_id,
            model_name=model_name,
            input_content=input_data.text,
            response_content=response_content
        )

    # 执行评审流程
    results = await review_system.process_evaluation(input_data, ai_count=8, model_names=models_to_evaluate)

    # 输出结果
    print("评审完成！")
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
