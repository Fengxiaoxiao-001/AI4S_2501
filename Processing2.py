import asyncio
import json
import os
import time
from typing import Dict, List, Optional
from memory import EnhancedMemorySystem
from InputAndOutput_Layer import AsyncAPIScheduler, ExampleChatAPI, APIData
from Evaluation_Layer import ReviewerAgent, ConcreteAPI, SmartReviewSystem, DebateMode

# 定义要被评测的模型
models_to_evaluate = {
    "Qwen3-VL-30B-A3B-Thinking": ["qwen3-vl-30b-a3b-thinking","bce-v3/ALTAK-T3PDoDqawgltRn3529ACk/d2a58ee6fdf9359d0673e8b7872a52cc86613351"],
    "Qianfan-EngCard-VL": ["qianfan-engcard-vl","bce-v3/ALTAK-dtxnhO7rGseaHE9cnednN/f2c026193d2601cea310b5806f260d7ab1d79a2b"],
    "Qianfan-QI-VL": ["qianfan-qi-vl", "bce-v3/ALTAK-1ZbrHo15FfIZ7fwVEBbUq/c749613848f2cc502c7b91253f8b72fcf4a94246"],
    "Llama-4-Maverick-17B-128E-Instruct": ["llama-4-maverick-17b-128e-instruct", "bce-v3/ALTAK-Yv9zLhNLHph36dwGQAXDM/28e8fe6bb2572e12787c952821f096e7d24c66a0"],
    "Llama-4-Scout-17B-16E-Instruct": ["llama-4-scout-17b-16e-instruct", "bce-v3/ALTAK-y87q5FLcH8mC4q0mTEs50/ae0bd14acd7c16f28335fb3ec6b1f328ee6cea42"],
    "InternVL3-38B": ["internvl3-38b", "bce-v3/ALTAK-8C6KZKkeZcddWcAvsCeTp/60a44930d12f731e186685b5403c4a2cb3d76191"],
    "GLM-4.5V": ["glm-4.5v", "bce-v3/ALTAK-opnmfSkJMGShEKBvNm33E/50c573400a0877a615dbb9242a9ebffb446813cf"],
    "Qianfan-MultiPicOCR": ["qianfan-multipicocr", "bce-v3/ALTAK-mCW5Olq0sFzl80kF8ng4l/5d7f40c0763585a174a8793a7815cd2a5fae069a"]
}

base_path = "Data/0/PSQ/StaAndPro/"
# 初始化增强记忆系统
memory_system = EnhancedMemorySystem(base_path + "answer.json")

# 创建调度器
scheduler = AsyncAPIScheduler()

# 修正API配置 - 每个配置项应为4个参数：[API类, base_url, api_key, processing_type]
api_configs = [
    [ExampleChatAPI, *models_to_evaluate["Qwen3-VL-30B-A3B-Thinking"], 0],
    [ExampleChatAPI, *models_to_evaluate["Qianfan-EngCard-VL"], 0],
    [ExampleChatAPI, *models_to_evaluate["Qianfan-QI-VL"], 0],
    [ExampleChatAPI, *models_to_evaluate["Llama-4-Maverick-17B-128E-Instruct"], 0],
    [ExampleChatAPI, *models_to_evaluate["Llama-4-Scout-17B-16E-Instruct"], 0],
    [ExampleChatAPI, *models_to_evaluate["InternVL3-38B"], 0],
    [ExampleChatAPI, *models_to_evaluate["GLM-4.5V"], 0],
    [ExampleChatAPI, *models_to_evaluate["Qianfan-MultiPicOCR"], 0]
]

# 创建模型名称映射表，用于后续识别和记录
model_name_mapping = {
    0: "Qwen3-VL-30B-A3B-Thinking",
    1: "Qianfan-EngCard-VL",
    2: "Qianfan-QI-VL",
    3: "Llama-4-Maverick-17B-128E-Instruct",
    4: "Llama-4-Scout-17B-16E-Instruct",
    5: "InternVL3-38B",
    6: "GLM-4.5V",
    7: "Qianfan-MultiPicOCR"
}

# 常量定义
GROUP_NAME = "EasyAqGeom"
JSON_FILENAME = base_path + "results.json"
EVALUATION_FILENAME = base_path + "evaluation.json"
TEST_IMAGE_PATH = r"E:\Preprocessing\AI4S_2\Math\Math_train\0\PSQ\StaAndPro\1.png"  # 修改为常量，避免使用global

# 创建答案数据
Answer_Data: APIData = APIData(
    text="《注：图片内容为题目标准答案》。你是一个数学解题评分专家。作为一个数学解题能力评分专家，你需要对输入的8个学生解题过程，按七个维度评分（每个维度满分100分）。每个维度内，比较8名学生，分为两个得分，第一个为相对得分(表现最优者得100分，其余按相对表现给分。),第二个为绝对评分，需要你逐个并单独地对某个学生的回答进行绝对客观的评价，不参考其他学生的回答（满分100分）。每个维度的总得分 = （每个维度的相对评分*0.3）+（每个维度的绝对评分*0.7），按照这个规则生成七维度评分，无需额外的回答解释。",
    image_path= r"E:\Preprocessing\AI4S_2\Math\Math_test\0\PSQ\StaAndPro\1.png"
)

# 注册API组
scheduler.register_apis(GROUP_NAME, api_configs)


def get_model_name(response, index: int) -> str:
    """根据索引获取模型名称

    Args:
        response: API响应对象
        index: 模型索引

    Returns:
        str: 模型名称
    """
    # 使用预定义的模型名称映射
    if index in model_name_mapping:
        return model_name_mapping[index]

    # 备用方法：基于api_name的映射
    model_mapping_by_api_name = {
        "multi_modal_group_api_0": "Qwen3-VL-30B-A3B-Thinking",
        "multi_modal_group_api_1": "Qianfan-EngCard-VL",
        "multi_modal_group_api_2": "Qianfan-QI-VL",
        "multi_modal_group_api_3": "Llama-4-Maverick-17B-128E-Instruct",
        "multi_modal_group_api_4": "Llama-4-Scout-17B-16E-Instruct",
        "multi_modal_group_api_5": "InternVL3-38B",
        "multi_modal_group_api_6": "GLM-4.5V",
        "multi_modal_group_api_7": "Qianfan-MultiPicOCR"
    }

    return model_mapping_by_api_name.get(response.api_name, f"unknown_model_{index}")


def print_api_result(model_name: str, response) -> None:
    """打印API结果信息

    Args:
        model_name: 模型名称
        response: API响应对象
    """
    status = "成功" if response.success else "失败"
    processing_type = response.processing_type

    if response.success:
        print(f"  {model_name} (处理类型{processing_type}, {status}, 耗时{response.response_time:.2f}s)")
        # 可选：打印部分内容预览
        if response.content and len(response.content) > 100:
            print(f"    内容预览: {response.content[:100]}...")
    else:
        print(f"  {model_name} (处理类型{processing_type}, {status}): {response.error_message}")


def save_json_results(json_results: Dict, filename: str) -> None:
    """保存JSON结果到文件

    Args:
        json_results: 要保存的JSON数据
        filename: 文件名
    """
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, ensure_ascii=False, indent=2)

        print(f"=== API结果已保存 ===")
        print(f"JSON文件: {filename}")
    except Exception as e:
        print(f"保存JSON结果失败: {e}")


async def initialize_review_system() -> Optional[SmartReviewSystem]:
    """初始化评审系统

    Returns:
        Optional[SmartReviewSystem]: 初始化成功的评审系统，失败返回None
    """
    print("\n=== 初始化评审系统 ===")

    try:
        # 初始化评分标准和历史数据
        scoring_criteria = [
            {"content": "识记维度：准确性（50分）和完整性（50分）"},
            {"content": "理解维度：清晰度（40分）、深度（40分）和转换能力（20分）"},
            {"content": "应用维度：方法合理性（40分）、过程正确性（40分）和结果准确性（20分）"},
            {"content": "分析维度：条理性（40分）、逻辑关系识别（40分）和洞察力（20分）"},
            {"content": "综合维度：整合有效性（50分）和新颖性（50分）"},
            {"content": "评价维度：标准明确性（40分）、论证充分性（40分）和判断精准性（20分）"},
            {"content": "创新维度：新颖性（50分）和合理性与价值（50分）"}
        ]

        historical_data = []

        # 初始化三个评审智能体（使用具体的API实现类）
        # 注意：这里需要确保ConcreteAPI类已正确定义
        reviewer1 = ReviewerAgent("Qwen3-VL-30B-A3B-Thinking",
                                  ConcreteAPI("qwen3-vl-30b-a3b-thinking", "bce-v3/ALTAK-T3PDoDqawgltRn3529ACk/d2a58ee6fdf9359d0673e8b7872a52cc86613351"))
        reviewer2 = ReviewerAgent("Qianfan-EngCard-VL",
                                  ConcreteAPI("qianfan-engcard-vl", "bce-v3/ALTAK-NYfnCwhLTgLFXYkSWoe65/215efc6fd8f4c176bb0305acf0eec111a84f5b82"))
        reviewer3 = ReviewerAgent("Qianfan-QI-VL",
                                  ConcreteAPI("qianfan-qi-vl", "bce-v3/ALTAK-E57sPHSjhrLEXrInc7D7z/f12a8baf2fae413311c8ca93ea4a0edd261dc77e"))

        # 初始化评审系统（开启辩论模式）
        review_system = SmartReviewSystem(
            debate_mode=DebateMode.ENABLED,
            scoring_criteria=scoring_criteria,
            historical_data=historical_data
        )
        review_system.add_reviewer(reviewer1)
        review_system.add_reviewer(reviewer2)
        review_system.add_reviewer(reviewer3)

        print("评审系统初始化完成")
        return review_system

    except Exception as e:
        print(f"评审系统初始化失败: {e}")
        return None


async def record_model_responses(review_system, results: Dict, input_data: APIData) -> List[str]:
    """记录模型回答到评审系统

    Args:
        review_system: 评审系统实例
        results: API调用结果
        input_data: 输入数据

    Returns:
        list: 模型名称列表
    """
    print("\n=== 记录模型回答到评审系统 ===")

    model_names = []

    try:
        for group_name, response_list in results.items():
            print(f"处理组: {group_name}")

            for i, response in enumerate(response_list):
                if i >= len(model_name_mapping):
                    print(f"警告: 响应数量超过模型映射表大小，跳过第{i}个响应")
                    break

                model_name = get_model_name(response, i)
                internal_id = f"AI{i + 1}"

                # 记录模型回答
                if review_system:
                    review_system.record_model_response(
                        internal_id=internal_id,
                        model_name=model_name,
                        input_content=input_data.text,
                        response_content=response.content if response.success else "API调用失败"
                    )

                model_names.append(model_name)
                print(f"  已记录: {model_name} ({internal_id}) - 成功: {response.success}")

        return model_names

    except Exception as e:
        print(f"记录模型回答失败: {e}")
        return []


async def run_smart_review(review_system, input_data: APIData, model_names: List[str]) -> Optional[Dict]:
    """运行智能评审流程

    Args:
        review_system: 评审系统实例
        input_data: 输入数据
        model_names: 模型名称列表

    Returns:
        dict: 评审结果，失败返回None
    """
    print("\n=== 开始智能评审流程 ===")

    try:
        if not review_system:
            print("错误: 评审系统未正确初始化")
            return None

        if not model_names:
            print("错误: 无有效的模型名称列表")
            return None

        # 执行评审流程
        review_results = await review_system.process_evaluation(
            input_data=input_data,
            ai_count=len(model_names),
            model_names=model_names
        )

        print("智能评审完成！")
        return review_results

    except Exception as e:
        print(f"智能评审过程中发生错误: {e}")
        return None


async def print_review_summary(review_results: Dict) -> None:
    """打印评审摘要

    Args:
        review_results: 评审结果字典
    """
    if not review_results or "final_evaluation" not in review_results:
        print("无有效评审结果")
        return

    final_eval = review_results["final_evaluation"]

    print("\n=== 智能评审摘要 ===")
    print("模型排名结果:")

    try:
        # 按排名排序
        ranked_models = sorted(
            [(name, data) for name, data in final_eval.items() if isinstance(data, dict)],
            key=lambda x: x[1].get("ranking", 999)
        )

        for model_name, result in ranked_models:
            score = result.get("average_final_score", 0)
            rank = result.get("ranking", "未排名")
            success = result.get("success", False)
            status = "正确" if success else "错误"

            print(f"第{rank}名: {model_name} - 得分: {score:.2f} - 回答: {status}")

    except Exception as e:
        print(f"打印评审摘要时发生错误: {e}")


async def call_apis_and_process_results() -> Optional[Dict]:
    """调用API并处理结果

    Returns:
        Dict: API调用结果
    """
    print("\n=== 测试: 文本+图片多模态处理 ===")

    try:
        # 创建测试数据
        image_text_data = APIData(
            text="请回答所提供图片中的题目。若题目是选择题，输出格式为：【答案】具体选项；【答案内容】简要说明；【解题思路】步骤分析。若题目是大题，输出格式为：【题目分解】关键部分拆解，你认为可以按几步解决；【解题思路】你的方法论，你的思考；【解题过程】详细推导;【最终的答案】答案结果。回答需简洁、准确，避免冗余。（要求回答尽量简洁，干练，不超过200字）",
            image_path=TEST_IMAGE_PATH if TEST_IMAGE_PATH and os.path.exists(TEST_IMAGE_PATH) else ""
        )

        # 调用API获取模型回答
        results = await scheduler.schedule_single_group(GROUP_NAME, image_text_data)

        if not results or GROUP_NAME not in results:
            print("错误: 未获取到有效的API响应")
            return None

        return results

    except Exception as e:
        print(f"调用API和处理结果时发生错误: {e}")
        return None


async def main() -> Optional[Dict]:
    """主函数 - 整合模型调用和智能评审

    Returns:
        Dict: 包含API结果和评审结果的字典
    """
    start_time = time.time()

    try:
        # 阶段1: 获取模型回答
        results = await call_apis_and_process_results()
        if not results:
            print("阶段1失败: 无法获取模型回答")
            return None

        # 准备存储结果的字典结构
        json_results = {
            GROUP_NAME: {
                "input_text": "",
                "image_path": TEST_IMAGE_PATH,
                "models": {},
                "timestamp": time.time()
            }
        }

        # 处理每个API的响应结果并存储
        response_list = results.get(GROUP_NAME, [])
        input_data = None

        for i, response in enumerate(response_list):
            model_name = get_model_name(response, i)

            # 构建模型结果字典
            model_result = {
                "success": response.success,
                "content": response.content if response.success else None,
                "error_message": response.error_message if not response.success else None,
                "response_time": response.response_time,
                "api_name": response.api_name,
                "processing_type": response.processing_type,
            }

            # 添加到结果字典中
            json_results[GROUP_NAME]["models"][model_name] = model_result

            # 存储到记忆系统
            if i == 0 and response_list:  # 使用第一个响应的输入数据
                input_data = APIData(
                    text=response_list[0].get("input_text", "") if hasattr(response_list[0], 'get') else "",
                    image_path=TEST_IMAGE_PATH
                )

            memory_system.store_api_result(GROUP_NAME, model_name, response, input_data)

            # 输出简要信息
            print_api_result(model_name, response)

        # 保存原始API结果到JSON文件
        save_json_results(json_results, JSON_FILENAME)

        # 阶段2: 初始化评审系统
        review_system = await initialize_review_system()
        if not review_system:
            print("阶段2失败: 评审系统初始化失败")
            return {"api_results": json_results, "review_results": None}

        # 阶段3: 记录模型回答到评审系统
        model_names = await record_model_responses(review_system, results, input_data or APIData(text=""))
        if not model_names:
            print("阶段3失败: 无法记录模型回答")
            return {"api_results": json_results, "review_results": None}

        # 阶段4: 运行智能评审
        review_results = await run_smart_review(review_system, Answer_Data, model_names)

        # 阶段5: 保存评审结果
        if review_results:
            try:
                with open(EVALUATION_FILENAME, 'w', encoding='utf-8') as f:
                    json.dump(review_results, f, ensure_ascii=False, indent=2)
                print(f"\n=== 评审结果已保存 ===")
                print(f"文件: {EVALUATION_FILENAME}")

                # 输出简要评审摘要
                await print_review_summary(review_results)
            except Exception as e:
                print(f"保存评审结果失败: {e}")

        total_time = time.time() - start_time
        print(f"\n=== 整个流程完成 ===")
        print(f"总耗时: {total_time:.2f}秒")

        return {
            "api_results": json_results,
            "review_results": review_results
        }

    except Exception as e:
        print(f"处理过程中发生错误: {e}")
        return None


if __name__ == "__main__":
    # 使用asyncio.run()正确运行异步函数[1,2](@ref)
    try:
        # 确保事件循环正确关闭[5](@ref)
        results = asyncio.run(main())
        if results:
            print("程序执行成功")
        else:
            print("程序执行完成，但部分阶段可能失败")
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"程序执行失败: {e}")