import asyncio
import json
import os
import time
from typing import Dict, Optional
from InputAndOutput_Layer import AsyncAPIScheduler, ExampleChatAPI, APIData

# 定义要被调用的模型
models_to_evaluate = {
    "Qwen2.5-VL-32B-Instruct": ["qwen2.5-vl-32b-instruct","bce-v3/ALTAK-XbGDRaOfJTlbDGnrtZAsJ/6f01dcc68f9caf7000652a2a0dbeef62b41d8a90"],
    "DeepSeek-VL2-Small": ["deepseek-vl2-small","bce-v3/ALTAK-n6urhfuXGQ9TH3uyUuuof/0f201cb1f6d2af8ff0ceeb02ae876e5fa151c913"],
    "Qianfan-QI-VL": ["qianfan-qi-vl", "bce-v3/ALTAK-1ZbrHo15FfIZ7fwVEBbUq/c749613848f2cc502c7b91253f8b72fcf4a94246"],
    "Llama-4-Maverick-17B-128E-Instruct": ["llama-4-maverick-17b-128e-instruct", "bce-v3/ALTAK-Yv9zLhNLHph36dwGQAXDM/28e8fe6bb2572e12787c952821f096e7d24c66a0"],
    "Llama-4-Scout-17B-16E-Instruct": ["llama-4-scout-17b-16e-instruct", "bce-v3/ALTAK-y87q5FLcH8mC4q0mTEs50/ae0bd14acd7c16f28335fb3ec6b1f328ee6cea42"],
    "InternVL3-38B": ["internvl3-38b", "bce-v3/ALTAK-8C6KZKkeZcddWcAvsCeTp/60a44930d12f731e186685b5403c4a2cb3d76191"],
    "GLM-4.5V": ["glm-4.5v", "bce-v3/ALTAK-opnmfSkJMGShEKBvNm33E/50c573400a0877a615dbb9242a9ebffb446813cf"],
    "Gpt-o4-mini": ["o4-mini", "sk-aP4qsxNjhz8SLmDbvBHMStKBY6KcG2vC55mo9kPM9yOevGJp"]
}

# 创建调度器
scheduler = AsyncAPIScheduler()

# API配置 - 每个配置项包含4个参数：[API类, base_url, api_key, processing_type]
api_configs = [
    [ExampleChatAPI, *models_to_evaluate["Qwen2.5-VL-32B-Instruct"], 0],
    [ExampleChatAPI, *models_to_evaluate["DeepSeek-VL2-Small"], 0],
    [ExampleChatAPI, *models_to_evaluate["Qianfan-QI-VL"], 0],
    [ExampleChatAPI, *models_to_evaluate["Llama-4-Maverick-17B-128E-Instruct"], 0],
    [ExampleChatAPI, *models_to_evaluate["Llama-4-Scout-17B-16E-Instruct"], 0],
    [ExampleChatAPI, *models_to_evaluate["InternVL3-38B"], 0],
    [ExampleChatAPI, *models_to_evaluate["GLM-4.5V"], 0],
    [ExampleChatAPI, *models_to_evaluate["Gpt-o4-mini"], 0]
]

# 创建模型名称映射表，用于识别和记录
model_name_mapping = {
    0: "Qwen2.5-VL-32B-Instruct",
    1: "DeepSeek-VL2-Small",
    2: "Qianfan-QI-VL",
    3: "Llama-4-Maverick-17B-128E-Instruct",
    4: "Llama-4-Scout-17B-16E-Instruct",
    5: "InternVL3-38B",
    6: "GLM-4.5V",
    7: "Gpt-o4-mini"
}

# 基础路径配置
base_path = "Data3/无处理/PSQ/Geom/2/"
TEST_IMAGE_PATH = r"E:\Preprocessing\AI4S_2\M\train\PSQ\Geom\2.png"

# 常量定义
GROUP_NAME = "DiffGroup"
ANSWER_FILENAME = base_path + "answer.json"  # 只保存answer.json


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
        "multi_modal_group_api_0": "Qwen2.5-VL-32B-Instruct",
        "multi_modal_group_api_1": "DeepSeek-VL2-Small",
        "multi_modal_group_api_2": "Qianfan-QI-VL",
        "multi_modal_group_api_3": "Llama-4-Maverick-17B-128E-Instruct",
        "multi_modal_group_api_4": "Llama-4-Scout-17B-16E-Instruct",
        "multi_modal_group_api_5": "InternVL3-38B",
        "multi_modal_group_api_6": "GLM-4.5V",
        "multi_modal_group_api_7": "Gpt-o4-mini"
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
        # 打印部分内容预览
        # if response.content and len(response.content) > 100:
        #     print(f"    内容预览: {response.content[:100]}...")
    else:
        print(f"  {model_name} (处理类型{processing_type}, {status}): {response.error_message}")


def save_answer_results(results_data: Dict, filename: str) -> None:
    """保存答案结果到answer.json文件

    Args:
        results_data: 要保存的结果数据
        filename: 文件名
    """
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, ensure_ascii=False, indent=2)

        print(f"=== 答案结果已保存 ===")
        print(f"文件: {filename}")
        print(f"包含模型数量: {len(results_data.get(GROUP_NAME, {}).get('models', {}))}")
    except Exception as e:
        print(f"保存答案结果失败: {e}")


async def call_apis_and_process_results() -> Optional[Dict]:
    """调用API并处理结果

    Returns:
        Dict: API调用结果
    """
    print("\n=== 开始调用多模态API ===")

    try:
        # 创建测试数据 - 数学题目求解
        image_text_data = APIData(
            text="""
请严格遵循以下格式解答用户提供的数学题目图片：
一、选择题输出格式
【答案】
直接给出正确选项字母（如"A"或"C"）。
【答案内容】
简要说明该选项对应的结果或结论，避免冗余描述。
【解题思路】
分步骤分析关键逻辑，包括：
题目涉及的公式、定理或方法；
选项排除或验证的依据；
核心推理链条（如数值计算、逻辑推导）。
二、解答题（大题）输出格式
【题目分解】
明确解题步骤数量；
拆解关键子问题，指出核心难点（如隐含条件、复杂变换）。
【解题思路】
阐述方法论（如等价转化、数形结合、数学模型构建）；
说明选择该方法的理由及预期路径。
【解题过程】
详细推导步骤，需包含：
每一步的公式、计算过程；
依据（定理定义、逻辑关系）；
必要的中途结果验证。
【最终答案】
清晰标注最终结果，若涉及单位需注明。
注意：避免冗余，确保关键步骤无遗漏，且解题过程完整,尽量简洁。
            """,
            image_path=TEST_IMAGE_PATH
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
    """主函数 - 只进行API调用和结果保存

    Returns:
        Dict: 包含API结果的字典
    """
    start_time = time.time()

    try:
        # 阶段1: 获取模型回答
        print("=== 阶段1: 调用API获取模型回答 ===")
        results = await call_apis_and_process_results()
        if not results:
            print("API调用失败: 无法获取模型回答")
            return None

        # 准备存储结果的字典结构
        answer_results = {
            GROUP_NAME: {
                "input_text": "请回答所提供图片中的数学题目",
                "image_path": TEST_IMAGE_PATH,
                "models": {},
                "total_models": len(results.get(GROUP_NAME, [])),
                "success_count": 0,
                "failed_count": 0
            }
        }

        # 处理每个API的响应结果并存储
        response_list = results.get(GROUP_NAME, [])
        success_count = 0

        print(f"\n=== 处理API响应结果 ===")
        print(f"共获取到 {len(response_list)} 个模型响应")

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

            # 更新成功/失败计数
            if response.success:
                success_count += 1

            # 添加到结果字典中
            answer_results[GROUP_NAME]["models"][model_name] = model_result

            # 输出简要信息
            print_api_result(model_name, response)

        # 更新统计信息
        answer_results[GROUP_NAME]["success_count"] = success_count
        answer_results[GROUP_NAME]["failed_count"] = len(response_list) - success_count

        # 阶段2: 保存答案结果到answer.json
        print("\n=== 阶段2: 保存结果到answer.json ===")
        save_answer_results(answer_results, ANSWER_FILENAME)

        total_time = time.time() - start_time
        print(f"\n=== 流程完成 ===")
        print(f"总耗时: {total_time:.2f}秒")
        print(f"成功模型: {success_count}/{len(response_list)}")
        print(f"结果文件: {ANSWER_FILENAME}")

        return answer_results

    except Exception as e:
        print(f"处理过程中发生错误: {e}")
        return None


if __name__ == "__main__":
    # 使用asyncio.run()正确运行异步函数[1](@ref)
    try:
        # 确保事件循环正确关闭[5](@ref)
        results = asyncio.run(main())
        if results:
            print("程序执行成功 - 仅完成API调用和结果保存")
        else:
            print("程序执行完成，但API调用可能失败")
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"程序执行失败: {e}")
