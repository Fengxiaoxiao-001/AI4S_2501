import asyncio
from InputAndOutput_Layer import ExampleChatAPI, APIData  # 替换为实际模块名


async def test_chatgpt_api_with_image():
    """测试ChatGPTAPI处理文本+图片地址的功能"""

    # 初始化API实例 - 使用您的配置
    api = ExampleChatAPI(
        name="Gpt-o4-mini",
        api_key="sk-aP4qsxNjhz8SLmDbvBHMStKBY6KcG2vC55mo9kPM9yOevGJp",  # 您的API密钥
        model_name="o4-mini",  # 或多模态模型
        processing_type=0  # 直接处理
    )

    try:
        # 测试数据：文本 + 图片路径
        # 请将路径替换为您的实际图片路径
        test_data = APIData(
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
注意：避免冗余，确保关键步骤无遗漏，且解题过程完整。
注意(图片以base64的格式发送给你)
            """,
            image_path =  r"E:\Preprocessing\AI4S_2\M\train\PSQ\FunAndCal\2.png"  # 替换为您的图片路径
        )

        print("开始测试ChatGPTAPI...")

        # 调用API处理
        response = await api.process(test_data)

        # 输出结果
        print("\n=== 测试结果 ===")
        print(f"成功: {response.success}")
        print(f"API名称: {response.api_name}")
        print(f"处理类型: {response.processing_type}")
        print(f"响应时间: {response.response_time:.2f}秒")

        if response.success:
            print(f"返回内容: {response.content}")
            if response.data:
                print("原始数据可用")
        else:
            print(f"错误信息: {response.error_message}")

    except Exception as e:
        print(f"测试过程中发生错误: {e}")
    finally:
        # 清理资源
        await api.close()


# 运行测试
if __name__ == "__main__":
    print("正在测试ChatGPTAPI多模态功能...")

    # 运行主测试
    asyncio.run(test_chatgpt_api_with_image())

    # 可选：运行简单测试用例
    # asyncio.run(simple_test_cases())

    print("\n测试完成！")
