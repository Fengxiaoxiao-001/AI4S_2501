import os
import pandas as pd
from openpyxl import load_workbook
import glob


def extract_final_scores(root_path):
    """
    从复杂的文件夹结构中提取所有Excel文件的绝对加权最终评分

    Parameters:
    root_path (str): 根目录路径（包含"简单"和"中等"文件夹）

    Returns:
    pd.DataFrame: 包含所有提取数据的DataFrame
    """

    # 定义目标模型列表（作为输出文件的列名）
    target_models = [
        "Qwen2.5-VL-32B-Instruct",
        "DeepSeek-VL2-Small",
        "Qianfan-QI-VL",
        "Llama-4-Maverick-17B-128E-Instruct",
        "Llama-4-Scout-17B-16E-Instruct",
        "InternVL3-38B",
        "GLM-4.5V",
        "Gpt-o4-mini"
    ]

    # 存储所有提取的数据
    all_data = []
    file_paths = []  # 记录文件路径用于行标识

    # 第一层：简单和中等文件夹
    for level in ["简单", "中等"]:
        level_path = os.path.join(root_path, level)
        if not os.path.exists(level_path):
            continue

        # 第二层：多智能体、角色扮演、无处理
        for method in ["多智能体", "角色扮演", "无处理"]:
            method_path = os.path.join(level_path, method)
            if not os.path.exists(method_path):
                continue

            # 第三层：AQ, FBQ, MCQ, PSQ
            for qtype in ["AQ", "FBQ", "MCQ", "PSQ"]:
                qtype_path = os.path.join(method_path, qtype)
                if not os.path.exists(qtype_path):
                    continue

                # 第四层：可能有的子文件夹（Alge, Geom, StaAndPro）
                possible_subfolders = ["Alge", "Geom", "StaAndPro"]
                search_paths = [qtype_path]  # 包含当前路径

                # 添加可能存在的子文件夹路径
                for subfolder in possible_subfolders:
                    sub_path = os.path.join(qtype_path, subfolder)
                    if os.path.exists(sub_path):
                        search_paths.append(sub_path)

                # 在所有搜索路径中查找Excel文件
                for search_path in search_paths:
                    # 查找所有xlsx和xls文件，排除临时文件
                    excel_files = []
                    for pattern in ["*.xlsx", "*.xls"]:
                        excel_files.extend(glob.glob(os.path.join(search_path, pattern)))

                    # 排除以~$开头的临时文件
                    excel_files = [f for f in excel_files if not os.path.basename(f).startswith('~$')]

                    # 处理每个Excel文件
                    for excel_file in excel_files:
                        try:
                            # 读取Excel文件
                            df = pd.read_excel(excel_file, sheet_name='Sheet1')

                            # 检查必要的列是否存在
                            if '模型名' not in df.columns or '绝对加权最终评分' not in df.columns:
                                print(f"警告：文件 {excel_file} 中缺少必要的列")
                                continue

                            # 创建当前文件的评分字典
                            score_dict = {}

                            # 提取每个模型的评分
                            for _, row in df.iterrows():
                                model_name = row['模型名'].strip()  # 去除前后空格
                                final_score = row['绝对加权最终评分']

                                # 只收集目标模型的数据
                                if model_name in target_models:
                                    score_dict[model_name] = final_score

                            # 如果找到了数据，添加到总数据中
                            if score_dict:
                                # 确保所有目标模型都有值（缺失的设为NaN）
                                complete_row = {model: score_dict.get(model, float('nan'))
                                                for model in target_models}
                                complete_row['文件路径'] = excel_file
                                complete_row['难度级别'] = level
                                complete_row['方法类型'] = method
                                complete_row['题型'] = qtype
                                complete_row['子类别'] = os.path.basename(os.path.dirname(excel_file))
                                if complete_row['子类别'] not in possible_subfolders:
                                    complete_row['子类别'] = '无'

                                all_data.append(complete_row)
                                file_paths.append(excel_file)

                        except Exception as e:
                            print(f"错误处理文件 {excel_file}: {str(e)}")
                            continue

    if not all_data:
        print("未找到任何Excel文件或数据")
        return None

    # 创建最终的DataFrame
    result_df = pd.DataFrame(all_data)

    # 重新排列列的顺序，将模型评分列放在前面
    metadata_cols = ['文件路径', '难度级别', '方法类型', '题型', '子类别']
    model_cols = [col for col in result_df.columns if col not in metadata_cols]

    final_df = result_df[model_cols + metadata_cols]

    return final_df


def save_consolidated_data(df, output_file="consolidated_scores.xlsx"):
    """
    保存合并后的数据到Excel文件

    Parameters:
    df (pd.DataFrame): 要保存的数据
    output_file (str): 输出文件名
    """
    if df is None:
        print("没有数据可保存")
        return

    try:
        # 保存到Excel
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # 主数据表
            df.to_excel(writer, sheet_name='合并数据', index=False)

            # 创建摘要表（每个文件的汇总统计）
            summary_data = []
            for i, row in df.iterrows():
                file_scores = row[~pd.isna(row)].to_dict()
                # 只保留数值型数据（模型评分）
                numeric_scores = {k: v for k, v in file_scores.items()
                                  if k not in ['文件路径', '难度级别', '方法类型', '题型', '子类别']
                                  and isinstance(v, (int, float))}

                if numeric_scores:
                    summary_row = {
                        '文件路径': row['文件路径'],
                        '难度级别': row['难度级别'],
                        '方法类型': row['方法类型'],
                        '题型': row['题型'],
                        '子类别': row['子类别'],
                        '有效评分数量': len(numeric_scores),
                        '平均分': sum(numeric_scores.values()) / len(numeric_scores),
                        '最高分': max(numeric_scores.values()),
                        '最低分': min(numeric_scores.values())
                    }
                    summary_data.append(summary_row)

            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='文件摘要', index=False)

        print(f"数据已成功保存到: {output_file}")
        print(f"共处理了 {len(df)} 个文件的数据")

    except Exception as e:
        print(f"保存文件时出错: {str(e)}")


# 主程序
if __name__ == "__main__":
    # 设置根目录路径（请根据实际情况修改）
    root_directory = r"E:\Preprocessing\AI4S_3\掌握程度M"  # 替换为你的实际路径

    # 检查路径是否存在
    if not os.path.exists(root_directory):
        print(f"错误：路径 {root_directory} 不存在")
        print("请修改root_directory变量为正确的路径")
    else:
        # 提取数据
        print("开始提取数据...")
        consolidated_data = extract_final_scores(root_directory)

        if consolidated_data is not None:
            # 保存结果
            output_filename = "掌握程度M/模型评分汇总.xlsx"
            save_consolidated_data(consolidated_data, output_filename)

            # 显示前几行数据预览
            print("\n数据预览（前5行）:")
            print(consolidated_data.head())

            # 显示基本统计信息
            print(f"\n共提取了 {len(consolidated_data)} 个文件的数据")
            print(f"包含列: {list(consolidated_data.columns)}")
        else:
            print("未能提取到任何数据")