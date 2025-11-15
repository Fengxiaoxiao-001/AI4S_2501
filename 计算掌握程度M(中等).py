import os
import pandas as pd
import glob
import re


def extract_final_scores_medium(root_path):
    """
    专门处理中等文件夹结构，提取所有Excel文件的绝对加权最终评分

    Parameters:
    root_path (str): 根目录路径（包含"中等"文件夹）

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

    # 中等文件夹路径
    medium_path = os.path.join(root_path, "中等")
    if not os.path.exists(medium_path):
        print(f"错误：中等文件夹路径不存在: {medium_path}")
        return None

    # 第二层：多智能体、角色扮演、无处理
    for method in ["多智能体", "角色扮演", "无处理"]:
        method_path = os.path.join(medium_path, method)
        if not os.path.exists(method_path):
            print(f"警告：方法文件夹不存在: {method_path}")
            continue

        # 第三层：AQ, FBQ, MCQ, PSQ
        for qtype in ["AQ", "FBQ", "MCQ", "PSQ"]:
            qtype_path = os.path.join(method_path, qtype)
            if not os.path.exists(qtype_path):
                print(f"警告：题型文件夹不存在: {qtype_path}")
                continue

            # 查找所有xlsx文件，排除临时文件
            excel_files = []
            for pattern in ["*.xlsx", "*.xls"]:
                excel_files.extend(glob.glob(os.path.join(qtype_path, pattern)))

            # 排除以~$开头的临时文件
            excel_files = [f for f in excel_files if not os.path.basename(f).startswith('~$')]

            # 处理每个Excel文件
            for excel_file in excel_files:
                try:
                    # 从文件名解析主题和编号
                    filename = os.path.basename(excel_file)
                    # 文件名格式：主题_编号.xlsx
                    name_parts = filename.replace('.xlsx', '').replace('.xls', '').split('_')

                    if len(name_parts) < 2:
                        print(f"警告：文件名格式不正确: {filename}")
                        continue

                    theme = name_parts[0]  # 主题：Alge, Geom, StaAndPro等
                    file_id = name_parts[1] if len(name_parts) > 1 else "未知"

                    # 验证主题的合法性
                    valid_themes = ["Alge", "Geom", "StaAndPro"]  # 可以根据实际情况扩展

                    if theme not in valid_themes:
                        print(f"警告：可能无效的主题: {theme}，文件: {filename}")
                        # 不跳过，只是警告，因为可能有新的主题

                    # 读取Excel文件
                    df = pd.read_excel(excel_file, sheet_name='Sheet1')

                    # 检查并清理列名（去除前后空格和特殊字符）
                    df.columns = df.columns.str.strip()

                    # 动态确定评分列名 - 支持多种可能的列名
                    score_column = None
                    possible_score_columns = ['绝对加权最终评分', '绝对平均分（最终）', '绝对平均分', '最终评分']

                    for col in possible_score_columns:
                        if col in df.columns:
                            score_column = col
                            break

                    if score_column is None:
                        print(f"警告：文件 {excel_file} 中缺少评分列")
                        print(f"现有列: {list(df.columns)}")
                        # 尝试查找包含"评分"或"平均"的列
                        for col in df.columns:
                            if '评分' in col or '平均' in col:
                                score_column = col
                                print(f"使用近似列名: {score_column}")
                                break

                    if score_column is None:
                        print(f"无法确定评分列，跳过文件: {excel_file}")
                        continue

                    # 检查必要的列是否存在
                    if '模型名' not in df.columns:
                        print(f"警告：文件 {excel_file} 中缺少'模型名'列")
                        print(f"现有列: {list(df.columns)}")
                        continue

                    # 创建当前文件的评分字典
                    score_dict = {}

                    # 提取每个模型的评分
                    for _, row in df.iterrows():
                        model_name = str(row['模型名']).strip()  # 确保是字符串并去除前后空格

                        # 跳过空模型名
                        if pd.isna(model_name) or model_name == '':
                            continue

                        final_score = row[score_column]

                        # 只收集目标模型的数据
                        if model_name in target_models:
                            score_dict[model_name] = final_score

                    # 如果找到了数据，添加到总数据中
                    if score_dict:
                        # 确保所有目标模型都有值（缺失的设为NaN）
                        complete_row = {model: score_dict.get(model, float('nan'))
                                        for model in target_models}
                        complete_row['文件路径'] = excel_file
                        complete_row['难度级别'] = "中等"
                        complete_row['方法类型'] = method
                        complete_row['题型'] = qtype
                        complete_row['主题'] = theme
                        complete_row['文件编号'] = file_id
                        complete_row['使用的评分列'] = score_column  # 记录实际使用的列名

                        all_data.append(complete_row)
                        print(f"成功处理文件: {filename} (使用列: {score_column})")
                    else:
                        print(f"警告：文件 {filename} 中未找到目标模型的评分数据")
                        # 调试信息：显示文件中实际存在的模型名
                        actual_models = [str(row['模型名']).strip() for _, row in df.iterrows()
                                         if not pd.isna(row['模型名']) and str(row['模型名']).strip() != '']
                        print(f"文件中实际模型: {actual_models}")

                except Exception as e:
                    print(f"错误处理文件 {excel_file}: {str(e)}")
                    import traceback
                    print(f"详细错误信息: {traceback.format_exc()}")
                    continue

    if not all_data:
        print("未找到任何Excel文件或数据")
        return None

    # 创建最终的DataFrame
    result_df = pd.DataFrame(all_data)

    # 重新排列列的顺序，将模型评分列放在前面
    metadata_cols = ['文件路径', '难度级别', '方法类型', '题型', '主题', '文件编号', '使用的评分列']
    model_cols = [col for col in result_df.columns if col not in metadata_cols]

    final_df = result_df[model_cols + metadata_cols]

    return final_df


def save_consolidated_data_medium(df, output_file="中等_模型评分汇总.xlsx"):
    """
    保存合并后的数据到Excel文件（专门为中等文件夹结构优化）

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

            # 创建按题型和主题分类的汇总表
            if not df.empty:
                # 分组统计
                summary_by_type_theme = df.groupby(['题型', '主题']).agg({
                    '文件路径': 'count',
                    **{model: ['mean', 'min', 'max'] for model in df.columns
                       if model not in ['文件路径', '难度级别', '方法类型', '题型', '主题', '文件编号', '使用的评分列']}
                }).round(2)

                # 扁平化多级列索引
                if not summary_by_type_theme.empty:
                    summary_by_type_theme.columns = ['_'.join(col).strip() for col in
                                                     summary_by_type_theme.columns.values]
                    summary_by_type_theme.reset_index(inplace=True)
                    summary_by_type_theme.rename(columns={'文件路径_count': '文件数量'}, inplace=True)
                    summary_by_type_theme.to_excel(writer, sheet_name='题型主题汇总', index=False)

            # 创建方法类型汇总表
            if not df.empty:
                summary_by_method = df.groupby(['方法类型']).agg({
                    '文件路径': 'count',
                    **{model: 'mean' for model in df.columns
                       if model not in ['文件路径', '难度级别', '方法类型', '题型', '主题', '文件编号', '使用的评分列']}
                }).round(2)

                if not summary_by_method.empty:
                    summary_by_method.columns = [f"{col}_平均分" if col != '文件路径' else '文件数量'
                                                 for col in summary_by_method.columns]
                    summary_by_method.reset_index(inplace=True)
                    summary_by_method.to_excel(writer, sheet_name='方法类型汇总', index=False)

            # 创建文件摘要表
            summary_data = []
            for i, row in df.iterrows():
                file_scores = row[~pd.isna(row)].to_dict()
                # 只保留数值型数据（模型评分）
                numeric_scores = {k: v for k, v in file_scores.items()
                                  if k not in ['文件路径', '难度级别', '方法类型', '题型', '主题', '文件编号',
                                               '使用的评分列']
                                  and isinstance(v, (int, float))}

                if numeric_scores:
                    summary_row = {
                        '文件路径': row['文件路径'],
                        '难度级别': row['难度级别'],
                        '方法类型': row['方法类型'],
                        '题型': row['题型'],
                        '主题': row['主题'],
                        '文件编号': row['文件编号'],
                        '使用的评分列': row.get('使用的评分列', '未知'),
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
        import traceback
        print(f"详细错误信息: {traceback.format_exc()}")


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
        print("开始提取中等文件夹数据...")
        consolidated_data = extract_final_scores_medium(root_directory)

        if consolidated_data is not None:
            # 保存结果
            output_filename = "掌握程度M/中等_模型评分汇总.xlsx"
            save_consolidated_data_medium(consolidated_data, output_filename)

            # 显示前几行数据预览
            print("\n数据预览（前5行）:")
            print(consolidated_data.head())

            # 显示基本统计信息
            print(f"\n共提取了 {len(consolidated_data)} 个文件的数据")
            print(f"包含列: {list(consolidated_data.columns)}")

            # 显示数据类型分布
            if not consolidated_data.empty:
                print("\n文件类型分布:")
                type_dist = consolidated_data.groupby(['方法类型', '题型', '主题']).size().reset_index(name='文件数量')
                print(type_dist)

                # 显示使用的评分列分布
                print("\n使用的评分列分布:")
                score_col_dist = consolidated_data['使用的评分列'].value_counts()
                print(score_col_dist)

                # 显示方法类型分布
                print("\n方法类型分布:")
                method_dist = consolidated_data['方法类型'].value_counts()
                print(method_dist)
        else:
            print("未能提取到任何数据")