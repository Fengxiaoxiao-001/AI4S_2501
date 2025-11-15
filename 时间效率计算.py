import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
import os
import glob
from collections import defaultdict


def parse_question_id(question_id):
    """解析题目ID，提取难度、主题、题型信息"""
    # 根据你的实际命名规则调整
    if question_id.startswith('Easy'):
        difficulty = 'Easy'
        rest = question_id[4:]
    elif question_id.startswith('Medium'):
        difficulty = 'Medium'
        rest = question_id[6:]
    elif question_id.startswith('Hard'):
        difficulty = 'Hard'
        rest = question_id[4:]
    else:
        difficulty = 'Unknown'
        rest = question_id

    # 解析题型和主题
    question_type = rest[:2]  # 示例: Aq
    topic = rest[2:]  # 示例: Geom

    return {
        'difficulty': difficulty,
        'question_type': question_type,
        'topic': topic
    }


def load_all_json_data(base_path):
    """
    加载所有JSON数据，考虑不同的提示词工程策略
    base_path: 根目录路径，如 "E:/Preprocessing/AI4S_2/Math/"
    """
    all_data = {}

    # 定义提示词工程类型
    prompt_strategies = {
        "无处理": "no_processing",
        "多智能体": "multi_agent",
        "角色扮演": "role_playing"
    }

    # 定义难度级别
    difficulties = ["简单", "中等", "困难"]

    # 遍历所有目录结构
    for difficulty in difficulties:
        for strategy_name, strategy_code in prompt_strategies.items():
            # 构建搜索模式
            search_pattern = os.path.join(
                base_path,
                difficulty,
                strategy_name,
                "*",  # 题型
                "*",  # 主题
                "*",  # 题目编号
                "answer.json"
            )

            # 查找所有匹配的JSON文件
            json_files = glob.glob(search_pattern)

            for json_file in json_files:
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        file_data = json.load(f)

                    # 提取题目ID和题目数据
                    for question_id, question_data in file_data.items():
                        # 添加提示词工程信息
                        question_data['prompt_strategy'] = strategy_name
                        question_data['prompt_strategy_code'] = strategy_code
                        question_data['file_path'] = json_file

                        # 添加到总数据中
                        all_data[question_id] = question_data

                except Exception as e:
                    print(f"Error loading {json_file}: {e}")

    print(f"成功加载 {len(all_data)} 个题目数据")
    return all_data


def comprehensive_efficiency_analysis_all_questions(full_json_data):
    """分析所有题目的效率（保留双效率指标）"""

    all_results = []

    for question_id, question_data in full_json_data.items():
        # 解析题目信息
        question_info = parse_question_id(question_id)

        # 添加提示词工程信息
        question_info['prompt_strategy'] = question_data.get('prompt_strategy', 'Unknown')
        question_info['prompt_strategy_code'] = question_data.get('prompt_strategy_code', 'unknown')

        if 'models' in question_data:
            for model_name, model_data in question_data['models'].items():
                if model_data['success']:
                    rt = model_data['response_time']
                    content = model_data['content']
                    content_len = len(content)

                    # 保留双效率指标
                    raw_efficiency = content_len / rt
                    normalized_efficiency = math.log(content_len + 1) / rt

                    result = {
                        'question_id': question_id,
                        'model': model_name,
                        'response_time': rt,
                        'content_length': content_len,
                        'raw_efficiency': raw_efficiency,
                        'normalized_efficiency': normalized_efficiency,
                        'efficiency_ratio': normalized_efficiency / raw_efficiency if raw_efficiency > 0 else 0,
                        'prompt_strategy': question_info['prompt_strategy'],
                        'prompt_strategy_code': question_info['prompt_strategy_code']
                    }

                    # 添加题目分类信息
                    result.update(question_info)
                    all_results.append(result)

    df = pd.DataFrame(all_results)
    return df


def create_comprehensive_analysis_charts(df):
    """创建综合分析图表 - 结合双效率对比、分层分析和提示词工程分析"""

    # 创建3x3的子图布局
    fig, axes = plt.subplots(3, 3, figsize=(24, 18))

    # 1. 总体双效率对比（左上）
    model_avg = df.groupby('model').agg({
        'raw_efficiency': 'mean',
        'normalized_efficiency': 'mean',
        'efficiency_ratio': 'mean'
    }).reset_index()

    # 原始效率排名
    model_avg_raw = model_avg.sort_values('raw_efficiency', ascending=False)
    axes[0, 0].bar(model_avg_raw['model'], model_avg_raw['raw_efficiency'],
                   color='lightblue', alpha=0.7)
    axes[0, 0].set_title('📊 总体原始效率排名\n(字符/秒)', fontsize=12, fontweight='bold')
    axes[0, 0].tick_params(axis='x', rotation=45)

    # 2. 标准化效率排名（中上）
    model_avg_norm = model_avg.sort_values('normalized_efficiency', ascending=False)
    axes[0, 1].bar(model_avg_norm['model'], model_avg_norm['normalized_efficiency'],
                   color='lightgreen', alpha=0.7)
    axes[0, 1].set_title('⚡ 总体标准化效率排名\n(log(长度)/秒)', fontsize=12, fontweight='bold')
    axes[0, 1].tick_params(axis='x', rotation=45)

    # 3. 效率比值分析（右上）
    model_avg_ratio = model_avg.sort_values('efficiency_ratio', ascending=False)
    axes[0, 2].bar(model_avg_ratio['model'], model_avg_ratio['efficiency_ratio'],
                   color='orange', alpha=0.7)
    axes[0, 2].set_title('💎 效率比值排名\n(标准化/原始)', fontsize=12, fontweight='bold')
    axes[0, 2].tick_params(axis='x', rotation=45)
    axes[0, 2].axhline(y=1, color='red', linestyle='--', alpha=0.5)

    # 4. 按难度分层的原始效率（中左）
    difficulties = ['Easy', 'Medium', 'Hard']
    colors = ['#ff9999', '#66b3ff', '#99ff99']

    # 准备数据
    difficulty_data = []
    for difficulty in difficulties:
        df_diff = df[df['difficulty'] == difficulty]
        diff_avg = df_diff.groupby('model')['raw_efficiency'].mean().reset_index()
        diff_avg['difficulty'] = difficulty
        difficulty_data.append(diff_avg)

    # 创建分组柱状图
    difficulty_df = pd.concat(difficulty_data)
    pivot_df = difficulty_df.pivot(index='model', columns='difficulty', values='raw_efficiency')

    x = np.arange(len(pivot_df.index))
    width = 0.25

    for i, difficulty in enumerate(difficulties):
        axes[1, 0].bar(x + i * width, pivot_df[difficulty], width,
                       label=difficulty, color=colors[i], alpha=0.7)

    axes[1, 0].set_xlabel('模型')
    axes[1, 0].set_ylabel('原始效率')
    axes[1, 0].set_title('📈 按难度分层的原始效率', fontsize=12, fontweight='bold')
    axes[1, 0].set_xticks(x + width)
    axes[1, 0].set_xticklabels(pivot_df.index, rotation=45)
    axes[1, 0].legend()

    # 5. 响应时间热力图（中中）- 按难度和模型
    time_pivot = df.pivot_table(values='response_time',
                                index='model',
                                columns='difficulty',
                                aggfunc='mean')

    im = axes[1, 1].imshow(time_pivot, cmap='YlOrRd', aspect='auto')
    axes[1, 1].set_title('⏱️ 平均响应时间热力图', fontsize=12, fontweight='bold')
    axes[1, 1].set_xticks(range(len(time_pivot.columns)))
    axes[1, 1].set_xticklabels(time_pivot.columns)
    axes[1, 1].set_yticks(range(len(time_pivot.index)))
    axes[1, 1].set_yticklabels(time_pivot.index)
    plt.colorbar(im, ax=axes[1, 1])

    # 6. 效率-时间散点图（中右）
    avg_data = df.groupby('model').agg({
        'raw_efficiency': 'mean',
        'response_time': 'mean',
        'normalized_efficiency': 'mean'
    }).reset_index()

    scatter = axes[1, 2].scatter(avg_data['response_time'],
                                 avg_data['raw_efficiency'],
                                 s=avg_data['normalized_efficiency'] * 100,
                                 alpha=0.7, cmap='viridis')

    for i, row in avg_data.iterrows():
        axes[1, 2].annotate(row['model'],
                            (row['response_time'], row['raw_efficiency']),
                            xytext=(5, 5), textcoords='offset points',
                            fontsize=8)

    axes[1, 2].set_xlabel('平均响应时间 (秒)')
    axes[1, 2].set_ylabel('平均原始效率')
    axes[1, 2].set_title('🔄 效率-时间关系图\n(点大小反映标准化效率)', fontsize=12, fontweight='bold')
    axes[1, 2].grid(True, alpha=0.3)

    # 7. 按提示词工程策略分析（左下）- 原始效率
    prompt_strategies = df['prompt_strategy'].unique()
    prompt_data = []

    for strategy in prompt_strategies:
        df_strategy = df[df['prompt_strategy'] == strategy]
        strategy_avg = df_strategy.groupby('model')['raw_efficiency'].mean().reset_index()
        strategy_avg['strategy'] = strategy
        prompt_data.append(strategy_avg)

    prompt_df = pd.concat(prompt_data)
    prompt_pivot = prompt_df.pivot(index='model', columns='strategy', values='raw_efficiency')

    x_prompt = np.arange(len(prompt_pivot.index))
    width_prompt = 0.25

    for i, strategy in enumerate(prompt_strategies):
        if strategy in prompt_pivot.columns:
            axes[2, 0].bar(x_prompt + i * width_prompt, prompt_pivot[strategy], width_prompt,
                           label=strategy, alpha=0.7)

    axes[2, 0].set_xlabel('模型')
    axes[2, 0].set_ylabel('原始效率')
    axes[2, 0].set_title('🎭 按提示词工程策略的效率对比', fontsize=12, fontweight='bold')
    axes[2, 0].set_xticks(x_prompt + width_prompt)
    axes[2, 0].set_xticklabels(prompt_pivot.index, rotation=45)
    axes[2, 0].legend()

    # 8. 提示词工程策略效果热力图（下中）
    strategy_time_pivot = df.pivot_table(values='response_time',
                                         index='model',
                                         columns='prompt_strategy',
                                         aggfunc='mean')

    im_strategy = axes[2, 1].imshow(strategy_time_pivot, cmap='YlOrRd', aspect='auto')
    axes[2, 1].set_title('🔧 提示词工程策略响应时间热力图', fontsize=12, fontweight='bold')
    axes[2, 1].set_xticks(range(len(strategy_time_pivot.columns)))
    axes[2, 1].set_xticklabels(strategy_time_pivot.columns)
    axes[2, 1].set_yticks(range(len(strategy_time_pivot.index)))
    axes[2, 1].set_yticklabels(strategy_time_pivot.index)
    plt.colorbar(im_strategy, ax=axes[2, 1])

    # 9. 提示词工程策略效果对比（右下）
    strategy_effectiveness = df.groupby(['model', 'prompt_strategy']).agg({
        'raw_efficiency': 'mean',
        'response_time': 'mean'
    }).reset_index()

    # 计算每个模型在不同策略下的效率提升
    model_strategy_comparison = []
    for model in df['model'].unique():
        model_data = strategy_effectiveness[strategy_effectiveness['model'] == model]
        if len(model_data) > 1:
            # 找到最佳策略
            best_strategy = model_data.loc[model_data['raw_efficiency'].idxmax()]
            worst_strategy = model_data.loc[model_data['raw_efficiency'].idxmin()]
            improvement = ((best_strategy['raw_efficiency'] - worst_strategy['raw_efficiency']) /
                           worst_strategy['raw_efficiency']) * 100

            model_strategy_comparison.append({
                'model': model,
                'best_strategy': best_strategy['prompt_strategy'],
                'best_efficiency': best_strategy['raw_efficiency'],
                'worst_strategy': worst_strategy['prompt_strategy'],
                'worst_efficiency': worst_strategy['raw_efficiency'],
                'improvement_percent': improvement
            })

    comparison_df = pd.DataFrame(model_strategy_comparison)
    if not comparison_df.empty:
        comparison_df = comparison_df.sort_values('improvement_percent', ascending=False)
        bars = axes[2, 2].bar(comparison_df['model'], comparison_df['improvement_percent'],
                              color='purple', alpha=0.7)
        axes[2, 2].set_xlabel('模型')
        axes[2, 2].set_ylabel('效率提升百分比 (%)')
        axes[2, 2].set_title('📊 提示词工程策略效果提升对比', fontsize=12, fontweight='bold')
        axes[2, 2].tick_params(axis='x', rotation=45)

        # 添加数值标签
        for bar, improvement in zip(bars, comparison_df['improvement_percent']):
            axes[2, 2].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                            f'{improvement:.1f}%', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    return fig


def print_detailed_efficiency_insights(df):
    """打印详细的效率分析洞察，包括提示词工程效果"""

    print("=" * 80)
    print("📊 详细效率分析报告（包含提示词工程效果）")
    print("=" * 80)

    # 总体统计
    overall_stats = df.groupby('model').agg({
        'raw_efficiency': ['mean', 'std'],
        'normalized_efficiency': ['mean', 'std'],
        'response_time': ['mean', 'std'],
        'efficiency_ratio': 'mean',
        'question_id': 'count'
    }).round(3)

    # 重命名列
    overall_stats.columns = ['原始效率均值', '原始效率标准差',
                             '标准化效率均值', '标准化效率标准差',
                             '响应时间均值', '响应时间标准差',
                             '效率比值均值', '题目数量']

    print("\n📈 总体效率排名:")
    print(overall_stats.sort_values('原始效率均值', ascending=False))

    # 效率王者
    raw_winner = df.groupby('model')['raw_efficiency'].mean().idxmax()
    norm_winner = df.groupby('model')['normalized_efficiency'].mean().idxmax()
    fastest = df.groupby('model')['response_time'].mean().idxmin()

    print(f"\n🏆 效率王者:")
    print(f"  原始效率最高: {raw_winner}")
    print(f"  标准化效率最高: {norm_winner}")
    print(f"  响应最快: {fastest}")

    # 按难度分析
    print(f"\n🎯 按难度分析:")
    for difficulty in ['Easy', 'Medium', 'Hard']:
        df_diff = df[df['difficulty'] == difficulty]
        if len(df_diff) > 0:
            diff_winner = df_diff.groupby('model')['raw_efficiency'].mean().idxmax()
            avg_time = df_diff['response_time'].mean()
            print(f"  {difficulty}: 最佳模型={diff_winner}, 平均响应时间={avg_time:.2f}秒")

    # 稳定性分析
    print(f"\n📊 稳定性分析 (响应时间标准差):")
    stability = df.groupby('model')['response_time'].std().sort_values()
    for model, std in stability.items():
        print(f"  {model}: {std:.2f}秒")

    # 提示词工程效果分析
    print(f"\n🔧 提示词工程效果分析:")
    prompt_strategies = df['prompt_strategy'].unique()

    for strategy in prompt_strategies:
        df_strategy = df[df['prompt_strategy'] == strategy]
        if len(df_strategy) > 0:
            strategy_winner = df_strategy.groupby('model')['raw_efficiency'].mean().idxmax()
            avg_efficiency = df_strategy['raw_efficiency'].mean()
            avg_time_strategy = df_strategy['response_time'].mean()
            print(
                f"  {strategy}: 最佳模型={strategy_winner}, 平均效率={avg_efficiency:.2f}, 平均响应时间={avg_time_strategy:.2f}秒")

    # 提示词工程策略对比
    print(f"\n📈 提示词工程策略对比:")
    strategy_comparison = df.groupby('prompt_strategy').agg({
        'raw_efficiency': 'mean',
        'response_time': 'mean',
        'question_id': 'count'
    }).round(3)

    strategy_comparison.columns = ['平均效率', '平均响应时间', '题目数量']
    print(strategy_comparison.sort_values('平均效率', ascending=False))

    # 最佳提示词工程策略推荐
    print(f"\n💡 最佳提示词工程策略推荐:")
    best_strategy_overall = strategy_comparison.loc[strategy_comparison['平均效率'].idxmax()]
    print(f"  总体最佳策略: {strategy_comparison['平均效率'].idxmax()}")
    print(f"  平均效率: {best_strategy_overall['平均效率']:.2f}")
    print(f"  平均响应时间: {best_strategy_overall['平均响应时间']:.2f}秒")

    # 按模型分析最佳策略
    print(f"\n🤖 各模型最佳提示词工程策略:")
    model_strategy_analysis = df.groupby(['model', 'prompt_strategy']).agg({
        'raw_efficiency': 'mean',
        'response_time': 'mean'
    }).reset_index()

    for model in df['model'].unique():
        model_data = model_strategy_analysis[model_strategy_analysis['model'] == model]
        if len(model_data) > 0:
            best_for_model = model_data.loc[model_data['raw_efficiency'].idxmax()]
            print(
                f"  {model}: 最佳策略={best_for_model['prompt_strategy']}, 效率={best_for_model['raw_efficiency']:.2f}")


def save_analysis_results(df, chart, output_dir="results"):
    """保存分析结果到文件"""

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 保存图表
    chart.savefig(os.path.join(output_dir, 'comprehensive_efficiency_analysis.png'),
                  dpi=300, bbox_inches='tight')

    # 保存数据
    df.to_csv(os.path.join(output_dir, 'efficiency_analysis_data.csv'),
              index=False, encoding='utf-8')

    # 保存统计摘要
    with open(os.path.join(output_dir, 'analysis_summary.txt'), 'w', encoding='utf-8') as f:
        f.write("效率分析摘要报告\n")
        f.write("=" * 50 + "\n")

        # 总体统计
        overall_stats = df.groupby('model').agg({
            'raw_efficiency': 'mean',
            'response_time': 'mean',
            'question_id': 'count'
        }).round(3)

        f.write("\n总体效率排名:\n")
        f.write(overall_stats.sort_values('raw_efficiency', ascending=False).to_string())

        # 提示词工程效果
        prompt_stats = df.groupby('prompt_strategy').agg({
            'raw_efficiency': 'mean',
            'response_time': 'mean'
        }).round(3)

        f.write("\n\n提示词工程效果:\n")
        f.write(prompt_stats.to_string())

    print(f"分析结果已保存到 {output_dir} 目录")


# 主程序
if __name__ == "__main__":
    # 设置你的数据根目录路径
    BASE_PATH = "E:/Preprocessing/AI4S_2/Math/"  # 请根据实际情况修改

    try:
        # 1. 加载所有JSON数据
        print("正在加载数据...")
        your_json_data = load_all_json_data(BASE_PATH)

        if not your_json_data:
            print("未找到任何数据，请检查路径设置")
            exit()

        # 2. 分析所有题目
        print("正在分析效率...")
        df_all = comprehensive_efficiency_analysis_all_questions(your_json_data)

        # 3. 创建综合分析图表
        print("正在生成图表...")
        chart = create_comprehensive_analysis_charts(df_all)

        # 4. 打印详细洞察
        print_detailed_efficiency_insights(df_all)

        # 5. 保存结果
        save_analysis_results(df_all, chart)

        # 6. 显示数据样本
        print("\n" + "=" * 80)
        print("数据样本")
        print("=" * 80)
        print(df_all.head(10))

        # 7. 显示图表
        plt.show()

        print("\n✅ 分析完成！")

    except Exception as e:
        print(f"❌ 分析过程中出现错误: {e}")
        import traceback

        traceback.print_exc()