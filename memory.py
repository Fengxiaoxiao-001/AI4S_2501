import json
import datetime
import os
from typing import Dict, Any, Optional, List


class MemorySystem:
    """
    记忆体系统类，支持索引存储、标签管理和JSON持久化
    """

    def __init__(self, storage_file: str = "memory_data.json"):
        """
        初始化记忆体系统

        Args:
            storage_file: JSON存储文件名
        """
        self.storage_file = storage_file
        self.memory_index: Dict[int, Dict[str, Dict[str, Any]]] = {}

        # 如果存储文件已存在，则加载数据
        if os.path.exists(storage_file):
            self.load_from_json()

    def add_memory(self, index: int, label: str, content: Any) -> None:
        """
        添加记忆到指定索引和标签

        Args:
            index: 索引编号（如1、2）
            label: 标签（如"0"表示答案，"1"表示解题过程）
            content: 要存储的内容
        """
        # 确保索引存在
        if index not in self.memory_index:
            self.memory_index[index] = {}

        # 存储到指定标签
        self.memory_index[index][label] = {
            "content": content,
            "index": index,
            "label": label,
            "created_at": datetime.datetime.now().isoformat(),  # 记录创建时间
            "last_modified": datetime.datetime.now().isoformat()  # 记录最后修改时间
        }

        # 自动保存到JSON文件
        self.save_to_json()

    def get_memory(self, index: int, label: str) -> Optional[Dict[str, Any]]:
        """
        获取指定索引和标签的记忆内容

        Args:
            index: 索引编号
            label: 标签

        Returns:
            记忆内容字典，包含内容、创建时间等信息
        """
        try:
            return self.memory_index.get(index, {}).get(label)
        except KeyError:
            return None

    def get_index_memories(self, index: int) -> Dict[str, Dict[str, Any]]:
        """
        获取指定索引下的所有记忆

        Args:
            index: 索引编号

        Returns:
            该索引下的所有记忆字典
        """
        return self.memory_index.get(index, {})

    def get_memories_by_label(self, label: str, indexes: Optional[List[int]] = None) -> List[Dict[str, Any]]:
        """
        获取指定标签的所有记忆，可限定在特定索引范围内

        Args:
            label: 要查找的标签
            indexes: 索引列表，如果为None则搜索所有索引

        Returns:
            匹配标签的所有记忆列表
        """
        results = []

        if indexes is None:
            # 搜索所有索引
            search_indexes = self.memory_index.keys()
        else:
            # 只在指定的索引中搜索
            search_indexes = [idx for idx in indexes if idx in self.memory_index]

        for index in search_indexes:
            if label in self.memory_index[index]:
                results.append(self.memory_index[index][label])

        return results

    def get_memories_by_labels(self, labels: List[str], indexes: Optional[List[int]] = None) -> Dict[
        str, List[Dict[str, Any]]]:
        """
        批量获取多个标签的记忆

        Args:
            labels: 标签列表
            indexes: 索引列表，如果为None则搜索所有索引

        Returns:
            按标签分组的记忆字典
        """
        results = {}
        for label in labels:
            results[label] = self.get_memories_by_label(label, indexes)
        return results

    def get_all_memories(self) -> Dict[int, Dict[str, Dict[str, Any]]]:
        """
        获取所有记忆数据

        Returns:
            完整的记忆索引结构
        """
        return self.memory_index

    def get_timeline(self) -> List[Dict[str, Any]]:
        """
        按创建时间排序的所有记忆时间线

        Returns:
            按创建时间排序的记忆列表
        """
        all_memories = []
        for index, labels in self.memory_index.items():
            for label, memory_data in labels.items():
                all_memories.append(memory_data)

        return sorted(all_memories, key=lambda x: x["created_at"])

    def search_memories(self, keyword: str, search_content: bool = True,
                        search_label: bool = False, indexes: Optional[List[int]] = None) -> List[Dict[str, Any]]:
        """
        增强版搜索记忆内容，支持按索引范围搜索

        Args:
            keyword: 搜索关键词
            search_content: 是否搜索内容
            search_label: 是否搜索标签
            indexes: 在特定索引范围内搜索，如果为None则搜索所有索引

        Returns:
            匹配的记忆列表
        """
        results = []

        if indexes is None:
            # 搜索所有索引
            search_indexes = self.memory_index.keys()
        else:
            # 只在指定的索引中搜索
            search_indexes = [idx for idx in indexes if idx in self.memory_index]

        for index in search_indexes:
            for label, memory_data in self.memory_index[index].items():
                matched = False

                if search_content and keyword in str(memory_data["content"]):
                    matched = True
                elif search_label and keyword == label:
                    matched = True
                elif search_label and keyword in label:
                    matched = True

                if matched:
                    results.append(memory_data)

        return results

    def get_indexes_with_label(self, label: str) -> List[int]:
        """
        获取包含指定标签的所有索引编号

        Args:
            label: 要查找的标签

        Returns:
            包含该标签的索引编号列表
        """
        indexes = []
        for index, labels in self.memory_index.items():
            if label in labels:
                indexes.append(index)
        return indexes

    def get_all_labels(self, indexes: Optional[List[int]] = None) -> List[str]:
        """
        获取所有唯一的标签

        Args:
            indexes: 在特定索引范围内查找，如果为None则查找所有索引

        Returns:
            唯一的标签列表
        """
        labels_set = set()

        if indexes is None:
            # 搜索所有索引
            search_indexes = self.memory_index.keys()
        else:
            # 只在指定的索引中搜索
            search_indexes = [idx for idx in indexes if idx in self.memory_index]

        for index in search_indexes:
            labels_set.update(self.memory_index[index].keys())

        return sorted(list(labels_set))

    def save_to_json(self, filename: Optional[str] = None) -> bool:
        """
        将记忆数据保存到JSON文件

        Args:
            filename: 文件名，如果为None则使用初始化时的文件名

        Returns:
            保存是否成功
        """
        if filename is None:
            filename = self.storage_file

        try:
            data_to_save = {
                "memory_index": self.memory_index,
                "last_updated": datetime.datetime.now().isoformat()
            }

            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data_to_save, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"保存到JSON失败: {e}")
            return False

    def load_from_json(self, filename: Optional[str] = None) -> bool:
        """
        从JSON文件加载记忆数据

        Args:
            filename: 文件名，如果为None则使用初始化时的文件名

        Returns:
            加载是否成功
        """
        if filename is None:
            filename = self.storage_file

        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.memory_index = data.get("memory_index", {})
            return True
        except Exception as e:
            print(f"从JSON加载失败: {e}")
            return False

    def export_memory_report(self, report_file: str = "memory_report.txt") -> None:
        """
        导出记忆报告

        Args:
            report_file: 报告文件名
        """
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=== 记忆系统报告 ===\n")
            f.write(f"生成时间: {datetime.datetime.now().isoformat()}\n")
            f.write(f"总记忆数量: {self.get_total_memory_count()}\n")
            f.write(f"总索引数量: {len(self.memory_index)}\n")
            f.write(f"所有标签: {', '.join(self.get_all_labels())}\n\n")

            for index in sorted(self.memory_index.keys()):
                f.write(f"索引 {index}:\n")
                memories = self.get_index_memories(index)
                for label, memory in sorted(memories.items()):
                    created_at = memory['created_at']
                    content_preview = str(memory['content'])[:50] + "..." if len(str(memory['content'])) > 50 else str(
                        memory['content'])
                    f.write(f"  标签 {label} (创建于: {created_at}): {content_preview}\n")
                f.write("\n")

    def get_total_memory_count(self) -> int:
        """
        获取总记忆数量

        Returns:
            记忆总数
        """
        count = 0
        for index_memories in self.memory_index.values():
            count += len(index_memories)
        return count

    def clear_memory(self, index: Optional[int] = None, label: Optional[str] = None) -> None:
        """
        清除记忆

        Args:
            index: 如果指定，只清除该索引的记忆
            label: 如果指定，只清除该标签的记忆
        """
        if index is None and label is None:
            # 清除所有记忆
            self.memory_index.clear()
        elif index is not None and label is None:
            # 清除整个索引
            if index in self.memory_index:
                del self.memory_index[index]
        elif index is not None and label is not None:
            # 清除特定标签的记忆
            if index in self.memory_index and label in self.memory_index[index]:
                del self.memory_index[index][label]
                # 如果该索引下没有其他记忆，删除整个索引
                if not self.memory_index[index]:
                    del self.memory_index[index]

        self.save_to_json()

    def update_memory_content(self, index: int, label: str, new_content: Any) -> bool:
        """
        更新指定记忆的内容

        Args:
            index: 索引编号
            label: 标签
            new_content: 新的内容

        Returns:
            更新是否成功
        """
        try:
            if index in self.memory_index and label in self.memory_index[index]:
                self.memory_index[index][label]["content"] = new_content
                self.memory_index[index][label]["last_modified"] = datetime.datetime.now().isoformat()
                self.save_to_json()
                return True
            return False
        except Exception as e:
            print(f"更新记忆内容失败: {e}")
            return False


class EnhancedMemorySystem(MemorySystem):
    """
    增强的记忆体系统，继承自MemorySystem
    支持按组和模型存储API调用结果，采用扁平化结构
    优化存储结构：input_text和image_path提升到组级别，减少重复存储
    """

    def __init__(self, storage_file: str = "memory_data.json"):
        """
        初始化增强记忆体系统

        Args:
            storage_file: JSON存储文件名
        """
        super().__init__(storage_file)
        self.model_mapping = {}  # 映射api_name到模型名
        self.flat_data = {}  # 扁平化数据存储

        # 如果存储文件已存在，则加载数据
        if os.path.exists(self.storage_file):
            self.load_flat_data()

    def load_flat_data(self):
        """从文件加载扁平化数据"""
        try:
            with open(self.storage_file, 'r', encoding='utf-8') as f:
                self.flat_data = json.load(f)

            # 同时将数据加载到父类的索引结构中，保持兼容性
            self._sync_to_parent_structure()
        except Exception as e:
            print(f"加载扁平化数据失败: {e}, 将创建新的数据存储")
            self.flat_data = {}

    def _sync_to_parent_structure(self):
        """将扁平化数据同步到父类的索引结构中"""
        self.memory_index.clear()

        group_index = 1000  # 为每个组分配一个基础索引
        for group_name, group_data in self.flat_data.items():
            # 获取组的输入数据
            input_text = group_data.get("input_text", "")
            image_path = group_data.get("image_path", "")
            models = group_data.get("models", {})

            for model_index, (model_name, model_data) in enumerate(models.items()):
                # 为每个模型记录创建唯一索引
                unique_index = group_index + model_index

                # 存储到父类结构中（包含组级别的输入数据）
                self.add_memory(
                    unique_index,
                    model_name,
                    {
                        "group_name": group_name,
                        "input_text": input_text,
                        "image_path": image_path,
                        "model_data": model_data
                    }
                )

    def save_flat_data(self):
        """保存扁平化数据到文件"""
        try:
            with open(self.storage_file, 'w', encoding='utf-8') as f:
                json.dump(self.flat_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存扁平化数据失败: {e}")

    def map_api_name_to_model(self, api_name: str, model_name: str) -> None:
        """
        建立api_name到模型名的映射关系

        Args:
            api_name: API名称（如"multi_modal_group_api_0"）
            model_name: 对应的模型名称
        """
        self.model_mapping[api_name] = model_name

    def get_model_by_api_name(self, api_name: str) -> str:
        """
        通过api_name获取对应的模型名

        Args:
            api_name: API名称

        Returns:
            模型名称
        """
        return self.model_mapping.get(api_name, "unknown_model")

    def store_api_result(self, group_name: str, model_name: str, api_response, input_data) -> None:
        """
        存储API调用结果到记忆系统

        Args:
            group_name: 组名称（最高父级）
            model_name: 模型名称（次父级）
            api_response: API响应对象
            input_data: 输入数据对象
        """
        # 如果组不存在，创建组结构（包含input_text和image_path）
        if group_name not in self.flat_data:
            self.flat_data[group_name] = {
                "input_text": getattr(input_data, 'text', ''),
                "image_path": getattr(input_data, 'image_path', ''),
                "models": {}  # 存储各个模型的结果
            }
        else:
            # 如果组已存在，确保models字段存在
            if "models" not in self.flat_data[group_name]:
                self.flat_data[group_name]["models"] = {}

        # 准备模型级别的数据（不包含input_text和image_path）
        model_data = {
            "success": api_response.success,
            "content": api_response.content,
            "error_message": api_response.error_message,
            "response_time": api_response.response_time,
            "api_name": api_response.api_name,
            "processing_type": api_response.processing_type,
        }

        # 存储到扁平化结构
        self.flat_data[group_name]["models"][model_name] = model_data

        # 同时同步到父类结构
        self._sync_to_parent_structure()

        # 建立api_name到模型名的映射
        if hasattr(api_response, 'api_name'):
            self.map_api_name_to_model(api_response.api_name, model_name)

        # 保存到文件
        self.save_flat_data()

    def get_api_results_by_group(self, group_name: str) -> Dict[str, Any]:
        """
        获取指定组的所有API结果（包含组级别信息和模型结果）

        Args:
            group_name: 组名称

        Returns:
            该组的完整信息，包括输入数据和所有模型结果
        """
        return self.flat_data.get(group_name, {})

    def get_models_results(self, group_name: str) -> Dict[str, Any]:
        """
        获取指定组的所有模型结果（不包含组级别输入数据）

        Args:
            group_name: 组名称

        Returns:
            该组下所有模型的API结果
        """
        group_data = self.flat_data.get(group_name, {})
        return group_data.get("models", {})

    def get_group_input_data(self, group_name: str) -> Dict[str, Any]:
        """
        获取指定组的输入数据

        Args:
            group_name: 组名称

        Returns:
            组的输入数据（input_text和image_path）
        """
        group_data = self.flat_data.get(group_name, {})
        return {
            "input_text": group_data.get("input_text", ""),
            "image_path": group_data.get("image_path", "")
        }

    def get_latest_model_result(self, group_name: str, model_name: str) -> Optional[Dict[str, Any]]:
        """
        获取指定模型的最新结果

        Args:
            group_name: 组名称
            model_name: 模型名称

        Returns:
            最新API结果字典
        """
        models_data = self.get_models_results(group_name)
        return models_data.get(model_name)

    def get_complete_model_result(self, group_name: str, model_name: str) -> Optional[Dict[str, Any]]:
        """
        获取指定模型的完整结果（包含组级别输入数据）

        Args:
            group_name: 组名称
            model_name: 模型名称

        Returns:
            包含输入数据和模型响应的完整结果
        """
        group_data = self.flat_data.get(group_name, {})
        model_data = group_data.get("models", {}).get(model_name)

        if not model_data:
            return None

        # 合并组级别输入数据和模型级别响应数据
        return {
            "input_text": group_data.get("input_text", ""),
            "image_path": group_data.get("image_path", ""),
            **model_data
        }

    def get_model_history(self, group_name: str, model_name: str) -> List[Dict[str, Any]]:
        """
        获取指定模型的完整调用历史
        注意：这个简化版本只存储最新结果，历史功能需要额外实现

        Args:
            group_name: 组名称
            model_name: 模型名称

        Returns:
            该模型的历史调用记录列表
        """
        # 当前版本只返回最新结果
        latest = self.get_latest_model_result(group_name, model_name)
        return [latest] if latest else []

    def export_memory_report(self, report_file: str = "memory_report.txt") -> None:
        """
        导出记忆报告

        Args:
            report_file: 报告文件名
        """
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write("=== API调用记忆报告 ===\n")
                f.write(f"生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

                for group_name, group_data in self.flat_data.items():
                    f.write(f"组: {group_name}\n")
                    f.write(f"输入文本: {group_data.get('input_text', '')}\n")
                    f.write(f"图片路径: {group_data.get('image_path', '')}\n")
                    f.write("=" * 50 + "\n")

                    models = group_data.get("models", {})
                    for model_name, result in models.items():
                        f.write(f"  模型: {model_name}\n")
                        f.write(f"  成功: {result['success']}\n")
                        f.write(f"  响应时间: {result['response_time']:.2f}s\n")
                        f.write(f"  处理类型: {result['processing_type']}\n")
                        f.write(f"  API名称: {result['api_name']}\n")

                        if result['success']:
                            content_preview = result['content'][:100] + "..." if len(result['content']) > 100 else \
                                result['content']
                            f.write(f"  内容预览: {content_preview}\n")
                        else:
                            f.write(f"  错误信息: {result['error_message']}\n")

                        f.write("-" * 30 + "\n")

                    f.write("\n")

            print(f"记忆报告已导出到: {report_file}")
        except Exception as e:
            print(f"导出记忆报告失败: {e}")

    def get_flat_data(self) -> Dict[str, Any]:
        """
        获取扁平化数据

        Returns:
            完整的扁平化数据
        """
        return self.flat_data


# 使用示例和测试代码
def demo_memory_system():
    """演示记忆体系统的使用方法"""

    # 创建记忆体实例
    memory_system = MemorySystem("demo_memory.json")

    print("=== 记忆体系统演示 ===\n")

    # 示例1: 添加解题过程和答案
    print("1. 添加索引1的记忆...")

    # 先添加解题过程（索引1->标签1）
    memory_system.add_memory(
        index=1,
        label="解题过程",
        content="首先分析题目，确定使用二次方程求根公式..."
    )

    # 添加答案（索引1->标签0）
    memory_system.add_memory(
        index=1,
        label="答案",
        content="最终答案: x = 2 或 x = 3"
    )

    # 示例2: 添加另一个索引的记忆
    print("2. 添加索引2的记忆...")
    memory_system.add_memory(2, "答案", "这是第二个问题的答案")
    memory_system.add_memory(2, "解题过程", "这是第二个问题的详细推导过程")

    # 示例3: 添加索引3的记忆
    print("3. 添加索引3的记忆...")
    memory_system.add_memory(3, "答案", "第三个问题的答案是42")
    memory_system.add_memory(3, "备注", "这个问题需要特别注意边界条件")

    # 示例4: 检索单个记忆
    print("4. 检索记忆...")
    memory_1_answer = memory_system.get_memory(1, "答案")
    if memory_1_answer:
        print(f"索引1-标签答案: {memory_1_answer['content']}")

    # 示例5: 获取索引1的所有记忆
    print("5. 索引1的所有记忆:")
    index1_memories = memory_system.get_index_memories(1)
    for label, memory in index1_memories.items():
        print(f"  标签{label}: {memory['content']}")

    # 示例6: 增强的取出逻辑 - 获取所有索引中标签为"答案"的记忆
    print("6. 所有索引中标签为'答案'的记忆:")
    all_answers = memory_system.get_memories_by_label("答案")
    for answer in all_answers:
        print(f"  索引{answer['index']}: {answer['content']}")

    # 示例7: 在特定索引范围内获取记忆
    print("7. 在索引1和2中获取标签为'解题过程'的记忆:")
    specific_processes = memory_system.get_memories_by_label("解题过程", indexes=[1, 2])
    for process in specific_processes:
        print(f"  索引{process['index']}: {process['content']}")

    # 示例8: 批量获取多个标签的记忆
    print("8. 批量获取'答案'和'解题过程'的记忆:")
    batch_memories = memory_system.get_memories_by_labels(["答案", "解题过程"])
    for label, memories in batch_memories.items():
        print(f"  标签{label}:")
        for memory in memories:
            print(f"    索引{memory['index']}: {memory['content']}")

    # 示例9: 搜索记忆
    print("9. 搜索包含'答案'的记忆:")
    results = memory_system.search_memories("答案")
    for result in results:
        print(f"  索引{result['index']}-标签{result['label']}: {result['content']}")

    # 示例10: 显示时间线
    print("10. 记忆时间线:")
    timeline = memory_system.get_timeline()
    for memory in timeline:
        print(f"  时间{memory['created_at']}: 索引{memory['index']}-标签{memory['label']}")

    # 示例11: 获取包含特定标签的所有索引
    print("11. 包含'答案'标签的索引:")
    answer_indexes = memory_system.get_indexes_with_label("答案")
    print(f"  索引: {answer_indexes}")

    # 示例12: 获取所有标签
    print("12. 所有标签:")
    all_labels = memory_system.get_all_labels()
    print(f"  标签: {all_labels}")

    # 示例13: 在特定索引范围内获取标签
    print("13. 索引1和2中的所有标签:")
    specific_labels = memory_system.get_all_labels(indexes=[1, 2])
    print(f"  标签: {specific_labels}")

    # 示例14: 导出报告
    memory_system.export_memory_report("demo_report.txt")
    print("14. 记忆报告已导出到 demo_report.txt")

    # 显示统计信息
    print(f"\n=== 系统统计 ===")
    print(f"总记忆数量: {memory_system.get_total_memory_count()}")
    print(f"使用的索引: {list(memory_system.memory_index.keys())}")

    return memory_system


if __name__ == "__main__":
    # 运行演示
    demo = demo_memory_system()


# 演示代码
async def demo_enhanced_memory_system():
    """演示增强记忆体系统的使用方法"""

    # 创建增强记忆体实例
    memory_system = EnhancedMemorySystem("enhanced_memory.json")

    print("=== 增强记忆体系统演示 ===\n")

    # 模拟API响应和输入数据类
    class MockAPIResponse:
        def __init__(self, success, content, error_message, response_time, api_name, processing_type):
            self.success = success
            self.content = content
            self.error_message = error_message
            self.response_time = response_time
            self.api_name = api_name
            self.processing_type = processing_type

    class MockInputData:
        def __init__(self, text, image_path):
            self.text = text
            self.image_path = image_path

    # 示例：存储API调用结果
    print("1. 存储API调用结果...")

    # 模拟第一个模型的API响应
    api_response1 = MockAPIResponse(
        success=True,
        content="根据您提供的图片，图中显示的是一个城市天际线，包含多座摩天大楼...",
        error_message=None,
        response_time=2.34,
        api_name="multi_modal_group_api_0",
        processing_type=0
    )

    input_data1 = MockInputData(
        text="请分析这张图片中的内容并描述主要特征",
        image_path="fake_image_binary_data.png"
    )

    # 存储第一个模型的结果
    memory_system.store_api_result("multi_modal_group", "Qwen2.5-72B-Instruct", api_response1, input_data1)

    # 模拟第二个模型的API响应（失败情况）
    api_response2 = MockAPIResponse(
        success=False,
        content=None,
        error_message="API请求超时",
        response_time=5.67,
        api_name="multi_modal_group_api_1",
        processing_type=1
    )

    input_data2 = MockInputData(
        text="请分析这张图片中的内容并描述主要特征",  # 相同输入文本
        image_path="fake_image_binary_data.png"  # 相同图片路径
    )

    # 存储第二个模型的结果
    memory_system.store_api_result("multi_modal_group", "gpt-4o-mini", api_response2, input_data2)

    # 示例：检索数据
    print("2. 检索存储的数据...")

    # 获取整个组的结果
    group_results = memory_system.get_api_results_by_group("multi_modal_group")
    print(f"组 'multi_modal_group' 包含 {len(group_results.get('models', {}))} 个模型的结果")

    # 获取组的输入数据
    input_data = memory_system.get_group_input_data("multi_modal_group")
    print(f"组输入数据: 文本='{input_data['input_text']}', 图片路径='{input_data['image_path']}'")

    # 获取特定模型的完整结果
    complete_result = memory_system.get_complete_model_result("multi_modal_group", "Qwen2.5-72B-Instruct")
    if complete_result:
        print(f"\n模型 Qwen2.5-72B-Instruct 的完整结果:")
        print(f"  输入文本: {complete_result['input_text']}")
        print(f"  图片路径: {complete_result['image_path']}")
        print(f"  成功: {complete_result['success']}")
        print(f"  响应时间: {complete_result['response_time']:.2f}s")
        print(f"  内容预览: {complete_result['content'][:50]}...")

    # 示例：显示存储的JSON结构
    print("\n3. 存储的JSON数据结构:")
    flat_data = memory_system.get_flat_data()
    print(json.dumps(flat_data, ensure_ascii=False, indent=2))

    # 示例：导出报告
    memory_system.export_memory_report("enhanced_memory_report.txt")
    print("\n4. 记忆报告已导出到 enhanced_memory_report.txt")

    # 演示父类功能仍然可用
    print("\n5. 父类功能演示:")
    all_memories = memory_system.get_all_memories()
    print(f"父类索引中的记忆数量: {memory_system.get_total_memory_count()}")

    return memory_system


if __name__ == "__main__":
    # 运行演示
    import asyncio

    asyncio.run(demo_enhanced_memory_system())
