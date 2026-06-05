"""
表格数据增强脚本
目标：将 manual_tables.json 中的表格数据转换为包含上下文的 Markdown 格式文本
每块数据附带来源 Metadata（文件名、页码、表格标题）
"""

import json
from pathlib import Path
from typing import List, Dict


# ==================== 配置区 ====================
PROJECT_ROOT = Path(__file__).parent.parent
INPUT_JSON = PROJECT_ROOT / "backend" / "output" / "manual_tables.json"
OUTPUT_JSON = PROJECT_ROOT / "backend" / "output" / "enhanced_chunks.json"
# ================================================


def table_to_markdown(table: Dict) -> str:
    """
    将单个表格转换为 Markdown 格式
    
    Args:
        table: 表格数据字典
        
    Returns:
        Markdown 格式的表格字符串
    """
    lines = []
    
    # 表格标题
    page = table.get('page', '未知')
    table_idx = table.get('table_index', 1)
    lines.append(f"## 维修手册 - 第 {page} 页，表格 {table_idx}")
    lines.append("")
    
    # 表头
    headers = table.get('headers', [])
    if not headers:
        return ""
    
    # Markdown 表头
    header_row = "| " + " | ".join(headers) + " |"
    lines.append(header_row)
    
    # 分隔行
    separator = "| " + " | ".join(["---"] * len(headers)) + " |"
    lines.append(separator)
    
    # 数据行
    rows = table.get('rows', [])
    for row in rows:
        data = row.get('data', {})
        cells = [str(data.get(h, "")) for h in headers]
        data_row = "| " + " | ".join(cells) + " |"
        lines.append(data_row)
    
    return "\n".join(lines)


def table_to_contextual_text(table: Dict) -> List[Dict]:
    """
    将表格转换为带上下文的文本块（每行一个 chunk）
    
    Args:
        table: 表格数据字典
        
    Returns:
        文本块列表，每个 chunk 包含 text 和 metadata
    """
    chunks = []
    page = table.get('page', '未知')
    table_idx = table.get('table_index', 1)
    headers = table.get('headers', [])
    rows = table.get('rows', [])
    
    # 表头描述
    header_desc = "、".join(headers)
    
    for row in rows:
        data = row.get('data', {})
        row_idx = row.get('row_index', '未知')
        
        # 构建自然语言描述
        # 示例: "在《维修手册》第10页的危险符号表中，【注意/ATENCIÓN】这个符号的说明是【注意安全】"
        text_parts = [f"在《维修手册》第{page}页的表格{table_idx}中"]
        
        # 添加每列的键值对
        for header in headers:
            value = data.get(header, "")
            if value:
                text_parts.append(f"【{header}】是【{value}】")
        
        full_text = "，".join(text_parts) + "。"
        
        # 构建 metadata（ChromaDB 不支持 dict 类型，需要转换）
        metadata = {
            'source': '维修手册西语.pdf',
            'page': page,
            'table_index': table_idx,
            'row_index': row_idx,
            'chunk_type': 'table_row',
            'headers': ', '.join(headers)  # 将 list 转为字符串
        }
        
        chunks.append({
            'text': full_text,
            'metadata': metadata
        })
    
    return chunks


def enhance_all_tables(tables_data: Dict) -> List[Dict]:
    """
    增强所有表格数据
    
    Args:
        tables_data: 包含所有表格的 JSON 数据
        
    Returns:
        增强后的文本块列表
    """
    all_chunks = []
    tables = tables_data.get('tables', [])
    
    for table in tables:
        # 方式 1: 完整 Markdown 表格
        markdown = table_to_markdown(table)
        if markdown:
            all_chunks.append({
                'text': markdown,
                'metadata': {
                    'source': '维修手册西语.pdf',
                    'page': table.get('page', '未知'),
                    'table_index': table.get('table_index', 1),
                    'chunk_type': 'full_table_markdown',
                    'headers': ', '.join(table.get('headers', [])),  # 转为字符串
                    'row_count': table.get('row_count', 0)
                }
            })
        
        # 方式 2: 逐行上下文文本
        row_chunks = table_to_contextual_text(table)
        all_chunks.extend(row_chunks)
    
    return all_chunks


def main():
    print("=" * 60)
    print("表格数据增强工具")
    print("=" * 60)
    
    if not INPUT_JSON.exists():
        print(f"错误：找不到输入文件 {INPUT_JSON}")
        print("请先运行 parse_manual_tables.py 生成表格数据")
        return
    
    # 读取表格数据
    with open(INPUT_JSON, 'r', encoding='utf-8') as f:
        tables_data = json.load(f)
    
    total_tables = tables_data.get('total_tables_found', 0)
    print(f"输入文件: {INPUT_JSON}")
    print(f"表格总数: {total_tables}")
    
    # 增强数据
    print("\n开始数据增强...")
    enhanced_chunks = enhance_all_tables(tables_data)
    
    # 保存结果
    output_data = {
        'source': '维修手册西语.pdf',
        'total_tables': total_tables,
        'total_chunks': len(enhanced_chunks),
        'chunks': enhanced_chunks
    }
    
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n数据增强完成")
    print(f"{'='*60}")
    print(f"输出文件: {OUTPUT_JSON}")
    print(f"生成文本块: {len(enhanced_chunks)} 个")
    
    # 展示示例
    if enhanced_chunks:
        print(f"\n{'='*60}")
        print("示例文本块（前 3 个）:")
        print(f"{'='*60}")
        
        for i, chunk in enumerate(enhanced_chunks[:3]):
            print(f"\n--- Chunk {i+1} ---")
            print(f"类型: {chunk['metadata']['chunk_type']}")
            print(f"来源: 第 {chunk['metadata']['page']} 页")
            print(f"文本: {chunk['text'][:200]}...")


if __name__ == "__main__":
    main()
