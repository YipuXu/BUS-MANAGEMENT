"""
维修手册表格解析脚本
目标：从《维修手册西语.pdf》中提取结构化表格，保留行列对应关系
使用 pdfplumber 进行精确表格提取
"""

import json
from pathlib import Path
import pdfplumber


# ==================== 配置区 ====================
PROJECT_ROOT = Path(__file__).parent.parent
PDF_PATH = PROJECT_ROOT / "维修手册西语.pdf"
OUTPUT_DIR = PROJECT_ROOT / "backend" / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# 测试页码（从 1 开始，None 表示全部页面）
TEST_PAGES = None  # 例如: [5, 6, 7] 或 None
# ================================================


def extract_tables_from_page(page, page_num: int) -> list:
    """
    从单页 PDF 中提取所有表格
    
    Args:
        page: pdfplumber Page 对象
        page_num: 页码
        
    Returns:
        表格列表，每个表格包含表头和行数据
    """
    tables = page.extract_tables()
    
    if not tables:
        return []
    
    result = []
    for table_idx, table in enumerate(tables):
        if not table or len(table) == 0:
            continue
        
        # 第一行作为表头
        headers = table[0]
        # 清理表头（去除 None 和空字符串）
        headers = [h.strip() if h else f"列{idx}" for idx, h in enumerate(headers)]
        
        # 数据行
        rows = []
        for row_idx, row in enumerate(table[1:], start=2):
            # 清理行数据
            cleaned_row = {}
            for col_idx, cell in enumerate(row):
                col_name = headers[col_idx] if col_idx < len(headers) else f"列{col_idx}"
                cleaned_row[col_name] = cell.strip() if cell else ""
            
            rows.append({
                'row_index': row_idx,
                'data': cleaned_row
            })
        
        table_data = {
            'table_index': table_idx + 1,
            'page': page_num,
            'headers': headers,
            'row_count': len(rows),
            'rows': rows
        }
        result.append(table_data)
    
    return result


def detect_table_regions(page) -> list:
    """
    检测页面中的表格区域（用于调试）
    
    Returns:
        表格边界框列表
    """
    tables = page.find_tables()
    regions = []
    
    for table in tables:
        bbox = table.bbox
        regions.append({
            'x0': round(bbox[0], 2),
            'y0': round(bbox[1], 2),
            'x1': round(bbox[2], 2),
            'y1': round(bbox[3], 2),
            'width': round(bbox[2] - bbox[0], 2),
            'height': round(bbox[3] - bbox[1], 2)
        })
    
    return regions


def main():
    print("=" * 60)
    print("维修手册表格解析工具")
    print("=" * 60)
    
    if not PDF_PATH.exists():
        print(f"错误：找不到 PDF 文件 {PDF_PATH}")
        return
    
    print(f"PDF 文件: {PDF_PATH}")
    
    with pdfplumber.open(PDF_PATH) as pdf:
        total_pages = len(pdf.pages)
        print(f"总页数: {total_pages}")
        
        # 确定要处理的页面
        if TEST_PAGES:
            pages_to_process = [p - 1 for p in TEST_PAGES if p <= total_pages]
        else:
            pages_to_process = range(total_pages)
        
        all_tables = []
        total_tables_found = 0
        
        for page_idx in pages_to_process:
            page_num = page_idx + 1
            page = pdf.pages[page_idx]
            
            print(f"\n{'='*40}")
            print(f"处理第 {page_num} 页")
            print(f"{'='*40}")
            
            # 检测表格区域
            regions = detect_table_regions(page)
            print(f"  检测到 {len(regions)} 个表格区域")
            
            # 提取表格数据
            tables = extract_tables_from_page(page, page_num)
            
            if tables:
                total_tables_found += len(tables)
                print(f"  成功提取 {len(tables)} 个表格")
                
                for table in tables:
                    print(f"    表格 {table['table_index']}: "
                          f"{table['row_count']} 行, "
                          f"{len(table['headers'])} 列")
                    print(f"    表头: {table['headers']}")
                
                all_tables.extend(tables)
            else:
                print(f"  未找到可提取的表格")
        
        # 保存结果
        output = {
            'source': str(PDF_PATH),
            'total_pages_in_pdf': total_pages,
            'pages_processed': len(pages_to_process),
            'total_tables_found': total_tables_found,
            'tables': all_tables
        }
        
        output_path = OUTPUT_DIR / "manual_tables.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        
        print(f"\n{'='*60}")
        print("提取完成")
        print(f"{'='*60}")
        print(f"结果已保存到: {output_path}")
        print(f"共提取 {total_tables_found} 个表格")
        
        # 展示第一个表格的示例数据
        if all_tables:
            first_table = all_tables[0]
            print(f"\n示例表格（第 {first_table['page']} 页，表格 {first_table['table_index']}）:")
            print(f"表头: {first_table['headers']}")
            print(f"行数: {first_table['row_count']}")
            print(f"\n前 3 行数据:")
            for row in first_table['rows'][:3]:
                print(f"  行 {row['row_index']}: {row['data']}")


if __name__ == "__main__":
    main()
