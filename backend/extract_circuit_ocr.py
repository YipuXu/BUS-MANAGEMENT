"""
电路图坐标提取主脚本
使用策略模式，支持灵活切换 OCR 引擎
当前使用 PyMuPDF 内置文本提取（无需 OCR）
"""

import os
import json
import re
from io import BytesIO
from pathlib import Path
import fitz  # PyMuPDF
from PIL import Image
import cv2
import numpy as np

from ocr_strategy import (
    OCRStrategy,
    OCRResult,
    PyMuPDFTextStrategy,
    LocalTesseractStrategy,
    AWSTextractStrategy,
    RapidOCRStrategy,
    GoogleVisionStrategy
)


# ==================== 配置区 ====================
PROJECT_ROOT = Path(__file__).parent.parent
PDF_PATH = PROJECT_ROOT / "SWB6128EV56电路图（松江）.pdf"
OUTPUT_DIR = PROJECT_ROOT / "backend" / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# 选择 OCR 策略（切换这里即可更换引擎）
# 可选: "pymupdf", "tesseract", "aws_textract", "rapidocr", "google_vision"
OCR_STRATEGY = "rapidocr"  # 当前使用 RapidOCR（免费本地，识别率高）
# ================================================


def create_ocr_strategy(strategy_name: str) -> OCRStrategy:
    """工厂方法：根据名称创建 OCR 策略实例"""
    strategies = {
        "pymupdf": PyMuPDFTextStrategy(zoom=3.0),
        "tesseract": LocalTesseractStrategy(lang='chi_sim+eng'),
        "aws_textract": AWSTextractStrategy(),
        "rapidocr": RapidOCRStrategy(lang='ch'),
        "google_vision": GoogleVisionStrategy(),
    }
    
    if strategy_name not in strategies:
        raise ValueError(f"不支持的 OCR 策略: {strategy_name}")
    
    return strategies[strategy_name]


def filter_key_components(results: list) -> list:
    """筛选关键组件（插头、线号等）"""
    key_items = []
    for item in results:
        if item.item_type in ['connector', 'wire_number']:
            key_items.append(item)
        elif item.confidence > 70 and len(item.text) <= 10:
            key_items.append(item)
    return key_items


def save_annotated_image(page, items: list, page_num: int, zoom: float = 3.0):
    """保存带标注的 PDF 页面图片（用于验证）"""
    # 渲染 PDF 页面为图片
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    img_data = pix.tobytes("png")
    img = Image.open(BytesIO(img_data))
    img_array = np.array(img)
    img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    for item in items:
        bbox = item.bbox
        x = bbox['pixel_x']
        y = bbox['pixel_y']
        w = bbox['pixel_width']
        h = bbox['pixel_height']
        
        # 绘制矩形框
        color = (0, 255, 0) if item.item_type == 'connector' else (255, 0, 0)
        cv2.rectangle(img_cv, (x, y), (x + w, y + h), color, 2)
        
        # 添加文字标签
        label = f"{item.text} ({item.item_type})"
        cv2.putText(img_cv, label, (x, y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    output_path = OUTPUT_DIR / f"page_{page_num}_annotated.jpg"
    cv2.imwrite(str(output_path), img_cv)
    print(f"  标注图片已保存: {output_path}")


def main():
    print("=" * 60)
    print("电路图坐标提取工具 - 策略模式版")
    print("=" * 60)
    
    if not PDF_PATH.exists():
        print(f"错误：找不到 PDF 文件 {PDF_PATH}")
        return
    
    # 创建 OCR 策略
    strategy = create_ocr_strategy(OCR_STRATEGY)
    print(f"\n使用 OCR 引擎: {strategy.get_engine_name()}")
    print(f"PDF 文件: {PDF_PATH}")
    print()
    
    # 打开 PDF
    doc = fitz.open(PDF_PATH)
    print(f"总页数: {len(doc)}")
    
    # 测试第 3 页（电路图扫描件）
    test_pages = [2]  # 索引从 0 开始，第 3 页索引为 2
    all_results = []
    
    for page_idx in test_pages:
        page_num = page_idx + 1
        page = doc[page_idx]
        
        print(f"\n{'='*40}")
        print(f"处理第 {page_num} 页")
        print(f"{'='*40}")
        
        # 使用策略提取文字
        if isinstance(strategy, PyMuPDFTextStrategy):
            # PyMuPDF 策略直接从页面提取
            results = strategy.extract_from_page(page, page_num)
        else:
            # 其他策略需要先将页面转为图片
            mat = fitz.Matrix(3.0, 3.0)
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            image = Image.open(BytesIO(img_data))
            results = strategy.extract_from_image(image, page_num)
        
        # 筛选关键组件
        key_items = filter_key_components(results)
        
        # 构建页面结果
        page_result = {
            'page': page_num,
            'page_size': {
                'width': page.rect.width,
                'height': page.rect.height
            },
            'total_items': len(results),
            'key_items_count': len(key_items),
            'items': [item.to_dict() for item in key_items]
        }
        all_results.append(page_result)
        
        # 保存标注图片
        save_annotated_image(page, key_items, page_num, zoom=3.0)
    
    doc.close()
    
    # 保存 JSON 结果
    output = {
        'source': str(PDF_PATH),
        'ocr_engine': strategy.get_engine_name(),
        'total_pages_processed': len(all_results),
        'pages': all_results
    }
    
    output_path = OUTPUT_DIR / "circuit_ocr_results.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*60}")
    print("提取完成")
    print(f"{'='*60}")
    print(f"结果已保存到: {output_path}")
    
    # 打印摘要
    for page in all_results:
        connectors = [i for i in page['items'] if i['type'] == 'connector']
        wires = [i for i in page['items'] if i['type'] == 'wire_number']
        print(f"\n第 {page['page']} 页摘要:")
        print(f"  总文本块: {page['total_items']}")
        print(f"  关键组件: {page['key_items_count']}")
        print(f"  插头编号: {len(connectors)} 个")
        print(f"  线号: {len(wires)} 个")
        
        # 显示前 10 个插头编号示例
        if connectors:
            print(f"\n  插头编号示例（前 10 个）:")
            for c in connectors[:10]:
                print(f"    {c['text']} - 坐标: ({c['bbox']['x']:.4f}, {c['bbox']['y']:.4f})")
        
        # 显示前 10 个线号示例
        if wires:
            print(f"\n  线号示例（前 10 个）:")
            for w in wires[:10]:
                print(f"    {w['text']} - 坐标: ({w['bbox']['x']:.4f}, {w['bbox']['y']:.4f})")


if __name__ == "__main__":
    main()
