"""
OCR 策略接口定义
使用策略模式，支持灵活切换不同的 OCR 引擎
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
from PIL import Image
import fitz  # PyMuPDF


class OCRResult:
    """OCR 识别结果数据结构"""
    
    def __init__(self, text: str, item_type: str, bbox: Dict[str, float], confidence: float = 0.0):
        self.text = text
        self.item_type = item_type  # connector, wire_number, chinese_text, other
        self.bbox = bbox  # 包含归一化坐标和像素坐标
        self.confidence = confidence
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'text': self.text,
            'type': self.item_type,
            'bbox': self.bbox,
            'confidence': self.confidence
        }


class OCRStrategy(ABC):
    """OCR 策略抽象基类"""
    
    @abstractmethod
    def extract_from_image(self, image: Image.Image, page_num: int) -> List[OCRResult]:
        """
        从图片中提取文字及坐标
        
        Args:
            image: PIL Image 对象
            page_num: 页码（用于日志）
            
        Returns:
            OCRResult 列表
        """
        pass
    
    @abstractmethod
    def get_engine_name(self) -> str:
        """返回引擎名称"""
        pass


class LocalTesseractStrategy(OCRStrategy):
    """
    本地 Tesseract OCR 策略
    需要安装 Tesseract-OCR 并配置环境变量
    下载地址：https://github.com/UB-Mannheim/tesseract/wiki
    """
    
    def __init__(self, lang: str = 'chi_sim+eng', config: str = '--oem 3 --psm 6'):
        self.lang = lang
        self.config = config
    
    def extract_from_image(self, image: Image.Image, page_num: int) -> List[OCRResult]:
        import pytesseract
        import re
        
        print(f"[OCR] 使用 Tesseract 对第 {page_num} 页进行识别...")
        
        img_width, img_height = image.size
        
        data = pytesseract.image_to_data(
            image, 
            config=self.config, 
            lang=self.lang,
            output_type=pytesseract.Output.DICT
        )
        
        results = []
        for i in range(len(data['text'])):
            text = data['text'][i].strip()
            if not text:
                continue
            
            x = data['left'][i]
            y = data['top'][i]
            w = data['width'][i]
            h = data['height'][i]
            
            bbox = {
                'x': round(x / img_width, 4),
                'y': round(y / img_height, 4),
                'width': round(w / img_width, 4),
                'height': round(h / img_height, 4),
                'pixel_x': x,
                'pixel_y': y,
                'pixel_width': w,
                'pixel_height': h
            }
            
            item_type = self._classify_text(text)
            
            results.append(OCRResult(
                text=text,
                item_type=item_type,
                bbox=bbox,
                confidence=data['conf'][i]
            ))
        
        print(f"  识别到 {len(results)} 个文本块")
        return results
    
    def _classify_text(self, text: str) -> str:
        import re
        if re.match(r'^[A-Z]{1,3}\d+[A-Z]?$', text):
            return 'connector'
        if re.match(r'^\d{1,4}[A-Z]?$', text):
            return 'wire_number'
        if re.search(r'[\u4e00-\u9fff]', text):
            return 'chinese_text'
        return 'other'
    
    def get_engine_name(self) -> str:
        return 'Tesseract (Local)'


class AWSTextractStrategy(OCRStrategy):
    """
    AWS Textract 云端 OCR 策略
    适用于工业级高精度文档识别
    需要配置 AWS 凭证（Access Key + Secret Key）
    
    配置方式：
    1. 在项目根目录创建 .env 文件，填入：
       AWS_ACCESS_KEY_ID=你的AccessKey
       AWS_SECRET_ACCESS_KEY=你的SecretKey
       AWS_REGION=us-east-1  (或其他区域)
    2. 或者在 ~/.aws/credentials 中配置
    """
    
    def __init__(self, region: str = 'us-east-1'):
        import os
        from dotenv import load_dotenv
        import boto3
        
        # 加载 .env 文件中的环境变量
        load_dotenv()
        
        self.region = os.getenv('AWS_REGION', region)
        self.client = boto3.client(
            'textract',
            region_name=self.region,
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
        )
    
    def extract_from_image(self, image: Image.Image, page_num: int) -> List[OCRResult]:
        import io
        import re
        
        print(f"[OCR] 使用 AWS Textract 对第 {page_num} 页进行识别...")
        
        # 将 PIL Image 转为 bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_bytes = img_byte_arr.getvalue()
        
        img_width, img_height = image.size
        
        # 调用 Textract DetectDocumentText API
        response = self.client.detect_document_text(Document={'Bytes': img_bytes})
        
        results = []
        for block in response.get('Blocks', []):
            if block['BlockType'] != 'LINE':
                continue
            
            text = block.get('Text', '').strip()
            if not text:
                continue
            
            # 获取 Bounding Box（AWS 返回的是归一化坐标 0-1）
            bbox_data = block.get('Geometry', {}).get('BoundingBox', {})
            
            # AWS 的坐标已经是归一化的，直接转换为我们统一的格式
            x = bbox_data.get('Left', 0)
            y = bbox_data.get('Top', 0)
            w = bbox_data.get('Width', 0)
            h = bbox_data.get('Height', 0)
            
            bbox = {
                'x': round(x, 4),
                'y': round(y, 4),
                'width': round(w, 4),
                'height': round(h, 4),
                'pixel_x': int(x * img_width),
                'pixel_y': int(y * img_height),
                'pixel_width': int(w * img_width),
                'pixel_height': int(h * img_height)
            }
            
            # 获取置信度
            confidence = block.get('Confidence', 0)
            
            item_type = self._classify_text(text)
            
            results.append(OCRResult(
                text=text,
                item_type=item_type,
                bbox=bbox,
                confidence=confidence
            ))
        
        print(f"  识别到 {len(results)} 个文本块")
        return results
    
    def _classify_text(self, text: str) -> str:
        import re
        # 插头编号模式：K20D, X10, F15, CN1, PIN24 等
        if re.match(r'^[A-Z]{1,4}\d+[A-Z]?$', text):
            return 'connector'
        # 线号模式：纯数字或带字母的数字组合
        if re.match(r'^\d{1,6}[A-Z]?$', text):
            return 'wire_number'
        # 中文文本
        if re.search(r'[\u4e00-\u9fff]', text):
            return 'chinese_text'
        return 'other'
    
    def get_engine_name(self) -> str:
        return f'AWS Textract (Cloud, {self.region})'


class RapidOCRStrategy(OCRStrategy):
    """
    RapidOCR 策略（基于 PaddleOCR 的轻量本地免费方案）
    优势：
    - 完全免费，本地运行，无需网络
    - 对中文和工业图纸识别率远高于 Tesseract
    - 自动返回 Bounding Box 坐标
    - 轻量级，基于 ONNX Runtime 推理
    
    安装：pip install rapidocr_onnxruntime
    """
    
    def __init__(self, lang: str = 'ch'):
        from rapidocr_onnxruntime import RapidOCR
        
        print("[RapidOCR] 正在加载 OCR 模型（首次加载可能需要下载模型文件）...")
        self.ocr = RapidOCR(lang=lang)
        print("[RapidOCR] 模型加载完成")
    
    def extract_from_image(self, image: Image.Image, page_num: int) -> List[OCRResult]:
        import numpy as np
        
        print(f"[OCR] 使用 RapidOCR 对第 {page_num} 页进行识别...")
        
        # 将 PIL Image 转为 numpy array
        img_array = np.array(image)
        img_width, img_height = image.size
        
        # 调用 RapidOCR
        result, _ = self.ocr(img_array)
        
        results = []
        if result:
            for line in result:
                # line 格式: [bbox, text, confidence]
                # bbox 格式: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                bbox_points = line[0]
                text = line[1].strip()
                confidence = line[2]
                
                if not text:
                    continue
                
                # 计算外接矩形
                xs = [p[0] for p in bbox_points]
                ys = [p[1] for p in bbox_points]
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)
                
                x = x_min
                y = y_min
                w = x_max - x_min
                h = y_max - y_min
                
                bbox = {
                    'x': round(x / img_width, 4),
                    'y': round(y / img_height, 4),
                    'width': round(w / img_width, 4),
                    'height': round(h / img_height, 4),
                    'pixel_x': int(x),
                    'pixel_y': int(y),
                    'pixel_width': int(w),
                    'pixel_height': int(h)
                }
                
                item_type = self._classify_text(text)
                
                results.append(OCRResult(
                    text=text,
                    item_type=item_type,
                    bbox=bbox,
                    confidence=confidence * 100  # RapidOCR 返回 0-1，转为 0-100
                ))
        
        print(f"  识别到 {len(results)} 个文本块")
        return results
    
    def _classify_text(self, text: str) -> str:
        import re
        # 插头编号模式：K20D, X10, F15, CN1, PIN24 等
        if re.match(r'^[A-Z]{1,4}\d+[A-Z]?$', text):
            return 'connector'
        # 线号模式：纯数字或带字母的数字组合
        if re.match(r'^\d{1,6}[A-Z]?$', text):
            return 'wire_number'
        # 中文文本
        if re.search(r'[\u4e00-\u9fff]', text):
            return 'chinese_text'
        return 'other'
    
    def get_engine_name(self) -> str:
        return 'RapidOCR (Local, PaddleOCR-based)'


class GoogleVisionStrategy(OCRStrategy):
    """
    Google Cloud Vision OCR 策略（预留接口）
    适用于复杂场景的高精度识别
    需要配置 Google Cloud 凭证
    """
    
    def __init__(self):
        # self.client = vision.ImageAnnotatorClient()
        pass
    
    def extract_from_image(self, image: Image.Image, page_num: int) -> List[OCRResult]:
        # TODO: 实现 Google Cloud Vision 调用
        raise NotImplementedError("Google Cloud Vision 策略尚未实现")
    
    def get_engine_name(self) -> str:
        return 'Google Cloud Vision (Cloud)'


class PyMuPDFTextStrategy(OCRStrategy):
    """
    PyMuPDF 内置文本提取策略
    适用于 PDF 中已嵌入文字的情况（非扫描件）
    优势：无需 OCR，直接提取矢量文字及精确坐标
    """
    
    def __init__(self, zoom: float = 3.0):
        self.zoom = zoom
    
    def extract_from_page(self, page, page_num: int) -> List[OCRResult]:
        """
        直接从 PDF 页面提取文字及坐标
        
        Args:
            page: PyMuPDF Page 对象
            page_num: 页码
        """
        import re
        
        print(f"[OCR] 使用 PyMuPDF 对第 {page_num} 页进行文本提取...")
        
        # 获取页面尺寸
        page_rect = page.rect
        page_width = page_rect.width
        page_height = page_rect.height
        
        # 提取文字块（包含坐标信息）
        blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
        
        results = []
        for block in blocks:
            if block.get("type") != 0:  # 跳过图片块
                continue
            
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "").strip()
                    if not text:
                        continue
                    
                    # 获取边界框
                    bbox_rect = span.get("bbox", (0, 0, 0, 0))
                    x0, y0, x1, y1 = bbox_rect
                    
                    # 计算像素坐标（考虑 zoom）
                    pixel_x = int(x0 * self.zoom)
                    pixel_y = int(y0 * self.zoom)
                    pixel_w = int((x1 - x0) * self.zoom)
                    pixel_h = int((y1 - y0) * self.zoom)
                    
                    bbox = {
                        'x': round(x0 / page_width, 4),
                        'y': round(y0 / page_height, 4),
                        'width': round((x1 - x0) / page_width, 4),
                        'height': round((y1 - y0) / page_height, 4),
                        'pixel_x': pixel_x,
                        'pixel_y': pixel_y,
                        'pixel_width': pixel_w,
                        'pixel_height': pixel_h
                    }
                    
                    item_type = self._classify_text(text)
                    
                    results.append(OCRResult(
                        text=text,
                        item_type=item_type,
                        bbox=bbox,
                        confidence=100.0  # PDF 内置文字置信度为 100%
                    ))
        
        print(f"  提取到 {len(results)} 个文本块")
        return results
    
    def extract_from_image(self, image: Image.Image, page_num: int) -> List[OCRResult]:
        # 此策略主要用于 PDF 直接提取，不支持纯图片
        raise NotImplementedError("PyMuPDFTextStrategy 仅支持 PDF 页面直接提取")
    
    def _classify_text(self, text: str) -> str:
        import re
        if re.match(r'^[A-Z]{1,3}\d+[A-Z]?$', text):
            return 'connector'
        if re.match(r'^\d{1,4}[A-Z]?$', text):
            return 'wire_number'
        if re.search(r'[\u4e00-\u9fff]', text):
            return 'chinese_text'
        return 'other'
    
    def get_engine_name(self) -> str:
        return 'PyMuPDF (Built-in Text)'
