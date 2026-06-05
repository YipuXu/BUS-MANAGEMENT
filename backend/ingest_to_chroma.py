"""
向量数据库入库脚本
将增强后的表格数据和 OCR 图纸文本存入 ChromaDB
"""

import json
import uuid
from pathlib import Path
import chromadb
from chromadb.config import Settings


# ==================== 配置区 ====================
PROJECT_ROOT = Path(__file__).parent.parent
ENHANCED_JSON = PROJECT_ROOT / "backend" / "output" / "enhanced_chunks.json"
OCR_RESULTS_JSON = PROJECT_ROOT / "backend" / "output" / "circuit_ocr_results.json"
CHROMA_DIR = PROJECT_ROOT / "backend" / "chroma_db"
# ================================================


def create_chroma_client():
    """创建 ChromaDB 客户端（本地持久化存储）"""
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def ensure_collection(client, name: str, description: str):
    """获取或创建集合"""
    try:
        collection = client.get_collection(name=name)
        print(f"  使用已有集合: {name}")
    except:
        collection = client.create_collection(
            name=name,
            metadata={"description": description}
        )
        print(f"  创建新集合: {name}")
    return collection


def load_enhanced_chunks():
    """加载增强后的表格数据"""
    if not ENHANCED_JSON.exists():
        print(f"警告：找不到增强数据文件 {ENHANCED_JSON}")
        return []
    
    with open(ENHANCED_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data.get('chunks', [])


def load_ocr_results():
    """加载 OCR 图纸识别结果"""
    if not OCR_RESULTS_JSON.exists():
        print(f"警告：找不到 OCR 结果文件 {OCR_RESULTS_JSON}")
        return []
    
    with open(OCR_RESULTS_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = data.get('results', [])
    chunks = []
    
    for page_result in results:
        page_num = page_result.get('page', 0)
        ocr_items = page_result.get('ocr_items', [])
        
        # 按类型分组
        connectors = [item for item in ocr_items if item.get('item_type') == 'connector']
        wire_numbers = [item for item in ocr_items if item.get('item_type') == 'wire_number']
        
        # 为插头编号创建 chunk
        for connector in connectors:
            text = connector.get('text', '')
            bbox = connector.get('bbox', {})
            confidence = connector.get('confidence', 0)
            
            chunk_text = (
                f"在《SWB6128EV56电路图（松江）》第{page_num}页中，"
                f"插头编号【{text}】的位置坐标为："
                f"归一化坐标 X={bbox.get('x', 0)}, Y={bbox.get('y', 0)}，"
                f"像素坐标 X={bbox.get('pixel_x', 0)}, Y={bbox.get('pixel_y', 0)}，"
                f"识别置信度 {confidence:.1f}%"
            )
            
            chunks.append({
                'text': chunk_text,
                'metadata': {
                    'source': 'SWB6128EV56电路图（松江）.pdf',
                    'page': page_num,
                    'chunk_type': 'circuit_connector',
                    'connector_name': text,
                    'bbox': bbox,
                    'confidence': confidence
                }
            })
        
        # 为线号创建 chunk
        for wire in wire_numbers:
            text = wire.get('text', '')
            bbox = wire.get('bbox', {})
            confidence = wire.get('confidence', 0)
            
            chunk_text = (
                f"在《SWB6128EV56电路图（松江）》第{page_num}页中，"
                f"线束编号【{text}】的位置坐标为："
                f"归一化坐标 X={bbox.get('x', 0)}, Y={bbox.get('y', 0)}，"
                f"像素坐标 X={bbox.get('pixel_x', 0)}, Y={bbox.get('pixel_y', 0)}，"
                f"识别置信度 {confidence:.1f}%"
            )
            
            chunks.append({
                'text': chunk_text,
                'metadata': {
                    'source': 'SWB6128EV56电路图（松江）.pdf',
                    'page': page_num,
                    'chunk_type': 'circuit_wire_number',
                    'wire_number': text,
                    'bbox': bbox,
                    'confidence': confidence
                }
            })
    
    return chunks


def ingest_to_chroma():
    """主函数：将所有数据存入 ChromaDB"""
    print("=" * 60)
    print("向量数据库入库工具")
    print("=" * 60)
    
    # 创建客户端
    client = create_chroma_client()
    print(f"ChromaDB 存储路径: {CHROMA_DIR}")
    
    # 创建集合
    manual_collection = ensure_collection(
        client,
        "maintenance_manual",
        "维修手册表格数据（中西双语）"
    )
    
    circuit_collection = ensure_collection(
        client,
        "circuit_diagram",
        "电路图 OCR 坐标数据"
    )
    
    # 加载数据
    print("\n加载增强数据...")
    manual_chunks = load_enhanced_chunks()
    print(f"  维修手册文本块: {len(manual_chunks)} 个")
    
    circuit_chunks = load_ocr_results()
    print(f"  电路图文本块: {len(circuit_chunks)} 个")
    
    # 存入维修手册数据
    if manual_chunks:
        print(f"\n存入维修手册数据到 ChromaDB...")
        
        ids = [str(uuid.uuid4()) for _ in manual_chunks]
        documents = [chunk['text'] for chunk in manual_chunks]
        metadatas = [chunk['metadata'] for chunk in manual_chunks]
        
        # 分批存入（ChromaDB 限制）
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            batch_end = min(i + batch_size, len(documents))
            manual_collection.add(
                ids=ids[i:batch_end],
                documents=documents[i:batch_end],
                metadatas=metadatas[i:batch_end]
            )
            print(f"  已存入 {batch_end}/{len(documents)} 条")
        
        print(f"  维修手册数据入库完成: {len(documents)} 条")
    
    # 存入电路图数据
    if circuit_chunks:
        print(f"\n存入电路图数据到 ChromaDB...")
        
        ids = [str(uuid.uuid4()) for _ in circuit_chunks]
        documents = [chunk['text'] for chunk in circuit_chunks]
        metadatas = [chunk['metadata'] for chunk in circuit_chunks]
        
        # 分批存入
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            batch_end = min(i + batch_size, len(documents))
            circuit_collection.add(
                ids=ids[i:batch_end],
                documents=documents[i:batch_end],
                metadatas=metadatas[i:batch_end]
            )
            print(f"  已存入 {batch_end}/{len(documents)} 条")
        
        print(f"  电路图数据入库完成: {len(documents)} 条")
    
    # 统计信息
    print(f"\n{'='*60}")
    print("入库完成")
    print(f"{'='*60}")
    print(f"维修手册集合: {manual_collection.count()} 条记录")
    print(f"电路图集合: {circuit_collection.count()} 条记录")
    print(f"总计: {manual_collection.count() + circuit_collection.count()} 条记录")


def test_query():
    """测试查询功能"""
    print(f"\n{'='*60}")
    print("测试向量检索")
    print(f"{'='*60}")
    
    client = create_chroma_client()
    
    try:
        manual_collection = client.get_collection("maintenance_manual")
        
        # 测试查询
        test_queries = [
            "危险符号注意是什么意思",
            "扭矩参数",
            "故障排查步骤"
        ]
        
        for query in test_queries:
            print(f"\n查询: '{query}'")
            results = manual_collection.query(
                query_texts=[query],
                n_results=2
            )
            
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    print(f"  结果 {i+1}: {doc[:150]}...")
            else:
                print("  无结果")
    
    except Exception as e:
        print(f"查询测试失败: {e}")


if __name__ == "__main__":
    # 入库
    ingest_to_chroma()
    
    # 测试查询
    test_query()
